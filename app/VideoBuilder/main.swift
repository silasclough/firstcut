// FirstCut: the Mac app. A window (with the system's see-through sidebar material behind the page) onto the FirstCut engine,
// which this app starts and keeps running in the background (closing the window never stops a video; quitting asks first).
//
// Inside the app (made by scripts/package_videobuilder.sh):
//   Contents/Resources/app/   the engine (Python), its settings (config/app.yaml), bootstrap.sh, requirements-app.txt
//   Contents/Resources/bin/   uv (makes the app's own Python), ffmpeg, ffprobe
// Everything it makes lives in ~/Library/Application Support/FirstCut (VIDEOBUILDER_HOME), see clipfarm/clipper/studio.py; a Mac that had
// the first version (called Video Builder) keeps its "Video Builder" folder there, because the Python and models inside can't move.

import AppKit
import ServiceManagement
import WebKit

let fm = FileManager.default
let res = Bundle.main.resourceURL!
let appCode = res.appendingPathComponent("app")
let binDir = res.appendingPathComponent("bin")
let homeDir: URL = {
    if let h = ProcessInfo.processInfo.environment["VIDEOBUILDER_HOME"] { return URL(fileURLWithPath: (h as NSString).expandingTildeInPath) }
    let support = fm.homeDirectoryForCurrentUser.appendingPathComponent("Library/Application Support")
    let new = support.appendingPathComponent("FirstCut"), old = support.appendingPathComponent("Video Builder")
    if !fm.fileExists(atPath: new.path) && fm.fileExists(atPath: old.appendingPathComponent("settings.json").path) { return old }
    return new
}()
let envDir = homeDir.appendingPathComponent("runtime/env")
let homeShown = (homeDir.path as NSString).abbreviatingWithTildeInPath
let build = (Bundle.main.infoDictionary?["CFBundleVersion"] as? String) ?? "dev"

func jsString(_ s: String) -> String {
    let data = (try? JSONSerialization.data(withJSONObject: [s])) ?? Data("[\"\"]".utf8)
    return String(String(data: data, encoding: .utf8)!.dropFirst().dropLast())
}

func freePort() -> Int {
    let s = socket(AF_INET, SOCK_STREAM, 0)
    defer { close(s) }
    var addr = sockaddr_in(sin_len: UInt8(MemoryLayout<sockaddr_in>.size), sin_family: sa_family_t(AF_INET), sin_port: 0,
                           sin_addr: in_addr(s_addr: inet_addr("127.0.0.1")), sin_zero: (0, 0, 0, 0, 0, 0, 0, 0))
    var len = socklen_t(MemoryLayout<sockaddr_in>.size)
    _ = withUnsafePointer(to: &addr) { $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { bind(s, $0, len) } }
    _ = withUnsafeMutablePointer(to: &addr) { $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { getsockname(s, $0, &len) } }
    return Int(UInt16(bigEndian: addr.sin_port))
}

/// The top 52 points of the window (the page's title bar): pressing on empty space there moves the window, double-click zooms it,
/// like any title bar. The page reports where its own buttons up there are ("holes"): clicks there go through to the page.
/// The Dock icon with a progress bar along its bottom while something is being made (like HandBrake's).
final class DockProgress: NSView {
    var value: Double = -1 { didSet { needsDisplay = true } }
    override func draw(_ dirtyRect: NSRect) {
        NSApp.applicationIconImage?.draw(in: bounds)
        guard value >= 0 else { return }
        let h = max(6, bounds.height * 0.11), inset = bounds.width * 0.12
        let bar = NSRect(x: inset, y: bounds.height * 0.1, width: bounds.width - 2 * inset, height: h)
        NSColor(white: 0, alpha: 0.65).setFill()
        NSBezierPath(roundedRect: bar, xRadius: h / 2, yRadius: h / 2).fill()
        let fill = NSRect(x: bar.minX + 2, y: bar.minY + 2, width: max(h - 4, (bar.width - 4) * min(1, value)), height: h - 4)
        NSColor(calibratedRed: 0.35, green: 0.66, blue: 0.9, alpha: 1).setFill()
        NSBezierPath(roundedRect: fill, xRadius: (h - 4) / 2, yRadius: (h - 4) / 2).fill()
    }
}

/// Files and folders dropped from the Finder anywhere in the window, a text box included (WebKit would type the path into it, or do
/// nothing): each path goes to the page (window.vbDrop). Nothing is copied: FirstCut reads footage where it is. Drags inside the page
/// (shots in Find footage) have no file URLs and go on to WebKit as before.
final class DropWebView: WKWebView {
    // Photos (and some other apps) don't put the video on the drag as a file: the pasteboard holds a preview JPEG as a file URL and the real
    // video as a FILE PROMISE that has to be exported on drop. So a promise wins over any file URL that came with it, and the exported copy
    // goes to ~/Library/Caches/FirstCut/dropped/<id>/ (a copy: Finder drops are read where they are; subfolders older than a week are
    // removed at launch). The page is told while the export runs (window.vbDropBusy) so it can show a placeholder tile.
    static let droppedRoot: URL = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Library/Caches/FirstCut/dropped", isDirectory: true)
    private var exporting = 0

    static func cleanDropped() {
        let fm = FileManager.default
        guard let subs = try? fm.contentsOfDirectory(at: droppedRoot, includingPropertiesForKeys: [.contentModificationDateKey]) else { return }
        for u in subs {
            let when = (try? u.resourceValues(forKeys: [.contentModificationDateKey]))?.contentModificationDate ?? Date()
            if Date().timeIntervalSince(when) > 7 * 24 * 3600 { try? fm.removeItem(at: u) }
        }
    }

    override init(frame: CGRect, configuration: WKWebViewConfiguration) {
        super.init(frame: frame, configuration: configuration)
        registerForDraggedTypes(NSFilePromiseReceiver.readableDraggedTypes.map { NSPasteboard.PasteboardType($0) } + [.fileURL])
    }
    required init?(coder: NSCoder) { fatalError("init(coder:) is not used") }

    private func promises(_ info: NSDraggingInfo) -> [NSFilePromiseReceiver] {
        (info.draggingPasteboard.readObjects(forClasses: [NSFilePromiseReceiver.self], options: nil) as? [NSFilePromiseReceiver]) ?? []
    }
    private func files(_ info: NSDraggingInfo) -> [URL] {
        (info.draggingPasteboard.readObjects(forClasses: [NSURL.self], options: [.urlReadingFileURLsOnly: true]) as? [URL]) ?? []
    }
    private func ours(_ info: NSDraggingInfo) -> Bool { !promises(info).isEmpty || !files(info).isEmpty }

    private func hint(_ on: Bool) { evaluateJavaScript("window.vbDragOver && window.vbDragOver(\(on))") }   // ("Drop to add to your short clips")
    override func draggingEntered(_ sender: NSDraggingInfo) -> NSDragOperation {
        if ours(sender) { hint(true); return .copy }
        return super.draggingEntered(sender)
    }
    override func draggingExited(_ sender: NSDraggingInfo?) {
        hint(false)
        super.draggingExited(sender)
    }
    override func draggingEnded(_ sender: NSDraggingInfo) {
        hint(false)
        super.draggingEnded(sender)
    }
    override func draggingUpdated(_ sender: NSDraggingInfo) -> NSDragOperation {
        ours(sender) ? .copy : super.draggingUpdated(sender)
    }
    override func prepareForDragOperation(_ sender: NSDraggingInfo) -> Bool {
        ours(sender) ? true : super.prepareForDragOperation(sender)
    }
    override func performDragOperation(_ sender: NSDraggingInfo) -> Bool {
        hint(false)
        let proms = promises(sender)
        if !proms.isEmpty {                                              // (the real thing is the promise: any file URL here is its preview picture)
            let dest = DropWebView.droppedRoot.appendingPathComponent(UUID().uuidString, isDirectory: true)
            try? FileManager.default.createDirectory(at: dest, withIntermediateDirectories: true)
            exporting += proms.count
            evaluateJavaScript("window.vbDropBusy && window.vbDropBusy(true)")
            for r in proms {
                r.receivePromisedFiles(atDestination: dest, options: [:], operationQueue: .main) { [weak self] url, error in
                    guard let self = self else { return }
                    self.exporting -= 1
                    if let error = error {
                        self.evaluateJavaScript("window.vbDropFailed && window.vbDropFailed(\(jsString(error.localizedDescription)))")
                    } else {
                        self.evaluateJavaScript("window.vbDrop && window.vbDrop(\(jsString(url.path)))")
                    }
                    if self.exporting <= 0 { self.exporting = 0; self.evaluateJavaScript("window.vbDropBusy && window.vbDropBusy(false)") }
                }
            }
            return true
        }
        let urls = files(sender)
        if urls.isEmpty { return super.performDragOperation(sender) }
        for u in urls { evaluateJavaScript("window.vbDrop && window.vbDrop(\(jsString(u.path)))") }
        return true
    }
}

final class DragStrip: NSView {
    var holes: [NSRect] = []                                               // in the strip's own coordinates (top-left origin, points)
    override var isFlipped: Bool { true }
    override func hitTest(_ point: NSPoint) -> NSView? {
        let p = convert(point, from: superview)
        guard bounds.contains(p), !holes.contains(where: { $0.contains(p) }) else { return nil }
        return self
    }
    override func mouseDown(with event: NSEvent) {
        if event.clickCount == 2 {
            let action = UserDefaults.standard.string(forKey: "AppleActionOnDoubleClick") ?? "Maximize"
            if action == "Minimize" { window?.performMiniaturize(nil) } else if action != "None" { window?.performZoom(nil) }
            return
        }
        window?.performDrag(with: event)
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate, NSWindowDelegate, WKNavigationDelegate, WKUIDelegate, WKScriptMessageHandler {
    var window: NSWindow!
    var web: WKWebView!
    var dragStrip: DragStrip?
    let dockView = DockProgress(frame: NSRect(x: 0, y: 0, width: 128, height: 128))
    var server: Process?
    var port = 0
    var quitting = false
    let updates = Updates()                                              // automatic updates (updates.swift, Sparkle)
    var tries = 0

    // MARK: launch
    func applicationDidFinishLaunching(_ note: Notification) {
        buildMenu()
        let cfg = WKWebViewConfiguration()
        cfg.mediaTypesRequiringUserActionForPlayback = []
        cfg.preferences.setValue(true, forKey: "developerExtrasEnabled")
        cfg.preferences.isElementFullscreenEnabled = true                  // a video's own full-screen button (off by default in WKWebView)
        cfg.userContentController.add(self, name: "vb")
        var glassOn = false
        if #available(macOS 26.0, *) { glassOn = true }
        cfg.userContentController.addUserScript(WKUserScript(source: "window.vbNative = {build: \(jsString(build))}; document.documentElement.classList.add('native'\(glassOn ? ", 'glass'" : ""));",
                                                             injectionTime: .atDocumentStart, forMainFrameOnly: true))
        DropWebView.cleanDropped()
        web = DropWebView(frame: .zero, configuration: cfg)
        web.navigationDelegate = self
        web.uiDelegate = self
        web.setValue(false, forKey: "drawsBackground")                    // the page's see-through parts show the material below
        web.underPageBackgroundColor = .clear

        window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 1240, height: 820),
                          styleMask: [.titled, .closable, .miniaturizable, .resizable, .fullSizeContentView], backing: .buffered, defer: false)
        window.title = "FirstCut"
        window.titleVisibility = .hidden
        window.titlebarAppearsTransparent = true
        window.isReleasedWhenClosed = false
        window.minSize = NSSize(width: 860, height: 560)
        let material = NSVisualEffectView()                                // the desktop, blurred, behind the see-through parts
        material.material = .sidebar
        material.blendingMode = .behindWindow
        material.state = .followsWindowActiveState
        window.contentView = material
        web.translatesAutoresizingMaskIntoConstraints = false
        var glass: NSView?
        if #available(macOS 26.0, *) {                                      // Liquid Glass: the sidebar floats as a glass panel (macOS 26+)
            let g = NSGlassEffectView()
            g.cornerRadius = 18
            g.translatesAutoresizingMaskIntoConstraints = false
            material.addSubview(g)
            glass = g
        }
        material.addSubview(web)                                            // the page on top: its sidebar is see-through over the glass
        NSLayoutConstraint.activate([
            web.leadingAnchor.constraint(equalTo: material.leadingAnchor), web.trailingAnchor.constraint(equalTo: material.trailingAnchor),
            web.topAnchor.constraint(equalTo: material.topAnchor), web.bottomAnchor.constraint(equalTo: material.bottomAnchor)])
        if let g = glass {                                                  // (matches the page: #side is 240 wide, the panel 8 in from the edges)
            NSLayoutConstraint.activate([
                g.leadingAnchor.constraint(equalTo: material.leadingAnchor, constant: 8), g.widthAnchor.constraint(equalToConstant: 224),
                g.topAnchor.constraint(equalTo: material.topAnchor, constant: 8), g.bottomAnchor.constraint(equalTo: material.bottomAnchor, constant: -8)])
        }
        let strip = DragStrip()                                             // above the page, but only where the page has nothing to click
        strip.translatesAutoresizingMaskIntoConstraints = false
        material.addSubview(strip)
        NSLayoutConstraint.activate([
            strip.leadingAnchor.constraint(equalTo: material.leadingAnchor), strip.trailingAnchor.constraint(equalTo: material.trailingAnchor),
            strip.topAnchor.constraint(equalTo: material.topAnchor), strip.heightAnchor.constraint(equalToConstant: 52)])
        dragStrip = strip
        window.setFrameAutosaveName("FirstCutMain")
        if !window.setFrameUsingName("FirstCutMain") { window.center() }
        window.delegate = self
        window.makeKeyAndOrderFront(nil)
        placeWindowButtons()
        NSApp.activate(ignoringOtherApps: true)
        prepare()
        updates.send = { [weak self] state in                               // the page shows the update (a sidebar card), never a Sparkle window
            guard let self = self, let d = try? JSONSerialization.data(withJSONObject: state), let j = String(data: d, encoding: .utf8) else { return }
            self.web.evaluateJavaScript("window.vbUpdate && window.vbUpdate(\(j))")
        }
        updates.busyURL = { [weak self] in
            guard let self = self, self.port > 0 else { return nil }
            return URL(string: "http://127.0.0.1:\(self.port)/api/app/busy")
        }
        updates.start()
    }

    /// The red / yellow / green buttons, moved down and in so they sit inside the sidebar's glass panel (and on the page's 52-point title
    /// bar) instead of in its very corner. AppKit lays them out again on some changes (size, focus, full screen), so this runs after those.
    func placeWindowButtons() {
        guard !window.styleMask.contains(.fullScreen), let close = window.standardWindowButton(.closeButton),
              let container = close.superview?.superview else { return }
        let barHeight: CGFloat = 52, left: CGFloat = 22
        var f = container.frame
        f.size.height = barHeight
        f.origin.y = window.frame.height - barHeight
        container.frame = f
        let dx = left - close.frame.minX
        for type in [NSWindow.ButtonType.closeButton, .miniaturizeButton, .zoomButton] {
            guard let b = window.standardWindowButton(type) else { continue }
            b.setFrameOrigin(NSPoint(x: b.frame.minX + dx, y: (barHeight - b.frame.height) / 2))
        }
    }

    // MARK: the app's own Python (first run, and after an update)
    func prepare() {
        // A downloaded zip marks every file in it as "from the internet"; once the person has allowed the app itself, take that mark off
        // the tools inside it (uv, ffmpeg, ffprobe), so they run without asking again.
        let x = Process()
        x.executableURL = URL(fileURLWithPath: "/usr/bin/xattr")
        x.arguments = ["-dr", "com.apple.quarantine", binDir.path, appCode.path]
        try? x.run()
        x.waitUntilExit()
        let stamp = (try? String(contentsOf: envDir.appendingPathComponent(".vb-build"), encoding: .utf8))?.trimmingCharacters(in: .whitespacesAndNewlines)
        if stamp == build && fm.fileExists(atPath: envDir.appendingPathComponent("bin/python").path) { startServer(); return }
        showPage(title: "Getting FirstCut ready", detail: "Installing the video engine. This happens once and takes a few minutes.", pct: 2)
        try? fm.createDirectory(at: homeDir.appendingPathComponent("logs"), withIntermediateDirectories: true)
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/bin/bash")
        p.arguments = [appCode.appendingPathComponent("bootstrap.sh").path, binDir.appendingPathComponent("uv").path, envDir.path,
                       appCode.appendingPathComponent("requirements-app.txt").path, build, homeDir.path]
        let pipe = Pipe()
        p.standardOutput = pipe
        p.standardError = pipe
        var lines = ""
        pipe.fileHandleForReading.readabilityHandler = { h in
            guard let s = String(data: h.availableData, encoding: .utf8), !s.isEmpty else { return }
            lines += s
            for line in s.split(separator: "\n") where line.hasPrefix("PROGRESS ") {
                let parts = line.split(separator: " ", maxSplits: 2)
                if parts.count == 3, let pct = Int(parts[1]) {
                    DispatchQueue.main.async { self.web.evaluateJavaScript("setProgress(\(pct), \(jsString(String(parts[2]))))") }
                }
            }
        }
        p.terminationHandler = { proc in
            pipe.fileHandleForReading.readabilityHandler = nil
            DispatchQueue.main.async {
                if proc.terminationStatus == 0 { self.startServer() } else {
                    try? lines.write(to: homeDir.appendingPathComponent("logs/install.log"), atomically: true, encoding: .utf8)
                    self.showPage(title: "FirstCut couldn't install its engine",
                                  detail: "Check the internet connection and press Try again. Details: logs/install.log in \(homeShown).\n\n"
                                    + String(lines.suffix(600)), pct: -1, retry: true)
                }
            }
        }
        do { try p.run() } catch { showPage(title: "FirstCut couldn't start its installer", detail: "\(error)", pct: -1, retry: true) }
    }

    // MARK: the engine
    func startServer() {
        if let s = server, s.isRunning { return }
        if port == 0 { port = freePort() }                                   // a restart keeps the address, so the open page carries on
        let p = Process()
        p.executableURL = envDir.appendingPathComponent("bin/python")
        p.arguments = ["-m", "clipfarm", "studio", "--port", String(port)]
        p.currentDirectoryURL = homeDir
        var env = ProcessInfo.processInfo.environment
        env["PYTHONPATH"] = appCode.path
        env["PYTHONUNBUFFERED"] = "1"
        env["CLIPFARM_ROOT"] = homeDir.path
        env["CLIPFARM_CONFIG"] = appCode.appendingPathComponent("config/app.yaml").path
        env["VIDEOBUILDER_HOME"] = homeDir.path
        env["VIDEOBUILDER_UV"] = binDir.appendingPathComponent("uv").path
        env["VIDEOBUILDER_BIN"] = binDir.path
        env["UV_PYTHON_INSTALL_DIR"] = homeDir.appendingPathComponent("runtime/python").path
        env["UV_CACHE_DIR"] = homeDir.appendingPathComponent("runtime/uv-cache").path
        env["PATH"] = [binDir.path, homeDir.appendingPathComponent("runtime/bin").path, "/usr/bin", "/bin", "/usr/sbin", "/sbin",
                       (env["HOME"] ?? "") + "/.local/bin"].joined(separator: ":")   // (no Homebrew: the app runs on what it brings, as on any Mac)
        p.environment = env
        let log = homeDir.appendingPathComponent("logs/server.log")
        if !fm.fileExists(atPath: log.path) { fm.createFile(atPath: log.path, contents: nil) }
        if let h = try? FileHandle(forWritingTo: log) { h.seekToEndOfFile(); p.standardOutput = h; p.standardError = h }
        p.terminationHandler = { proc in
            DispatchQueue.main.async {
                if self.quitting { return }
                if proc.terminationStatus == 75 { self.startServer(); return }        // the engine asked to restart (new settings)
                self.showPage(title: "FirstCut stopped", detail: "The engine quit unexpectedly. Press Try again; if it keeps happening, the details are in logs/server.log in \(homeShown).\n\n"
                                + self.logTail(), pct: -1, retry: true)
            }
        }
        do { try p.run(); server = p } catch { showPage(title: "FirstCut couldn't start", detail: "\(error)", pct: -1, retry: true); return }
        if web.url?.host != "127.0.0.1" { showPage(title: "Starting FirstCut…", detail: "", pct: 0) }
        waitForServer()
    }

    func waitForServer() {
        var req = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/api/app/ping")!)
        req.timeoutInterval = 2
        URLSession.shared.dataTask(with: req) { _, resp, _ in
            DispatchQueue.main.async {
                if (resp as? HTTPURLResponse)?.statusCode == 200 {
                    self.tries = 0
                    self.web.load(URLRequest(url: URL(string: "http://127.0.0.1:\(self.port)/")!))
                } else if let s = self.server, s.isRunning {
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { self.waitForServer() }
                }
            }
        }.resume()
    }

    func logTail() -> String {
        let text = (try? String(contentsOf: homeDir.appendingPathComponent("logs/server.log"), encoding: .utf8)) ?? ""
        return String(text.suffix(900))
    }

    func showPage(title: String, detail: String, pct: Int, retry: Bool = false) {
        let html = """
        <!doctype html><html><head><meta charset="utf-8"><style>
        :root{color-scheme:light dark;--text:#16202c;--muted:#5b6776;--accent:#5aa9e6;--track:rgba(22,32,44,.12)}
        @media (prefers-color-scheme:dark){:root{--text:#ececf1;--muted:#a3abb7;--accent:#7dbdf0;--track:rgba(255,255,255,.14)}}
        body{margin:0;height:100vh;display:flex;align-items:center;justify-content:center;font:14px -apple-system,Helvetica;color:var(--text);background:transparent}
        .box{width:440px;text-align:center}.t{font-size:20px;font-weight:600}.d{color:var(--muted);margin-top:8px;white-space:pre-wrap;max-height:300px;overflow:auto;text-align:left}
        .bar{height:4px;border-radius:2px;background:var(--track);margin-top:22px;overflow:hidden}.bar i{display:block;height:100%;width:0;background:var(--accent);transition:width .4s}
        button{margin-top:20px;font:inherit;font-weight:600;padding:8px 18px;border:0;border-radius:8px;background:var(--accent);color:#0b1b2b}
        </style></head><body><div class="box"><div class="t" id="t">\(title)</div><div class="d" id="d"></div>
        \(pct >= 0 ? "<div class='bar'><i id='b' style='width:\(pct)%'></i></div>" : "")
        \(retry ? "<button onclick=\"webkit.messageHandlers.vb.postMessage({cmd:'retry'})\">Try again</button>" : "")</div>
        <script>document.getElementById('d').textContent=\(jsString(detail));function setProgress(p,t){var b=document.getElementById('b');if(b)b.style.width=p+'%';document.getElementById('d').textContent=t;}</script></body></html>
        """
        web.loadHTMLString(html, baseURL: nil)
    }

    // MARK: the page asks for Mac things: choose a folder or file, show in Finder, open, appearance
    func userContentController(_ ucc: WKUserContentController, didReceive message: WKScriptMessage) {
        guard let m = message.body as? [String: Any], let cmd = m["cmd"] as? String else { return }
        switch cmd {
        case "retry":
            tries = 0
            if fm.fileExists(atPath: envDir.appendingPathComponent(".vb-build").path) { startServer() } else { prepare() }
        case "pick":
            let id = m["id"] as? String ?? ""
            let panel = NSOpenPanel()
            let kind = m["kind"] as? String ?? "folder"
            panel.canChooseDirectories = kind != "file"
            panel.canChooseFiles = kind != "folder"
            panel.allowsMultipleSelection = false
            panel.message = m["prompt"] as? String ?? ""
            panel.prompt = "Choose"
            if let start = m["start"] as? String, !start.isEmpty { panel.directoryURL = URL(fileURLWithPath: (start as NSString).expandingTildeInPath) }
            panel.beginSheetModal(for: window) { r in
                let path = r == .OK ? (panel.url?.path ?? "") : ""
                self.web.evaluateJavaScript("window.vbPicked && window.vbPicked(\(jsString(id)), \(jsString(path)))")
            }
        case "trash":                                                        // a project's own folder, after the page asked (Put Back works)
            guard let p = m["path"] as? String, let id = m["id"] as? String else { break }
            NSWorkspace.shared.recycle([URL(fileURLWithPath: p)]) { _, error in
                DispatchQueue.main.async {
                    self.web.evaluateJavaScript("vbTrashed(\(jsString(id)), \(jsString(error?.localizedDescription ?? "")))")
                }
            }
        case "dock":                                                         // progress on the Dock icon; a bounce when something is done
            if let v = m["progress"] as? Double {
                let shown = NSApp.dockTile.contentView === dockView
                if v >= 0 {
                    dockView.value = v
                    if !shown { NSApp.dockTile.contentView = dockView }
                } else if shown {
                    NSApp.dockTile.contentView = nil
                }
                NSApp.dockTile.display()
            }
            if m["done"] != nil { NSApp.requestUserAttention(.informationalRequest) }   // (bounces once, only when FirstCut isn't in front)
        case "holes":                                                        // where the page has buttons in the title bar strip
            let rs = (m["rects"] as? [[Double]]) ?? []
            dragStrip?.holes = rs.filter { $0.count == 4 }.map { NSRect(x: $0[0], y: $0[1], width: $0[2], height: $0[3]) }
        case "reveal":
            if let p = m["path"] as? String { NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: p)]) }
        case "open":
            if let p = m["path"] as? String { NSWorkspace.shared.open(URL(fileURLWithPath: p)) }
        case "appearance":
            let v = m["value"] as? String ?? "system"
            NSApp.appearance = v == "dark" ? NSAppearance(named: .darkAqua) : v == "light" ? NSAppearance(named: .aqua) : nil
        case "update":                                                       // the page: updates mode, check now, install / later
            updates.handle(m)
        case "login":                                                        // "Open FirstCut when you log in"
            var problem = ""
            if let on = m["value"] as? Bool {
                do { if on { try SMAppService.mainApp.register() } else { try SMAppService.mainApp.unregister() } }
                catch let e { problem = e.localizedDescription }
            }
            let enabled = SMAppService.mainApp.status == .enabled
            web.evaluateJavaScript("window.vbLogin && window.vbLogin(\(enabled), \(jsString(problem)))")
        default:
            break
        }
    }

    // MARK: navigation: only our own pages here; dropped files and folders go to the page; links open in the browser
    func webView(_ webView: WKWebView, decidePolicyFor action: WKNavigationAction, decisionHandler: @escaping (WKNavigationActionPolicy) -> Void) {
        guard let url = action.request.url else { return decisionHandler(.cancel) }
        if url.isFileURL {
            webView.evaluateJavaScript("window.vbDrop && window.vbDrop(\(jsString(url.path)))")
            return decisionHandler(.cancel)
        }
        if url.host == "127.0.0.1" || url.scheme == "about" || url.scheme == "data" || url.scheme == "blob" { return decisionHandler(.allow) }
        NSWorkspace.shared.open(url)
        decisionHandler(.cancel)
    }

    func webView(_ webView: WKWebView, createWebViewWith configuration: WKWebViewConfiguration, for action: WKNavigationAction,
                 windowFeatures: WKWindowFeatures) -> WKWebView? {
        if let url = action.request.url { NSWorkspace.shared.open(url) }
        return nil
    }

    // Dictation: the microphone, for FirstCut's own page only (macOS asks the person the first time: NSMicrophoneUsageDescription)
    @available(macOS 12.0, *)
    func webView(_ webView: WKWebView, requestMediaCapturePermissionFor origin: WKSecurityOrigin, initiatedByFrame frame: WKFrameInfo,
                 type: WKMediaCaptureType, decisionHandler: @escaping (WKPermissionDecision) -> Void) {
        decisionHandler(origin.host == "127.0.0.1" && type == .microphone ? .grant : .deny)
    }

    func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error) {
        let e = error as NSError
        if e.code == NSURLErrorCancelled || e.code == 102 { return }
        if let s = server, s.isRunning { DispatchQueue.main.asyncAfter(deadline: .now() + 1) { self.waitForServer() } }
    }

    func webView(_ webView: WKWebView, runJavaScriptAlertPanelWithMessage message: String, initiatedByFrame frame: WKFrameInfo,
                 completionHandler: @escaping () -> Void) {
        let a = NSAlert()
        a.messageText = message
        a.addButton(withTitle: "OK")
        a.beginSheetModal(for: window) { _ in completionHandler() }
    }

    func webView(_ webView: WKWebView, runJavaScriptConfirmPanelWithMessage message: String, initiatedByFrame frame: WKFrameInfo,
                 completionHandler: @escaping (Bool) -> Void) {
        let a = NSAlert()
        a.messageText = message
        a.addButton(withTitle: "OK")
        a.addButton(withTitle: "Cancel")
        a.beginSheetModal(for: window) { r in completionHandler(r == .alertFirstButtonReturn) }
    }

    func webView(_ webView: WKWebView, runOpenPanelWith parameters: WKOpenPanelParameters, initiatedByFrame frame: WKFrameInfo,
                 completionHandler: @escaping ([URL]?) -> Void) {
        let p = NSOpenPanel()
        p.canChooseDirectories = parameters.allowsDirectories
        p.allowsMultipleSelection = parameters.allowsMultipleSelection
        p.beginSheetModal(for: window) { r in completionHandler(r == .OK ? p.urls : nil) }
    }

    // MARK: windows and quitting
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { false }

    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool {
        if !flag { window.makeKeyAndOrderFront(nil) }
        return true
    }

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        guard let s = server, s.isRunning, port > 0 else { return .terminateNow }
        var req = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/api/app/busy")!)
        req.timeoutInterval = 2
        URLSession.shared.dataTask(with: req) { data, _, _ in
            let busy = (try? JSONSerialization.jsonObject(with: data ?? Data())) as? [String: Any]
            let what = busy?["what"] as? String ?? ""
            DispatchQueue.main.async {
                if what.isEmpty || (self.updates.installing && !Updates.blocking(what)) {   // (an update's own quit: cataloguing resumes after it)
                    self.stopServer(); NSApp.reply(toApplicationShouldTerminate: true); return
                }
                let a = NSAlert()
                a.messageText = "Quit while \(what)?"
                a.informativeText = "Quitting stops it. Close the window instead and it keeps going in the background."
                a.addButton(withTitle: "Keep it running")
                a.addButton(withTitle: "Quit anyway")
                let r = a.runModal()
                if r == .alertSecondButtonReturn { self.stopServer(); NSApp.reply(toApplicationShouldTerminate: true) }
                else { NSApp.reply(toApplicationShouldTerminate: false) }
            }
        }.resume()
        return .terminateLater
    }

    func stopServer() {
        quitting = true
        if let s = server, s.isRunning {
            s.terminate()
            let deadline = Date().addingTimeInterval(5)
            while s.isRunning && Date() < deadline { usleep(100_000) }
            if s.isRunning { kill(s.processIdentifier, SIGKILL) }
        }
    }

    // MARK: menus (an Edit menu is what makes copy and paste work in the page)
    func buildMenu() {
        let main = NSMenu()
        func menu(_ title: String, _ items: [NSMenuItem]) -> NSMenu {
            let item = NSMenuItem()
            let m = NSMenu(title: title)
            items.forEach { m.addItem($0) }
            item.submenu = m
            main.addItem(item)
            return m
        }
        func item(_ title: String, _ action: Selector?, _ key: String, _ mods: NSEvent.ModifierFlags = .command, target: AnyObject? = nil) -> NSMenuItem {
            let i = NSMenuItem(title: title, action: action, keyEquivalent: key)
            i.keyEquivalentModifierMask = mods
            i.target = target
            return i
        }
        _ = menu("FirstCut", [item("About FirstCut", #selector(NSApplication.orderFrontStandardAboutPanel(_:)), ""),
                                   item("Check for Updates…", #selector(checkUpdates), "", target: self), .separator(),
                                   item("Settings…", #selector(openSettings), ",", target: self), .separator(),
                                   item("Hide FirstCut", #selector(NSApplication.hide(_:)), "h"), .separator(),
                                   item("Quit FirstCut", #selector(NSApplication.terminate(_:)), "q")])
        _ = menu("Edit", [item("Undo", Selector(("undo:")), "z"), item("Redo", Selector(("redo:")), "z", [.command, .shift]), .separator(),
                          item("Cut", #selector(NSText.cut(_:)), "x"), item("Copy", #selector(NSText.copy(_:)), "c"),
                          item("Paste", #selector(NSText.paste(_:)), "v"), item("Select All", #selector(NSText.selectAll(_:)), "a")])
        _ = menu("View", [item("Reload", #selector(reloadPage), "r", target: self)])
        let win = menu("Window", [item("Minimize", #selector(NSWindow.performMiniaturize(_:)), "m"),
                                  item("Close", #selector(NSWindow.performClose(_:)), "w"), .separator(),
                                  item("FirstCut", #selector(showMain), "0", target: self)])
        NSApp.mainMenu = main
        NSApp.windowsMenu = win
    }

    @objc func openSettings() { showMain(); web.evaluateJavaScript("window.openSettings && window.openSettings()") }
    @objc func showMain() { window.makeKeyAndOrderFront(nil) }
    @objc func checkUpdates() { updates.handle(["action": "check"]) }
    func windowDidResize(_ notification: Notification) { placeWindowButtons() }
    func windowDidBecomeKey(_ notification: Notification) { placeWindowButtons() }
    func windowDidResignKey(_ notification: Notification) { placeWindowButtons() }
    func windowDidExitFullScreen(_ notification: Notification) { placeWindowButtons() }
    func windowDidChangeBackingProperties(_ notification: Notification) { placeWindowButtons() }
    @objc func reloadPage() {
        if port > 0 { web.load(URLRequest(url: URL(string: "http://127.0.0.1:\(port)/")!)) } else { prepare() }
    }
}

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.setActivationPolicy(.regular)
app.run()
