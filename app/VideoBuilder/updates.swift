// FirstCut's automatic updates (Sparkle 2, MIT): the update feed is the appcast.xml on the newest GitHub release; every update is
// EdDSA-signed with a key that only the build Mac holds (SUPublicEDKey in Info.plist checks it), and notarized like the app.
//
// The page shows it, never a Sparkle window: this class is Sparkle's "user driver" and passes each state to window.vbUpdate(...):
//   checking · available (Ask me first) · downloading (progress) · ready ("Restart to update") · installing · none · error
// The page answers with native("update", {action: "mode"|"check"|"install"|"later"}).
//
// Never in the middle of work: before installing it asks the engine (/api/app/busy). A video, clips or a transcript being made blocks
// it; cataloguing and Setup's downloads don't (both carry on after the restart). "Install automatically" restarts only when FirstCut has
// been left alone for IDLE_S (or its window is closed), and otherwise on the next quit (Sparkle installs as the app terminates).

import AppKit
import Sparkle

final class Updates: NSObject, SPUUserDriver, SPUUpdaterDelegate {
    static let IDLE_S: TimeInterval = 600

    var send: ([String: Any]) -> Void = { _ in }                 // -> window.vbUpdate(...)
    var busyURL: () -> URL? = { nil }                            // the engine's /api/app/busy
    private(set) var updater: SPUUpdater?
    private(set) var installing = false                          // (the quit that follows is ours: no "quit while…?" question)
    var mode = "auto"                                            // "auto" | "ask" | "off"

    private var item: SUAppcastItem?
    private var foundReply: ((SPUUserUpdateChoice) -> Void)?
    private var readyReply: ((SPUUserUpdateChoice) -> Void)?
    private var userCheck = false
    private var wanted = false                                   // the person pressed "Restart to update" (or "Update")
    private var expected: UInt64 = 0, got: UInt64 = 0, lastSent = Date.distantPast
    private var lastInput = Date()
    private var failed = false                                   // (an error stays on the card; a later "dismiss" doesn't hide it)
    private var timer: Timer?
    private var monitor: Any?

    func start() {
        let u = SPUUpdater(hostBundle: .main, applicationBundle: .main, userDriver: self, delegate: self)
        updater = u
        do { try u.start() } catch {
            send(["state": "error", "error": "Updates can't start: \(error.localizedDescription)"])
            return
        }
        apply()
        monitor = NSEvent.addLocalMonitorForEvents(matching: [.keyDown, .leftMouseDown, .rightMouseDown, .scrollWheel, .mouseMoved]) { [weak self] e in
            self?.lastInput = Date(); return e
        }
        timer = Timer.scheduledTimer(withTimeInterval: 60, repeats: true) { [weak self] _ in self?.tryInstall() }
    }

    private func apply() {
        guard let u = updater else { return }
        u.automaticallyChecksForUpdates = mode != "off"
        u.automaticallyDownloadsUpdates = false                  // (this driver downloads itself when the mode says so)
        u.updateCheckInterval = 24 * 3600
    }

    /// Messages from the page.
    func handle(_ m: [String: Any]) {
        switch m["action"] as? String ?? "" {
        case "mode":
            let v = m["mode"] as? String ?? "auto"
            mode = ["auto", "ask", "off"].contains(v) ? v : "auto"
            apply()
        case "check":
            userCheck = true
            updater?.checkForUpdates()
        case "install":
            wanted = true
            if let r = foundReply { foundReply = nil; r(.install) }   // (not downloaded yet: download, then install when ready)
            tryInstall()
        case "later":
            wanted = false
            if let r = foundReply { foundReply = nil; r(.dismiss); send(["state": "none"]) }
        default:
            break
        }
    }

    // MARK: when it may restart
    static func blocking(_ what: String) -> Bool {
        let w = what.lowercased()
        return !w.isEmpty && !w.contains("catalogu") && !w.contains("download")
    }

    private func idle() -> Bool {
        let shown = NSApp.windows.contains { $0.isVisible && $0.className != "NSStatusBarWindow" }
        return Date().timeIntervalSince(lastInput) >= Updates.IDLE_S || !shown
    }

    private func tryInstall() {
        guard let reply = readyReply, wanted || (mode == "auto" && idle()) else { return }
        engineBusy { what in
            if Updates.blocking(what) {
                self.send(["state": "ready", "version": self.version, "notes": self.notes, "waiting": what])
                return
            }
            guard self.readyReply != nil else { return }
            self.readyReply = nil
            self.installing = true
            self.send(["state": "installing", "version": self.version])
            reply(.install)
        }
    }

    private func engineBusy(_ done: @escaping (String) -> Void) {
        guard let url = busyURL() else { return done("") }
        var req = URLRequest(url: url)
        req.timeoutInterval = 3
        URLSession.shared.dataTask(with: req) { data, _, _ in
            let j = (try? JSONSerialization.jsonObject(with: data ?? Data())) as? [String: Any]
            let what = j?["what"] as? String ?? ""
            DispatchQueue.main.async { done(what) }
        }.resume()
    }

    private var version: String { item?.displayVersionString ?? "" }
    private var notes: String {
        let raw = item?.itemDescription ?? ""
        return raw.replacingOccurrences(of: "<[^>]+>", with: " ", options: .regularExpression)
                  .replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression).trimmingCharacters(in: .whitespaces)
    }

    // MARK: SPUUpdaterDelegate
    /// A test build can point at another feed: `defaults write app.firstcut.mac FirstCutFeedURL http://127.0.0.1:…/appcast.xml`
    /// or the FIRSTCUT_FEED_URL environment variable. Normally nil: the feed in Info.plist.
    func feedURLString(for updater: SPUUpdater) -> String? {
        ProcessInfo.processInfo.environment["FIRSTCUT_FEED_URL"] ?? UserDefaults.standard.string(forKey: "FirstCutFeedURL")
    }

    // MARK: SPUUserDriver
    func show(_ request: SPUUpdatePermissionRequest, reply: @escaping (SUUpdatePermissionResponse) -> Void) {
        reply(SUUpdatePermissionResponse(automaticUpdateChecks: mode != "off", sendSystemProfile: false))
    }

    func showUserInitiatedUpdateCheck(cancellation: @escaping () -> Void) {
        send(["state": "checking"])
    }

    func showUpdateFound(with appcastItem: SUAppcastItem, state: SPUUserUpdateState, reply: @escaping (SPUUserUpdateChoice) -> Void) {
        item = appcastItem
        if state.stage != .notDownloaded || mode == "auto" || userCheck || wanted {
            userCheck = false
            reply(.install)                                      // download now (or carry on with what was already downloaded)
        } else {
            foundReply = reply                                   // Ask me first: the page offers "Update"
            send(["state": "available", "version": version, "notes": notes])
        }
    }

    func showUpdateReleaseNotes(with downloadData: SPUDownloadData) {}
    func showUpdateReleaseNotesFailedToDownloadWithError(_ error: Error) {}

    func showUpdateNotFoundWithError(_ error: Error, acknowledgement: @escaping () -> Void) {
        send(["state": "none", "asked": userCheck])
        userCheck = false
        acknowledgement()
    }

    func showUpdaterError(_ error: Error, acknowledgement: @escaping () -> Void) {
        if userCheck || wanted || item != nil {                  // (asked for, or an update under way: e.g. one that failed its signature check)
            failed = true
            send(["state": "error", "error": error.localizedDescription])
        }
        userCheck = false
        acknowledgement()
    }

    func showDownloadInitiated(cancellation: @escaping () -> Void) {
        expected = 0; got = 0; failed = false
        send(["state": "downloading", "version": version, "progress": 0])
    }

    func showDownloadDidReceiveExpectedContentLength(_ expectedContentLength: UInt64) { expected = expectedContentLength }

    func showDownloadDidReceiveData(ofLength length: UInt64) {
        got += length
        guard expected > 0, Date().timeIntervalSince(lastSent) > 0.5 else { return }
        lastSent = Date()
        send(["state": "downloading", "version": version, "progress": min(1.0, Double(got) / Double(expected))])
    }

    func showDownloadDidStartExtractingUpdate() { send(["state": "downloading", "version": version, "progress": 1.0]) }
    func showExtractionReceivedProgress(_ progress: Double) {}

    func showReady(toInstallAndRelaunch reply: @escaping (SPUUserUpdateChoice) -> Void) {
        readyReply = reply                                       // held: installed when asked, when idle (auto), or as the app quits
        send(["state": "ready", "version": version, "notes": notes])
        tryInstall()
    }

    func showInstallingUpdate(withApplicationTerminated applicationTerminated: Bool, retryTerminatingApplication: @escaping () -> Void) {
        installing = true
        send(["state": "installing", "version": version])
    }

    func showUpdateInstalledAndRelaunched(_ relaunched: Bool, acknowledgement: @escaping () -> Void) { acknowledgement() }

    func showUpdateInFocus() {}

    func dismissUpdateInstallation() {
        foundReply = nil
        if readyReply == nil && !installing && !failed { send(["state": "none"]) }
        item = nil
    }
}
