import Foundation
import Vision
import AppKit

// usage: cf_ocr image.jpg [more.jpg ...]  -> one JSON line per image: [{"t":text,"x":..,"y":..,"w":..,"h":..,"c":conf}] (normalised, origin top-left)
for path in CommandLine.arguments.dropFirst() {
    var out: [[String: Any]] = []
    if let img = NSImage(contentsOfFile: path), let cg = img.cgImage(forProposedRect: nil, context: nil, hints: nil) {
        let req = VNRecognizeTextRequest()
        req.recognitionLevel = .accurate
        req.usesLanguageCorrection = false
        try? VNImageRequestHandler(cgImage: cg, options: [:]).perform([req])
        for o in req.results ?? [] {
            if let c = o.topCandidates(1).first {
                let b = o.boundingBox
                out.append(["t": c.string, "x": b.minX, "y": 1 - b.maxY, "w": b.width, "h": b.height, "c": c.confidence])
            }
        }
    }
    let data = try! JSONSerialization.data(withJSONObject: out)
    print(String(data: data, encoding: .utf8)!)
}
