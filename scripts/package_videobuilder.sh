#!/bin/bash
# Build FirstCut (first called Video Builder) for any Apple Silicon Mac: dist/FirstCut.app and dist/FirstCut.zip (send the zip).
# Inside the app: the window (Swift), the engine (the clipfarm package, without its tests), its settings without anything personal
# (app/VideoBuilder/make_config.py stops the build if something is left), the first-run installer, and uv + ffmpeg + ffprobe.
# On first open the app makes its own Python (bootstrap.sh) and its Setup screen downloads the models (clipper/studio.py).
#   ./scripts/package_videobuilder.sh
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
OUT="$ROOT/dist"
APP="$OUT/FirstCut.app"
CACHE="$OUT/.cache"
RES="$APP/Contents/Resources"
BUILD="$(date +%Y%m%d%H%M)"                                       # CFBundleVersion: always increasing (Sparkle compares it)
VERSION="$(tr -d '[:space:]' < app/release/VERSION)"                # CFBundleShortVersionString: what people see (0.2.0...)
SPARKLE_V="2.10.1"                                                 # automatic updates (app/VideoBuilder/updates.swift); pinned + checksum
SPARKLE_SHA="1a856d2a9edb8e628de88121d7be04cf0e2167f8641650b78fca00b14244cf27"
FEED="https://github.com/you/firstcut/releases/latest/download/appcast.xml"
ED_KEY="$(tr -d '[:space:]' < app/release/SUPublicEDKey)"
PY="$ROOT/.venv/bin/python"
mkdir -p "$OUT" "$CACHE"
rm -rf "$APP" "$OUT/FirstCut.zip" "$OUT/Video Builder.app" "$OUT/Video Builder.zip"
mkdir -p "$APP/Contents/MacOS" "$RES/app/config" "$RES/bin"

echo "== Sparkle (automatic updates)"
SPK="$CACHE/sparkle-$SPARKLE_V"
if [ ! -d "$SPK/Sparkle.framework" ]; then
  curl -fsSL "https://github.com/sparkle-project/Sparkle/releases/download/$SPARKLE_V/Sparkle-$SPARKLE_V.tar.xz" -o "$CACHE/Sparkle-$SPARKLE_V.tar.xz"
  [ "$(shasum -a 256 "$CACHE/Sparkle-$SPARKLE_V.tar.xz" | awk '{print $1}')" = "$SPARKLE_SHA" ] || { echo "Sparkle download failed its checksum"; exit 1; }
  rm -rf "$SPK" && mkdir -p "$SPK" && tar -xJf "$CACHE/Sparkle-$SPARKLE_V.tar.xz" -C "$SPK"
fi
mkdir -p "$APP/Contents/Frameworks"
ditto "$SPK/Sparkle.framework" "$APP/Contents/Frameworks/Sparkle.framework"
rm -rf "$APP/Contents/Frameworks/Sparkle.framework/Versions/B/XPCServices"   # (only for sandboxed apps; FirstCut isn't)

echo "== the window"
swiftc -O -swift-version 5 -target arm64-apple-macos13 -o "$APP/Contents/MacOS/FirstCut" app/VideoBuilder/main.swift app/VideoBuilder/updates.swift \
  -F "$SPK" -framework Sparkle -framework AppKit -framework WebKit -Xlinker -rpath -Xlinker @executable_path/../Frameworks

echo "== the engine"
COPYFILE_DISABLE=1 tar -cf - --exclude "__pycache__" --exclude "*.pyc" --exclude ".DS_Store" clipfarm | tar -xf - -C "$RES/app"
"$PY" app/VideoBuilder/make_config.py config/config.yaml "$RES/app/config/app.yaml"
cp app/VideoBuilder/bootstrap.sh app/VideoBuilder/requirements-app.txt "$RES/app/"
chmod +x "$RES/app/bootstrap.sh"
PRIVATE_RE="${FIRSTCUT_PRIVATE_PATTERNS:-$(grep -v '^#' .private-patterns 2>/dev/null | paste -sd'|' -)}"   # your own names, hosts and ids (kept out of the repo)
if [ -n "$PRIVATE_RE" ] && grep -rliE "$PRIVATE_RE" "$RES/app" >/dev/null; then
  echo "private details found in the engine:"; grep -rliE "$PRIVATE_RE" "$RES/app"; exit 1
fi

echo "== tools (downloaded once into dist/.cache)"
if [ ! -x "$CACHE/uv" ]; then
  curl -fsSL "https://github.com/astral-sh/uv/releases/latest/download/uv-aarch64-apple-darwin.tar.gz" -o "$CACHE/uv.tgz"
  tar -xzf "$CACHE/uv.tgz" -C "$CACHE" && mv "$CACHE"/uv-aarch64-apple-darwin/uv "$CACHE/uv" && rm -rf "$CACHE/uv.tgz" "$CACHE"/uv-aarch64-apple-darwin
fi
for tool in ffmpeg ffprobe; do
  if [ ! -x "$CACHE/$tool" ]; then
    # Martin Riedl's static, signed macOS arm64 builds: the newest release build listed on the download page, checked against its .sha256
    url="$(curl -fsSL https://ffmpeg.martin-riedl.de/ | grep -oE "/download/macos/arm64/[^\"]*_[0-9.]+/$tool\.zip" | head -1)"
    [ -n "$url" ] || url="$(curl -fsSL https://ffmpeg.martin-riedl.de/ | grep -oE "/download/macos/arm64/[^\"]*/$tool\.zip" | head -1)"
    [ -n "$url" ] || { echo "couldn't find a $tool build to download"; exit 1; }
    curl -fsSL "https://ffmpeg.martin-riedl.de$url" -o "$CACHE/$tool.zip"
    want="$(curl -fsSL "https://ffmpeg.martin-riedl.de$url.sha256" | awk '{print $1}')"
    got="$(shasum -a 256 "$CACHE/$tool.zip" | awk '{print $1}')"
    [ "$want" = "$got" ] || { echo "$tool download failed its checksum"; exit 1; }
    unzip -o -q "$CACHE/$tool.zip" -d "$CACHE" && rm "$CACHE/$tool.zip"
    chmod +x "$CACHE/$tool"
  fi
done
cp "$CACHE/uv" "$CACHE/ffmpeg" "$CACHE/ffprobe" "$RES/bin/"
cp app/VideoBuilder/firstcut-mcp "$RES/bin/firstcut-mcp" && chmod +x "$RES/bin/firstcut-mcp"     # FirstCut for AI apps (Settings › Connect an AI)
swiftc -O -target arm64-apple-macos13 clipfarm/clipper/cf_ocr.swift -o "$RES/bin/cf_ocr"     # the text reader, ready-made (the app never compiles)
xattr -cr "$RES/bin" 2>/dev/null || true

echo "== icon, Info.plist, signature"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
"$PY" app/VideoBuilder/icon.py "$TMP/icon.png"
mkdir -p "$TMP/AppIcon.iconset"
for s in 16 32 128 256 512; do
  sips -z $s $s "$TMP/icon.png" --out "$TMP/AppIcon.iconset/icon_${s}x${s}.png" >/dev/null
  sips -z $((s * 2)) $((s * 2)) "$TMP/icon.png" --out "$TMP/AppIcon.iconset/icon_${s}x${s}@2x.png" >/dev/null
done
iconutil -c icns "$TMP/AppIcon.iconset" -o "$RES/AppIcon.icns"
cat > "$APP/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>FirstCut</string>
  <key>CFBundleDisplayName</key><string>FirstCut</string>
  <key>CFBundleIdentifier</key><string>app.firstcut.mac</string>
  <key>CFBundleExecutable</key><string>FirstCut</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>$VERSION</string>
  <key>CFBundleVersion</key><string>$BUILD</string>
  <key>LSMinimumSystemVersion</key><string>13.0</string>
  <key>LSArchitecturePriority</key><array><string>arm64</string></array>
  <key>LSApplicationCategoryType</key><string>public.app-category.video</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSAppTransportSecurity</key><dict><key>NSAllowsLocalNetworking</key><true/></dict>
  <key>SUFeedURL</key><string>$FEED</string>
  <key>SUPublicEDKey</key><string>$ED_KEY</string>
  <key>SUEnableAutomaticChecks</key><true/>
  <key>SUScheduledCheckInterval</key><integer>86400</integer>
  <key>NSMicrophoneUsageDescription</key><string>FirstCut listens only while you hold the microphone button, to type what you say. The words are worked out on this Mac.</string>
</dict>
</plist>
EOF
# Signing: with a "Developer ID Application" certificate in the Keychain, every program in the app is signed with the hardened runtime
# and the app + disk image are notarized by Apple (so they open without the "Open Anyway" step). Without one: signed only for this Mac.
#   one-time: Xcode > Settings > Accounts > Manage Certificates > + Developer ID Application, then
#             xcrun notarytool store-credentials firstcut-notary --apple-id <apple id> --team-id <team id>
IDENTITY="${FIRSTCUT_SIGN:-$(security find-identity -v -p codesigning | sed -n 's/.*"\(Developer ID Application: [^"]*\)".*/\1/p' | head -1)}"
NOTARY="${FIRSTCUT_NOTARY_PROFILE:-firstcut-notary}"
if [ -n "$IDENTITY" ]; then
  echo "== signing with $IDENTITY"
  cat > "$CACHE/entitlements.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>com.apple.security.device.audio-input</key><true/>
</dict></plist>
PLIST
  find "$APP/Contents/Resources" -type f -perm -u+x | while read -r f; do
    if file -b "$f" | grep -q "Mach-O"; then codesign --force --timestamp --options runtime -s "$IDENTITY" "$f"; fi
  done
  SPF="$APP/Contents/Frameworks/Sparkle.framework"                 # Sparkle inside-out (its own helpers first), as its docs sign it
  codesign --force --timestamp --options runtime -s "$IDENTITY" "$SPF/Versions/B/Autoupdate"
  codesign --force --timestamp --options runtime -s "$IDENTITY" "$SPF/Versions/B/Updater.app"
  codesign --force --timestamp --options runtime -s "$IDENTITY" "$SPF"
  codesign --force --timestamp --options runtime --entitlements "$CACHE/entitlements.plist" -s "$IDENTITY" "$APP"
  codesign --verify --deep --strict "$APP"
  echo "== notarizing the app (a few minutes)"
  ditto -c -k --keepParent "$APP" "$CACHE/notarize.zip"
  xcrun notarytool submit "$CACHE/notarize.zip" --keychain-profile "$NOTARY" --wait
  xcrun stapler staple "$APP"
  rm -f "$CACHE/notarize.zip"
else
  echo "== no Developer ID certificate: signed for this Mac only (others will need System Settings > Privacy & Security > Open Anyway)"
  codesign --force --deep -s - "$APP"
fi

echo "== the zip"
cat > "$OUT/Read me first.txt" <<'EOF'
FirstCut (for Macs with Apple silicon: M1 or newer)

1. Unzip, and drag "FirstCut" into your Applications folder.
2. Open it. It isn't from the App Store, so the first time macOS stops it once:
   - If you see an "Open" button: click it.
   - If you only see "Done" or "Move to Trash": click Done, open System Settings > Privacy & Security,
     scroll down to "FirstCut was blocked", click "Open Anyway", then open FirstCut again.
   (Right-click > Open also works on older macOS versions.)
3. It gets itself ready (a minute or two), then Setup downloads the models it needs and asks which AI should make the
   editing decisions: your ChatGPT account, your Claude account, or a model on this Mac. Setup shows the exact size first:
   about 6 GB on an 8 GB Mac and 10 GB on a 16 GB Mac (less if you turn off the local fallback).
4. The first time it opens a video on an external drive or in Desktop / Documents / Downloads, macOS asks whether
   FirstCut may use those files: click Allow.

Everything it downloads lives in ~/Library/Application Support/FirstCut and the models folder you choose in Setup.
To remove it: delete the app and those folders.
EOF
ditto -c -k --sequesterRsrc --keepParent "$APP" "$TMP/app.zip"
(cd "$TMP" && mkdir pack && cp "$OUT/Read me first.txt" pack/ && ditto -x -k app.zip pack/ && ditto -c -k --sequesterRsrc pack "$OUT/FirstCut.zip")
echo "== the disk image (for downloading: drag FirstCut onto Applications)"
rm -rf "$TMP/dmg" "$OUT/FirstCut.dmg"
mkdir "$TMP/dmg"
ditto "$APP" "$TMP/dmg/FirstCut.app"
cp "$OUT/Read me first.txt" "$TMP/dmg/"
sed -i '' 's/^1\. Unzip, and drag/1. Drag/' "$TMP/dmg/Read me first.txt"
ln -s /Applications "$TMP/dmg/Applications"
hdiutil create -quiet -volname "FirstCut" -srcfolder "$TMP/dmg" -fs HFS+ -format UDZO -ov "$OUT/FirstCut.dmg"
if [ -n "$IDENTITY" ]; then
  codesign --force --timestamp -s "$IDENTITY" "$OUT/FirstCut.dmg"
  echo "== notarizing the disk image"
  xcrun notarytool submit "$OUT/FirstCut.dmg" --keychain-profile "$NOTARY" --wait
  xcrun stapler staple "$OUT/FirstCut.dmg"
  spctl -a -t open --context context:primary-signature -v "$OUT/FirstCut.dmg"
fi
du -sh "$APP" "$OUT/FirstCut.zip" "$OUT/FirstCut.dmg"
echo "built $BUILD: $OUT/FirstCut.zip and $OUT/FirstCut.dmg"
