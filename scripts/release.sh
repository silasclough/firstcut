#!/bin/bash
# Publish a FirstCut update (automatic updates: app/VideoBuilder/updates.swift, Sparkle 2).
#   1. bump app/release/VERSION and add its notes at the top of app/release/CHANGES.md
#   2. ./scripts/release.sh            (builds, signs, notarizes, signs the update, writes the appcast; prints the publish command)
#      ./scripts/release.sh --no-build (reuses dist/FirstCut.app from the last package_videobuilder.sh run)
#   3. run the printed command: every installed FirstCut finds the new version within a day (or at once with "Check for updates")
#
# The update is signed with the EdDSA key in this Mac's login Keychain (Sparkle account "firstcut"); installed copies only accept
# updates signed with it (SUPublicEDKey in Info.plist). Losing that key means installed copies can't be updated: back it up with
#   dist/.cache/sparkle-2.10.1/bin/generate_keys --account firstcut -x <file>   (keep the file somewhere safe and private)
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="dist"; REL="$OUT/release"
VERSION="$(tr -d '[:space:]' < app/release/VERSION)"
SPK="$OUT/.cache/sparkle-2.10.1"
REPO="you/firstcut"

[ "${1:-}" = "--no-build" ] || ./scripts/package_videobuilder.sh
APP="$OUT/FirstCut.app"
[ -d "$APP" ] || { echo "no $APP: build first"; exit 1; }
BUILT="$(defaults read "$PWD/$APP/Contents/Info" CFBundleShortVersionString)"
BUILD="$(defaults read "$PWD/$APP/Contents/Info" CFBundleVersion)"
[ "$BUILT" = "$VERSION" ] || { echo "the app in dist/ is $BUILT, VERSION says $VERSION: build again"; exit 1; }
if ! spctl -a -vv "$APP" 2>&1 | grep -q "Notarized Developer ID"; then
  echo "dist/FirstCut.app isn't notarized: people would get the 'Open Anyway' step after updating. Build with the Developer ID first."
  [ "${FIRSTCUT_ALLOW_UNNOTARIZED:-}" = "1" ] || exit 1
fi

rm -rf "$REL"; mkdir -p "$REL"
ZIP="FirstCut-$VERSION.zip"
ditto -c -k --sequesterRsrc --keepParent "$APP" "$REL/$ZIP"          # the update: only the app (Sparkle unpacks it and swaps it in)
cp "$OUT/FirstCut.dmg" "$REL/FirstCut.dmg" 2>/dev/null || true       # the download for new people (unchanged name: the README links it)

SIG="$("$SPK/bin/sign_update" --account firstcut "$REL/$ZIP")"     # -> sparkle:edSignature="…" length="…"
URL="https://github.com/$REPO/releases/download/v$VERSION/$ZIP"
FEED_URL="${FIRSTCUT_RELEASE_BASE:-}"                                # (a test can serve the files itself: the enclosure points there)
[ -n "$FEED_URL" ] && URL="$FEED_URL/$ZIP"

python3 - "$VERSION" "$BUILD" "$URL" "$SIG" "$REL/appcast.xml" <<'PY'
import html, re, sys, email.utils
from pathlib import Path
version, build, url, sig, out = sys.argv[1:]
text = Path("app/release/CHANGES.md").read_text()
m = re.search(rf"^## {re.escape(version)}\s*\n(.*?)(?=^## |\Z)", text, re.S | re.M)
notes = [l.strip()[2:] for l in (m.group(1) if m else "").splitlines() if l.strip().startswith("- ")]
if not notes:
    sys.exit(f"no notes for {version} in app/release/CHANGES.md")
body = "<ul>" + "".join(f"<li>{html.escape(n, quote=False)}</li>" for n in notes) + "</ul>"
attrs = dict(re.findall(r'(sparkle:edSignature|length)="([^"]+)"', sig))
if "sparkle:edSignature" not in attrs:
    sys.exit("sign_update gave no signature")
Path(out).write_text(f'''<?xml version="1.0" encoding="utf-8"?>
<rss version="2.0" xmlns:sparkle="http://www.andymatuschak.org/xml-namespaces/sparkle" xmlns:dc="http://purl.org/dc/elements/1.1/">
  <channel>
    <title>FirstCut</title>
    <item>
      <title>FirstCut {version}</title>
      <pubDate>{email.utils.formatdate(usegmt=True)}</pubDate>
      <sparkle:version>{build}</sparkle:version>
      <sparkle:shortVersionString>{version}</sparkle:shortVersionString>
      <sparkle:minimumSystemVersion>13.0</sparkle:minimumSystemVersion>
      <description><![CDATA[{body}]]></description>
      <enclosure url="{html.escape(url)}" length="{attrs['length']}" type="application/octet-stream" sparkle:edSignature="{attrs['sparkle:edSignature']}"/>
    </item>
  </channel>
</rss>
''')
print(f"appcast: FirstCut {version} (build {build}), {len(notes)} notes")
PY

echo
echo "Ready in $REL: $(ls "$REL" | tr '\n' ' ')"
echo "Publish (gh here is an Intel build that can't run git, so the tag is made through the API):"
echo "  gh release create v$VERSION $REL/FirstCut.dmg $REL/$ZIP $REL/appcast.xml --repo $REPO --target main --title \"FirstCut $VERSION\" --notes-file <(sed -n '/^## $VERSION/,/^## /p' app/release/CHANGES.md | sed '1d;\$d')"
echo "Then check: curl -sL https://github.com/$REPO/releases/latest/download/appcast.xml | head"
