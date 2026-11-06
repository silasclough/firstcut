#!/bin/bash
# Install FirstCut (the app made by scripts/package_videobuilder.sh; it was called Video Builder until 2026-11-05) on THIS Mac, replacing
# any earlier copy (Video Builder.app, and the very first LaunchAgent version com.clipfarm.videobuilder):
#   install_videobuilder.sh <FirstCut.zip | FirstCut.app> [--work] [--apps <folder>]
# --apps: where the app goes (default ~/Applications; /Applications for every account on the Mac).
# --work: the work Mac. Its settings (the footage-drive drive as the footage library, the work Claude plan decides), its footage catalog and
#         the models it already has (Parakeet, the search model) are kept or carried over, and the work assistant's folder ~/FirstCut
#         (CLAUDE.md + bin/firstcut) is refreshed. Without --work nothing outside the app is written: the app makes its own folders.
# A Mac that had Video Builder keeps its "~/Library/Application Support/Video Builder" folder (its Python and models can't move), and
# the app goes on using it. Never touches the personal Clipper or its agents.
set -euo pipefail
SRC="$1"; shift
WORK=""; APPS="$HOME/Applications"
while [ $# -gt 0 ]; do
  case "$1" in
    --work) WORK=1 ;;
    --apps) APPS="$2"; shift ;;
    *) echo "unknown option $1"; exit 2 ;;
  esac
  shift
done
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SUPPORT="$HOME/Library/Application Support"
HOME_DIR="$SUPPORT/FirstCut"                                   # (the same rule as the app: app/VideoBuilder/main.swift homeDir)
if [ ! -e "$HOME_DIR" ] && [ -f "$SUPPORT/Video Builder/settings.json" ]; then HOME_DIR="$SUPPORT/Video Builder"; fi

echo "== remove earlier copies"
launchctl bootout "gui/$(id -u)/com.clipfarm.videobuilder" 2>/dev/null || true
rm -f "$HOME/Library/LaunchAgents/com.clipfarm.videobuilder.plist"
for id in com.clipfarm.videobuilder.app app.videobuilder.mac app.firstcut.mac; do
  osascript -e "tell application id \"$id\" to quit" 2>/dev/null || true
done
sleep 2
for d in "$HOME/Applications" "$APPS"; do
  rm -rf "$d/Video Builder.app" "$d/FirstCut.app"
done

echo "== install the app"
mkdir -p "$APPS"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
if [[ "$SRC" == *.zip ]]; then
  ditto -x -k "$SRC" "$TMP"
  SRC="$(find "$TMP" -maxdepth 2 -name "FirstCut.app" -type d | head -1)"
  [ -n "$SRC" ] || { echo "no FirstCut.app in that zip"; exit 1; }
fi
ditto "$SRC" "$APPS/FirstCut.app"
xattr -dr com.apple.quarantine "$APPS/FirstCut.app" 2>/dev/null || true
echo "   $APPS/FirstCut.app (its folder: $HOME_DIR)"

if [ -n "$WORK" ]; then
  echo "== the work Mac's settings, catalog and models"
  mkdir -p "$HOME_DIR"
  MODELS="$HOME_DIR/Models"
  if [ ! -f "$HOME_DIR/settings.json" ]; then
    cat > "$HOME_DIR/settings.json" <<EOF
{"ai": {"use": "claude", "fallback": true},
 "library": {"folders": ["/Volumes/Footage/Video"], "night": true},
 "places": {"videos": "$HOME/Movies/FirstCut", "models": "$MODELS", "catalog": "$HOME_DIR/Catalog", "work": "$HOME/Library/Caches/FirstCut"},
 "setup": {"done": true}}
EOF
  fi
  OLD="$ROOT/data/work/catalog"
  if [ -d "$OLD" ] && [ ! -d "$HOME_DIR/Catalog/files" ]; then
    mkdir -p "$HOME_DIR/Catalog" && cp -R "$OLD"/. "$HOME_DIR/Catalog"/ && echo "   catalog copied from $OLD"
  fi
  mkdir -p "$MODELS/huggingface/hub"
  for m in models--mlx-community--parakeet-tdt-0.6b-v3 models--google--siglip2-base-patch16-224; do
    if [ -d "$HOME/.cache/huggingface/hub/$m" ] && [ ! -d "$MODELS/huggingface/hub/$m" ]; then
      cp -R "$HOME/.cache/huggingface/hub/$m" "$MODELS/huggingface/hub/" && echo "   model $m copied"
    fi
  done

  echo "== the work assistant's folder (~/FirstCut)"
  A="$HOME/FirstCut"
  if [ -d "$HOME/VideoBuilder" ] && [ ! -L "$HOME/VideoBuilder" ] && [ ! -e "$A" ]; then
    mv "$HOME/VideoBuilder" "$A" && ln -s "$A" "$HOME/VideoBuilder" && echo "   ~/VideoBuilder moved to ~/FirstCut (the old name still works)"
  fi
  mkdir -p "$A/bin"
  [ -f "$ROOT/handoff/videobuilder/CLAUDE.md" ] && cp "$ROOT/handoff/videobuilder/CLAUDE.md" "$A/CLAUDE.md"
  [ -f "$ROOT/handoff/videobuilder/bin/vb" ] && cp "$ROOT/handoff/videobuilder/bin/vb" "$A/bin/firstcut"
  chmod +x "$A/bin/firstcut"
  ln -sf firstcut "$A/bin/vb"                                   # (its first name)
  [ -f "$A/ISSUES.md" ] || printf '# Problems for the code owner\n\nNewest first. Date, what you did, what happened (the exact error), what you expected.\n\n' > "$A/ISSUES.md"
  [ -f "$A/NOTES.md" ] || printf '# Notes\n\nThings learned while using FirstCut (yours to keep).\n\n' > "$A/NOTES.md"
fi
echo "done. Open FirstCut from $APPS (drag it to the Dock to keep it there)."
