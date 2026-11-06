#!/bin/bash
# FirstCut's first run (and the first run after an update): the app's own Python and the engine's packages, made with the bundled
# `uv` inside the app's home folder, so nothing touches the Mac's own Python. Prints "PROGRESS <percent> <text>" lines for the window.
#   bootstrap.sh <uv> <env dir> <requirements file> <build id> <home>
set -euo pipefail
UV="$1"; ENV="$2"; REQ="$3"; BUILD="$4"; HOME_DIR="$5"
export UV_PYTHON_INSTALL_DIR="$HOME_DIR/runtime/python" UV_CACHE_DIR="$HOME_DIR/runtime/uv-cache" UV_NO_CONFIG=1
mkdir -p "$HOME_DIR/runtime" "$HOME_DIR/logs"
echo "PROGRESS 5 Getting Python ready"
"$UV" python install 3.12 --quiet
echo "PROGRESS 25 Making the engine's own Python"
if [ ! -x "$ENV/bin/python" ]; then
  "$UV" venv --python 3.12 --quiet "$ENV"
fi
echo "PROGRESS 40 Installing the video engine (about 150 MB)"
"$UV" pip install --python "$ENV/bin/python" --quiet -r "$REQ"
echo "PROGRESS 95 Checking it works"
"$ENV/bin/python" -c "import yaml, numpy, PIL, cv2, requests, yt_dlp, huggingface_hub" >/dev/null
echo "$BUILD" > "$ENV/.vb-build"
echo "PROGRESS 100 Ready"
