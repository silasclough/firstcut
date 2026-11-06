# How FirstCut is built

FirstCut is two parts: a small macOS window written in Swift, and an engine written in Python that does the work. The engine's
package is called `clipfarm` for historical reasons.

## The app shell (`app/VideoBuilder/`)
- `main.swift`: the window (a WebKit view of the engine's page), the Dock progress bar, drag and drop, the microphone permission,
  and starting the engine (`python -m clipfarm studio`) with the app's own Python from `~/Library/Application Support/FirstCut`.
- `updates.swift`: in-app updates with Sparkle.
- `bootstrap.sh`: on first launch, installs the app's own Python and the packages in `requirements-app.txt` with uv.
- `firstcut-mcp`: the launcher AI apps start to use FirstCut's MCP server.

## The engine (`clipfarm/`)
- `clipper/server.py` serves the page and the local API on 127.0.0.1; `clipper/studio_web.py` is the app's API (`/api/app`);
  `clipper/studio.py` holds settings, first-run Setup (model downloads sized to the Mac's memory) and the AI sign-ins.
- **Long videos** (`clipper/localproj.py`): a footage folder is scanned and read where it is (nothing is copied), recorders are
  synced to cameras, speech is transcribed on the Mac (Parakeet), the AI plans the cut, B-roll and photos are placed over it,
  the sound is graded and levelled, and the result is rendered with ffmpeg and exported to Final Cut Pro (`clipper/fcpxml.py`).
- **Short clips** (`clipper/engine.py`, `clipper/moments.py`): a long video or a link becomes vertical clips with captions.
- **Footage library** (`clipper/catalog.py`): every shot's frames are embedded with SigLIP 2 for plain-words search, and
  described by the chosen AI.
- **Music** (`clipper/musiclib.py`, `clipper/musicsources.py`): your music, searchable by description (CLAP), and Openverse.
- **AI routing** (`clipper/localai.py`): the person's Claude or ChatGPT plan, or local models through Ollama; offline, the local
  model takes over when it is installed.
- **Graphics** (`clipper/gfxcards.py`, `clipper/skiagfx.py`, `clipper/graphics.py`): cards, lower thirds and animated charts
  drawn with Pillow and Skia, composited by ffmpeg.
- **Transcripts** (`clipper/transcript.py`) and the **MCP server** (`clipper/mcp_server.py`).

## State
Everything is files: settings in `settings.json`, each job in its own folder with a `job.json`, each project's working files in
the project folder's `.aivideo/`. Steps write to a temporary name and rename when done, so an interrupted job resumes.

## Building
`scripts/package_videobuilder.sh` builds `dist/FirstCut.app`, signs and notarizes it when a Developer ID certificate is in the
Keychain, and makes `FirstCut.zip` and `FirstCut.dmg`. `scripts/release.sh` prepares a release with its update feed.
