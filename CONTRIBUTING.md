# Contributing

Thanks for wanting to help. FirstCut is a work in progress made by one person, so keep changes small and focused.

- **Bugs:** open an issue with what you did, what you expected, what happened, your Mac (chip and memory) and FirstCut's version
  (Settings › General). The engine's log is in `~/Library/Application Support/FirstCut/logs/`; please remove anything private.
- **Changes:** fork, branch, and open a pull request. Describe the problem first, then the fix.
- **Tests:** `python -m pytest -q tests` must pass. Tests run offline: a test that needs an AI or the network must fake it
  (`tests/conftest.py` blocks real connections and AI programs).
- **Never commit** personal details, API keys, `.env` files, footage or logs. `scripts/package_videobuilder.sh` refuses to build if
  the patterns in your own `.private-patterns` file (gitignored) appear in the app.
- **Interface changes:** keep FirstCut's style: flat, hairline borders, one light-blue accent, the system font, plain words, light
  and dark themes, one clear action per screen. Look at the change in both themes before sending it.

By contributing you agree that your contribution is licensed under the GNU General Public License v3.0, like the rest of FirstCut.
