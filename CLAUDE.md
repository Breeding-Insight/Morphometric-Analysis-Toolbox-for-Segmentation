# MATS — Claude Code

Leaf morphometrics from photographs of a printed calibration template: detect
markers → perspective-correct → segment → measure → CSV.

Project instructions: @AGENTS.md

<!--
Maintainers: this file is public and committed, and it exists mainly so Claude
Code loads AGENTS.md (when a CLAUDE.md is present, AGENTS.md is read only
through an import like the one above). Keep the shared content in AGENTS.md so
every agent gets it; keep private working notes in CLAUDE.local.md, which is
gitignored and loads after this file. HTML comments are stripped before this
file enters the context window, so notes here cost nothing.
-->

## Claude-specific notes

- Run `pytest` before proposing code changes — the suite is offline and takes
  seconds. Do not add a module-level torch/streamlit import to a path it covers.
- Diagnose environment problems with `mats doctor` before reading code.
- `README.md` and `docs/` are user-facing and public: no internal planning,
  unpublished results, or private paths belong in them.
- Never commit checkpoints (`*.pth`, `*.pt`) or a user's images.
