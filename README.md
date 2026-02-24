# Support Bot Sample

Minimal Python support bot sample showing how to structure code, data, and tests.

## Project layout

- `src/` – application package (`support_bot`), keeps business logic and helpers.
- `data/` – static knowledge bases or datasets consumed by the bot.
- `test/` – `pytest` suite targeting the behavior described by the knowledge data.
- `.venv/` – optional virtual environment (already present) to isolate dependencies.

## Getting started

1. Activate the provided virtual environment (macOS zsh):
   ```bash
   uv venv
   source .venv/bin/activate
   ```
   (Win):
   ```PowerShell
   .\venv\Scripts\Activate.ps1
   ```

2. Install dependencies:
   ```bash
   uv pip install -r requirements.txt
   ```
3. Run the tests:
   ```bash
   uv python -m pytest
   ```
4. Run the CLI app
   - use VSCode debug, launch.json in code repo
   - or cmdline
   ```bash
   cd src
   ../.venv/bin/python -m support_bot.cli
   ```