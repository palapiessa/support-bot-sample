# Support Bot Sample

Minimal Python support bot sample showing how to structure code, data, and tests.

## Project layout

- `support_bot/src/` – application package (`src`), keeps bot logic and helpers.
- `support_bot/test/` – `pytest` suite for unit and regression behavior checks.
- `data/` – static knowledge bases or datasets consumed by the bot.
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
   PYTHONPATH=support_bot .venv/bin/python -m pytest support_bot/test
   ```
4. Run the CLI app
   - use VSCode debug, launch.json in code repo
   - or cmdline from support-bot-sample root folder
   ```bash
   PYTHONPATH=support_bot .venv/bin/python -m src.cli
   ```