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
2. Install dependencies:
   ```bash
   uv pip install -r requirements.txt
   ```
3. Run the tests:
   ```bash
   uv python -m pytest
   ```

## What to change next
- Expand `src/support_bot` with richer loaders, connectors, and prompt logic.
- Add more knowledge blobs inside `data/` and source them via configuration, or fetch them from a live API.
- Cover edge cases in `test/test_bot.py`, including empty inputs or malformed JSON.