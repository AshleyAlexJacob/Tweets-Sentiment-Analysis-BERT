# CLAUDE.md

## Project overview

Tweet sentiment analysis using a BERT-based model. This repository contains the ML pipeline for training and inference, a FastAPI backend for serving predictions, and a React (Vite) frontend.

## Setup and commands

### Environment setup

```bash
uv venv
source .venv/bin/activate
uv sync
# or: pip install -e .
cd app && npm install
```

### Running the application

- Backend API: `uvicorn api.main:app --reload`
- Frontend: `cd app && npm run dev`

### Data preprocessing

- Default config is **dev** (small model): `python -m src.pipelines.data_preprocessing`
- **Prod** (larger model, e.g. Kaggle): `APP_ENV=prod python -m src.pipelines.data_preprocessing`
- **Explicit config file**: `CONFIG_PATH=prod.config.yaml python -m src.pipelines.data_preprocessing`

Preprocessing cleans text, maps Sentiment140 labels (0/2/4) to 3 classes, and writes `train_<stem>.csv`, `val_<stem>.csv`, and `test_<stem>.csv` under `data.processed_path` (split fractions: `train_fraction`, `val_fraction`, `test_fraction`).

### Model training and evaluation

- Train: `python -m src.pipelines.model_training` (same `APP_ENV` / `CONFIG_PATH` rules as preprocessing).
- Evaluate on the test split: `python -m src.pipelines.model_evaluation` (loads `best_model.pt` from `model.checkpoint_path`, or `model.eval_checkpoint` if set).

Config resolution (see `src.utils.load_config`): optional `CONFIG_PATH`, then `APP_ENV` (`prod` → `prod.config.yaml`, else `dev.config.yaml`), else fallback `config.yaml`.

### Linting and formatting

- Python: `ruff check .` / `ruff format .`
- React: `cd app && npm run lint`

## Code style guidelines

### Python (PEP 8)

- Naming: `snake_case` for modules/functions/variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.
- Docstrings: Google-style (Args, Returns, Raises).
- Type annotations: required for all function parameters and return types.
- Line length: 88 characters.
- Imports: order by standard library, third-party, and internal modules.

### React (ES6)
- Use ES6+ syntax (arrow functions, destructuring, template literals).
- Components: use functional components with hooks.
- Naming: `PascalCase` for components, `camelCase` for variables and props.
- Imports: group by external libraries and internal components.

## Architecture

- `src/`: core logic for BERT training, evaluation, and data processing.
- `api/`: FastAPI implementation including routers, schemas, and dependencies.
- `app/`: React frontend using Vite and ES6 standards.
- `dev.config.yaml`: local development (small BERT for fast iteration).
- `prod.config.yaml`: heavier training (e.g. Kaggle / full fine-tune).
- `config.yaml`: legacy fallback if env-specific files are absent.

### Data processing modules

- `src/data/cleaner.py`: tweet text cleaning (`TweetCleaner`).
- `src/data/preprocessor.py`: tokenizer setup and dataframe cleaning (`TweetPreprocessor`).
- `src/data/loader.py`: CSV loading and `DataLoader` construction (`TweetLoader`).
- `src/data/dataset.py`: PyTorch `Dataset` for tokenized batches (`TweetDataset`).
- `src/data/splits.py`: deterministic train/val/test split helpers.
- `src/pipelines/data_preprocessing.py`: end-to-end raw CSV → cleaned + split CSVs.
- `src/pipelines/model_training.py` / `src/pipelines/model_evaluation.py`: training and test evaluation orchestration.
- `src/model/architecture.py`, `src/model/train.py`, `src/model/evaluate.py`: classifier, trainer, and metrics.

These modules use `try` / `except` for I/O, parsing and other failures: errors are logged (where a logger is used), re-raised with context, or in the CLI pipeline reported so a single bad file does not stop the whole run.
