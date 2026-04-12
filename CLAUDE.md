# CLAUDE.md

## Project Overview
Tweet sentiment analysis using a BERT-based model. This repository contains the ML pipeline for training/inference, a FastAPI backend for serving predictions, and a React (Vite) frontend.

## Setup and Commands

### Environment Setup
```bash
uv venv
source .venv/bin/activate
pip install -e .
cd app && npm install
```

### Running the Application
- Backend API: `uvicorn api.main:app --reload`
- Frontend: `cd app && npm run dev`

### Linting and Formatting
- Python: `ruff check .` / `ruff format .`
- React: `cd app && npm run lint`

## Code Style Guidelines

### Python (PEP 8)
- Naming: `snake_case` for modules/functions/variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.
- Docstrings: Google-style (Args, Returns, Raises).
- Type Annotations: Required for all function parameters and return types.
- Line Length: 88 characters.
- Imports: Order by standard library, third-party, and internal modules.

### React (ES6)
- Use ES6+ syntax (arrow functions, destructuring, template literals).
- Components: Use functional components with hooks.
- Naming: `PascalCase` for components, `camelCase` for variables and props.
- Imports: Group by external libraries and internal components.

## Architecture
- `src/`: Core logic for BERT model training, evaluation, and data processing.
- `api/`: FastAPI implementation including routers, schemas, and dependencies.
- `app/`: React frontend project using Vite and ES6 standards.
- `config.yaml`: Configuration for model parameters and training paths.
