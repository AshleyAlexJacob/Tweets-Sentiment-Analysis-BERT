# Frontend (Vite + React)

## Run with the FastAPI backend

1. Start the API (repo root), e.g. `uvicorn api.main:app --reload --port 8000`.
2. From this directory: `npm install` then `npm run dev`.

`vite.config.js` proxies `/v1` to `http://127.0.0.1:8000`, so the browser calls `POST /v1/predict` on the same origin as Vite and requests reach FastAPI without CORS setup.

## Production build

Set `VITE_API_BASE_URL` to your deployed API origin (no trailing slash), then `npm run build`. The app will call `${VITE_API_BASE_URL}/v1/predict`.

See `.env.example`.
