from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routers import sentiment
from api.middleware import setup_error_handlers
from src.utils import load_config

def create_app() -> FastAPI:
    """Application factory for the FastAPI system.

    Returns:
        FastAPI: The initialized FastAPI application.
    """
    config = load_config()
    
    app = FastAPI(
        title=config["project"]["name"],
        version=config["project"]["version"],
        description="REST API for Tweet Sentiment Analysis using BERT",
    )

    # Register middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Setup global error handlers
    setup_error_handlers(app)

    # Register routers
    app.include_router(
        sentiment.router,
        prefix=f"/{config['api']['version']}",
        tags=["sentiment"]
    )

    @app.get("/")
    def root() -> dict[str, str]:
        return {"message": "Welcome to the Tweet Sentiment Analysis API"}

    return app

app = create_app()
