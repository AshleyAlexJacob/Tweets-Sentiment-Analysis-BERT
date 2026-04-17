from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


def setup_error_handlers(app: FastAPI) -> None:
    """Registers global exception handlers for the application.

    Args:
        app (FastAPI): The FastAPI instance.
    """

    @app.exception_handler(Exception)
    async def global_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """Handles all unhandled exceptions.

        Args:
            request (Request): The incoming request.
            exc (Exception): The raised exception.

        Returns:
            JSONResponse: Standard error response.
        """
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "An internal server error occurred."},
        )
