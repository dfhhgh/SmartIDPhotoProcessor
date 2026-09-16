"""FastAPI application entrypoint for SmartIDPhotoProcessor.

Usage:
    uvicorn smartidphotoprocessor.api.app:app

The ``app`` object is the ASGI application entry point.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI

from smartidphotoprocessor.api.routes import router

logger = logging.getLogger(__name__)

app = FastAPI(
    title="SmartIDPhotoProcessor",
    description="AI-powered ID photo validation and processing service",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.include_router(router)


@app.on_event("startup")
async def startup_event():
    """Log application startup."""
    logger.info("SmartIDPhotoProcessor API starting up.")
