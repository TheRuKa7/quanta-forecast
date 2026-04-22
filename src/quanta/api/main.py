"""FastAPI entry point for quanta-forecast."""
from __future__ import annotations

from fastapi import FastAPI

from quanta import __version__

app = FastAPI(title="quanta-forecast", version=__version__)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.get("/")
async def root() -> dict[str, str]:
    return {
        "service": "quanta-forecast",
        "version": __version__,
        "repo": "https://github.com/TheRuKa7/quanta-forecast",
    }
