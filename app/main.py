"""FastAPI application for the Decipher survey base generator.

Phase 1 exposes only the document parser: upload a questionnaire, get back the
structured document model. Later phases add the intermediate survey model, the
local AI classifier, the review UI and XML generation on top of this same app.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes_classify import router as classify_router
from app.api.routes_parse import router as parse_router
from app.config import BASE_DIR, settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

STATIC_DIR = BASE_DIR / "app" / "static"

app = FastAPI(
    title="Decipher Survey Base Generator",
    description="Phase 1 — DOCX parsing and formatting extraction.",
    version="0.1.0",
)

if settings.cors_origins:
    from fastapi.middleware.cors import CORSMiddleware

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(parse_router)
app.include_router(classify_router)


@app.get("/health", tags=["meta"])
async def health() -> dict[str, str]:
    return {"status": "ok", "phase": "1", "version": app.version}


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> FileResponse:
    return FileResponse(STATIC_DIR / "favicon.svg", media_type="image/svg+xml")


def run() -> None:
    """Entry point for ``python -m app.main``."""
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=bool(__debug__),
    )


if __name__ == "__main__":
    run()
