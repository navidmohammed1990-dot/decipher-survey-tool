"""Parsing endpoints."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from io import BytesIO

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.config import settings
from app.models.document import ParsedDocument
from app.parsing.docx_parser import DocxParseError, parse_docx

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["parse"])


def _validate_upload(upload: UploadFile) -> str:
    filename = (upload.filename or "").strip()
    if not filename:
        raise HTTPException(status_code=400, detail="No filename supplied.")

    suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if suffix not in settings.allowed_extensions:
        allowed = ", ".join(settings.allowed_extensions)
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{suffix or filename}'. Expected: {allowed}.",
        )
    return filename


@router.post("/parse", response_model=ParsedDocument)
async def parse_upload(file: UploadFile = File(...)) -> ParsedDocument:
    """Parse an uploaded questionnaire into the Phase 1 document model."""
    filename = _validate_upload(file)

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(payload) > settings.max_upload_bytes:
        limit_mb = settings.max_upload_bytes // (1024 * 1024)
        raise HTTPException(
            status_code=413, detail=f"File exceeds the {limit_mb} MB upload limit."
        )

    if settings.keep_uploads:
        _persist(filename, payload)

    try:
        return parse_docx(BytesIO(payload), filename=filename)
    except DocxParseError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - unexpected parser failure
        logger.exception("Unhandled error parsing %s", filename)
        raise HTTPException(
            status_code=500, detail=f"Failed to parse document: {exc}"
        ) from exc


def _persist(filename: str, payload: bytes) -> None:
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    safe_name = filename.replace("/", "_").replace("\\", "_")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    (settings.upload_dir / f"{stamp}-{safe_name}").write_bytes(payload)
