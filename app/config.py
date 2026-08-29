"""Runtime settings, overridable through environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    host: str = os.environ.get("DECIPHER_HOST", "0.0.0.0")
    port: int = _int_env("DECIPHER_PORT", 8000)

    max_upload_bytes: int = _int_env("DECIPHER_MAX_UPLOAD_MB", 25) * 1024 * 1024
    allowed_extensions: tuple[str, ...] = (".docx",)

    upload_dir: Path = BASE_DIR / "uploads"
    keep_uploads: bool = os.environ.get("DECIPHER_KEEP_UPLOADS", "").lower() in {"1", "true", "yes"}
    """Off by default: questionnaires are client-confidential, so nothing is
    written to disk unless it is asked for explicitly."""

    cors_origins: tuple[str, ...] = tuple(
        origin.strip()
        for origin in os.environ.get("DECIPHER_CORS_ORIGINS", "").split(",")
        if origin.strip()
    )

    # -- local AI (Phase 2) --
    ollama_url: str = os.environ.get("DECIPHER_OLLAMA_URL", "http://localhost:11434")
    ollama_model: str = os.environ.get("DECIPHER_OLLAMA_MODEL", "llama3.1")
    ollama_timeout: float = float(os.environ.get("DECIPHER_OLLAMA_TIMEOUT", "60"))
    review_threshold: float = float(os.environ.get("DECIPHER_REVIEW_THRESHOLD", "0.75"))
    #: Projected remaining time above which the review screen suggests batching.
    slow_classify_seconds: float = float(os.environ.get("DECIPHER_SLOW_CLASSIFY_SECONDS", "150"))

    #: Where confirmed corrections are kept between runs. Set to an empty
    #: string to keep them in memory only.
    correction_library: str = os.environ.get(
        "DECIPHER_CORRECTION_LIBRARY", str(BASE_DIR / "data" / "corrections.json")
    )

    # -- resource tags (Phase 5) --
    #: Any file containing <res label="...">...</res> entries. Point this at the
    #: team's Standard_Template_Questions XML; the bundled file is a stand-in.
    template_xml: Path = Path(
        os.environ.get("DECIPHER_TEMPLATE_XML", str(BASE_DIR / "reference" / "res_catalog.xml"))
    )

    # -- XML export (Phase 3) --
    #: Placeholder namespace URIs for the survey root. The generated *fragments*
    #: use these prefixes without declaring them, which is correct — the team's
    #: canonical survey root declares them. Override before shipping exports
    #: that must open standalone.
    xmlns_atm1d: str = os.environ.get("DECIPHER_XMLNS_ATM1D", "http://decipherinc.com/atm1d")
    xmlns_ss: str = os.environ.get("DECIPHER_XMLNS_SS", "http://decipherinc.com/ss")


settings = Settings()
