"""
Engine-level defaults and shared utilities for the VLM pipeline.

This module contains **no project-specific schemas, prompts, or column lists**.
Those are defined by each project (see ``vlm_project.ProjectSpec``).

What lives here:

* Default values for retry / rate-limit / file-polling parameters that every
  project inherits unless it overrides them in its ``project.toml``.
* ``suppress_gemini_sdk_warnings`` — silences harmless SDK noise.
"""

from __future__ import annotations

import builtins
import warnings

# ── Default engine constants (projects override via their TOML) ──────────────

DEFAULT_PRIMARY_MODEL = "gemini-3.1-pro-preview"
DEFAULT_FALLBACK_MODEL = "gemini-3-flash-preview"
DEFAULT_RETRY_MAX_ATTEMPTS = 3
DEFAULT_RETRY_BASE_DELAY_SEC = 2
DEFAULT_RATE_LIMIT_MARGIN_SEC = 5
DEFAULT_FILE_ACTIVE_MAX_WAIT_SEC = 60
DEFAULT_FILE_POLL_INTERVAL_SEC = 1
DEFAULT_SHEETS_APPEND_CHUNK_SIZE = 100


def suppress_gemini_sdk_warnings() -> None:
    """Suppress harmless Gemini SDK warnings about non-text parts and thought signatures.

    Newer Gemini models may include ``thought_signature`` parts alongside
    standard text; the SDK emits ``UserWarning`` for these, but the pipeline
    only parses text parts so the warnings are noise.
    """
    warnings.filterwarnings(
        "ignore", message=r".*non-text parts.*", category=builtins.UserWarning
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*thought_signature.*",
        category=builtins.UserWarning,
    )
