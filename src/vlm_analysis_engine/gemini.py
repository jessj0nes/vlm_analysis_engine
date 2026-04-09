"""
Generic Gemini API call wrappers — schema-agnostic.

Every function that talks to the Gemini API lives here.  The caller passes the
``schema_cls`` (any ``@dataclass``) and ``prompt`` text; nothing in this module
knows about a particular analysis or annotation schema.

Retry / rate-limit behaviour is controlled by a :class:`RetryConfig` instance
so that each project can tune the parameters independently.
"""

from __future__ import annotations

import dataclasses
import json
import mimetypes
import os
import re
import time
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Optional, Tuple, Union, get_args, get_origin, get_type_hints

import pandas as pd
from google import genai
from google.genai import types as genai_types

from .media import (
    download_media,
    normalize_download_url,
    url_to_media_save_name,
)

from .core import (
    DEFAULT_FILE_ACTIVE_MAX_WAIT_SEC,
    DEFAULT_FILE_POLL_INTERVAL_SEC,
    DEFAULT_RATE_LIMIT_MARGIN_SEC,
    DEFAULT_RETRY_BASE_DELAY_SEC,
    DEFAULT_RETRY_MAX_ATTEMPTS,
)

# ═════════════════════════════════════════════════════════════════════════════
# Retry configuration
# ═════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class RetryConfig:
    """Tuneable parameters for Gemini API retries and file-upload polling."""

    max_attempts: int = DEFAULT_RETRY_MAX_ATTEMPTS
    base_delay_sec: int = DEFAULT_RETRY_BASE_DELAY_SEC
    rate_limit_margin_sec: int = DEFAULT_RATE_LIMIT_MARGIN_SEC
    file_active_max_wait_sec: int = DEFAULT_FILE_ACTIVE_MAX_WAIT_SEC
    file_poll_interval_sec: int = DEFAULT_FILE_POLL_INTERVAL_SEC


_DEFAULT_RETRY = RetryConfig()

# ═════════════════════════════════════════════════════════════════════════════
# Error classification
# ═════════════════════════════════════════════════════════════════════════════


def is_retryable_api_error(err: Exception) -> bool:
    """True if the error suggests a retry might help (500, JSON conversion failure)."""
    msg = str(err).lower()
    return (
        "500" in msg
        or "internal" in msg
        or "failed to convert server response to json" in msg
    )


def is_resource_exhausted_error(err: Exception) -> bool:
    """True if the error indicates resource / quota exhaustion (429, rate limit)."""
    msg = str(err).lower()
    return (
        "429" in msg
        or "resource exhausted" in msg
        or "resource_exhausted" in msg
        or "quota" in msg
        or "rate limit" in msg
    )


def get_retry_delay_seconds(err: Exception) -> Optional[float]:
    """Extract retry delay in seconds from a Gemini API error message.

    Handles ``"retry in 35.28s"`` and ``"retry in 14h30m13.78s"`` formats.
    Returns ``None`` when no delay can be parsed.
    """
    msg = str(err)
    m = re.search(r"retry in (\d+\.?\d*)s\b", msg, re.IGNORECASE)
    if m:
        return float(m.group(1))
    m = re.search(r"retry in (\d+)h(\d+)m(\d+\.?\d*)s", msg, re.IGNORECASE)
    if m:
        return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    return None


def is_per_minute_rate_limit(err: Exception) -> bool:
    """True for a short-window rate limit (seconds) rather than a daily quota."""
    if not is_resource_exhausted_error(err):
        return False
    msg = str(err).lower()
    if "per_day" in msg or "perday" in msg:
        return False
    delay = get_retry_delay_seconds(err)
    return delay is not None and delay < 3600


def is_daily_quota_resource_error(err: Optional[Union[Exception, str]]) -> bool:
    """True for 429 / quota errors indicating a daily (or multi-hour) limit."""
    if err is None:
        return False
    exc = err if isinstance(err, Exception) else Exception(str(err))
    if not is_resource_exhausted_error(exc):
        return False
    msg = str(err).lower()
    if "per_day" in msg or "perday" in msg or "per-day" in msg:
        return True
    if "per day" in msg and ("quota" in msg or "limit" in msg):
        return True
    delay = get_retry_delay_seconds(exc)
    if delay is not None and delay >= 3600:
        return True
    return False


# ═════════════════════════════════════════════════════════════════════════════
# Response parsing (generic)
# ═════════════════════════════════════════════════════════════════════════════


def extract_text_from_response(response: Any) -> Optional[str]:
    """Extract concatenated text from a Gemini ``generate_content`` response.

    Skips non-text parts (e.g. ``thought_signature``) that newer models emit.
    """
    chunks: list[str] = []
    for cand in getattr(response, "candidates", None) or []:
        content = getattr(cand, "content", None)
        if not content:
            continue
        for part in getattr(content, "parts", []) or []:
            t = getattr(part, "text", None)
            if t:
                chunks.append(t)
    return "".join(chunks) if chunks else None


def _unwrap_optional(tp: Any) -> Any:
    """Return the inner type of ``Optional[X]`` / ``X | None``, or *tp* unchanged."""
    origin = get_origin(tp)
    if origin is Union:
        args = [a for a in get_args(tp) if a is not type(None)]
        return args[0] if len(args) == 1 else tp
    return tp


def _coerce_field_value(raw: Any, field_type: Any) -> Any:
    """Best-effort coercion of a JSON value to the expected field type."""
    field_type = _unwrap_optional(field_type)

    if raw is None:
        return None

    if isinstance(field_type, type) and dataclasses.is_dataclass(field_type):
        if isinstance(raw, dict):
            return parse_response_dict(raw, field_type)
        return raw

    if isinstance(field_type, type) and issubclass(field_type, Enum):
        if isinstance(raw, field_type):
            return raw
        try:
            return field_type(str(raw).upper())
        except (ValueError, KeyError):
            try:
                return field_type(str(raw))
            except (ValueError, KeyError):
                members = list(field_type)
                return members[0] if members else raw

    if field_type is bool:
        return bool(raw)

    if field_type is str:
        return str(raw)

    return raw


def parse_response_dict(data: dict, cls: type) -> Any:
    """Build any ``@dataclass`` instance from a JSON dict with type coercion.

    Handles enums (by upper-cased value), bools, strings, and nested
    dataclasses recursively.  Missing fields receive ``None`` or the
    dataclass-declared default.
    """
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for f in dataclasses.fields(cls):
        raw = data.get(f.name)
        ft = hints.get(f.name, str)
        if raw is not None:
            kwargs[f.name] = _coerce_field_value(raw, ft)
        elif f.default is not dataclasses.MISSING:
            kwargs[f.name] = f.default
        elif f.default_factory is not dataclasses.MISSING:  # type: ignore[union-attr]
            kwargs[f.name] = f.default_factory()  # type: ignore[misc]
        else:
            kwargs[f.name] = None
    return cls(**kwargs)


# ═════════════════════════════════════════════════════════════════════════════
# Gemini API calls (schema-agnostic)
# ═════════════════════════════════════════════════════════════════════════════


def _do_generate(
    client: genai.Client,
    use_model: str,
    contents: list,
    schema_cls: type,
    rc: RetryConfig,
) -> Tuple[Any, Optional[Union[Exception, str]]]:
    """Inner retry loop shared by media and URL call paths."""
    last_error: Optional[Exception] = None
    for attempt in range(rc.max_attempts):
        try:
            response = client.models.generate_content(
                model=use_model,
                contents=contents,
                config=genai_types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema_cls,
                ),
            )
            parsed = getattr(response, "parsed", None)
            if isinstance(parsed, schema_cls):
                return parsed, None
            text = extract_text_from_response(response)
            if not text:
                return None, Exception("Empty response from Gemini")
            data = json.loads(text)
            if isinstance(data, dict):
                return parse_response_dict(data, schema_cls), None
            return None, Exception("Unexpected response JSON structure")
        except Exception as e:
            last_error = e
            if is_per_minute_rate_limit(e):
                delay = get_retry_delay_seconds(e)
                if delay is not None:
                    time.sleep(delay + rc.rate_limit_margin_sec)
                    continue
            if is_retryable_api_error(e) and attempt < rc.max_attempts - 1:
                time.sleep(rc.base_delay_sec * (2 ** attempt))
                continue
            break
    return None, last_error


def call_gemini_for_media(
    client: genai.Client,
    media_path: str,
    prompt: str,
    schema_cls: type,
    model: str,
    fallback_model: str,
    *,
    original_url: Optional[str] = None,
    rc: RetryConfig = _DEFAULT_RETRY,
) -> Tuple[Any, Optional[str], str]:
    """Upload a media file and call Gemini with *schema_cls* as the response schema.

    Returns ``(parsed_result, error_message, model_used)``.
    """
    contextual_prompt = prompt
    if original_url:
        contextual_prompt = f"{prompt}\n\nOriginal URL: {original_url}\n"

    try:
        mime, _ = mimetypes.guess_type(media_path)
        if not mime:
            ext = os.path.splitext(media_path)[1].lower()
            mime = {
                ".mp4": "video/mp4", ".webm": "video/webm",
                ".mov": "video/quicktime", ".avi": "video/x-msvideo",
                ".mkv": "video/x-matroska",
                ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".png": "image/png", ".gif": "image/gif",
                ".webp": "image/webp", ".bmp": "image/bmp",
                ".svg": "image/svg+xml",
            }.get(ext, "application/octet-stream")
        upload_cfg = {"mime_type": mime} if mime else {}
        media_file = client.files.upload(file=media_path, config=upload_cfg)
        wait = rc.file_active_max_wait_sec
        while media_file.state == genai_types.FileState.PROCESSING:
            if wait <= 0:
                return None, "File did not reach ACTIVE state before timeout", model
            time.sleep(rc.file_poll_interval_sec)
            wait -= rc.file_poll_interval_sec
            media_file = client.files.get(name=media_file.name)
        if media_file.state == genai_types.FileState.FAILED:
            return None, f"File failed to process: {media_file.state}", model

        result, err = _do_generate(
            client, model, [contextual_prompt, media_file], schema_cls, rc
        )
        if result is not None:
            return result, None, model
        if err and is_daily_quota_resource_error(err):
            return None, str(err), model
        if err and is_resource_exhausted_error(err) and fallback_model != model:
            r2, e2 = _do_generate(
                client, fallback_model, [contextual_prompt, media_file], schema_cls, rc
            )
            if r2 is not None:
                return r2, None, fallback_model
            return None, str(e2) if e2 else "Unknown error", fallback_model
        return None, str(err) if err else "Unknown error", model
    except Exception as e:
        return None, str(e), model


def call_gemini_for_url(
    client: genai.Client,
    url: str,
    prompt: str,
    schema_cls: type,
    fallback_model: str,
    *,
    rc: RetryConfig = _DEFAULT_RETRY,
) -> Tuple[Any, Optional[str], str]:
    """Call Gemini with only the URL in the prompt (no file upload).

    Always uses *fallback_model*.
    Returns ``(parsed_result, error_message, model_used)``.
    """
    contextual_prompt = (
        f"{prompt}\n\n"
        f"We could not download the media. Please analyze based on the following URL: {url}\n"
        f"If you cannot access or analyze the content from this URL, return an error."
    )
    result, err = _do_generate(
        client, fallback_model, [contextual_prompt], schema_cls, rc
    )
    if result is not None:
        return result, None, fallback_model
    return None, str(err) if err else "Unknown error", fallback_model


# ═════════════════════════════════════════════════════════════════════════════
# URL dispatch (download → media call, fallback → URL-only call)
# ═════════════════════════════════════════════════════════════════════════════


def safe_str(val: Any) -> str:
    """Coerce a cell value to a clean string (handles NaN, None, 'nan')."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    s = str(val).strip()
    return "" if s.lower() == "nan" else s


def send_url_to_api(
    row: pd.Series,
    client: genai.Client,
    prompt: str,
    schema_cls: type,
    model: str,
    fallback_model: str,
    media_downloads_dir: str,
    *,
    cookies_file: str = "",
    rc: RetryConfig = _DEFAULT_RETRY,
) -> Tuple[Any, str, Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Download media from the row's URLs and run the Gemini VLM call.

    Prefers ``post_url``; falls back to ``media_url``; falls back to URL-only.

    Returns ``(analysis, url_used, error, model_used, download_err, local_media_path)``.
    """
    post_u = normalize_download_url(safe_str(row.get("post_url")))
    media_u = normalize_download_url(safe_str(row.get("media_url")))
    os.makedirs(media_downloads_dir, exist_ok=True)

    for url, role in [(post_u, "post"), (media_u, "media")]:
        if not url:
            continue
        save_name = url_to_media_save_name(url, role)
        media_path, dl_err = download_media(
            url, media_downloads_dir, save_name, cookies_file=cookies_file,
        )
        if media_path:
            analysis, error, m_used = call_gemini_for_media(
                client, media_path, prompt, schema_cls, model, fallback_model,
                original_url=url, rc=rc,
            )
            return analysis, role, error, m_used, None, media_path
        analysis, error, m_used = call_gemini_for_url(
            client, url, prompt, schema_cls, fallback_model, rc=rc,
        )
        return analysis, "url", error, m_used, dl_err, None

    return None, "None", "No media_url or post_url available to download.", None, None, None
