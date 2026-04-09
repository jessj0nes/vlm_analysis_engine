"""
Google Sheets and CSV I/O helpers — schema-agnostic.

All persistence operations (read/write Google Sheets, local CSV append, dedup
key fetch) live here.  Nothing in this module references a specific analysis
or annotation schema.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd

from .media import col_index_to_letter, normalize_download_url
from .core import DEFAULT_SHEETS_APPEND_CHUNK_SIZE
from .gemini import safe_url_str

# ═════════════════════════════════════════════════════════════════════════════
# Deduplication key
# ═════════════════════════════════════════════════════════════════════════════


def compute_content_key(row: pd.Series, key_columns: list[str]) -> str:
    """SHA-256 hash over the values of *key_columns* for cross-run dedup."""
    parts = [normalize_download_url(safe_url_str(row.get(col))) for col in key_columns]
    raw = "\x1f".join(parts).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


# ═════════════════════════════════════════════════════════════════════════════
# Cell / row formatting
# ═════════════════════════════════════════════════════════════════════════════


def cell_str(v: Any) -> str:
    """Convert any value to a Sheets-friendly string."""
    if v is None:
        return ""
    if isinstance(v, float) and pd.isna(v):
        return ""
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    return str(v)


def df_row_to_sheet_row(
    df: pd.DataFrame, idx: int, columns: Sequence[str]
) -> list[str]:
    row = df.iloc[idx]
    return [cell_str(row.get(c)) for c in columns]


# ═════════════════════════════════════════════════════════════════════════════
# Column ordering
# ═════════════════════════════════════════════════════════════════════════════


def sheet_column_order_for_df(df: pd.DataFrame) -> list[str]:
    """Source columns first, then sorted ``vlm_*`` columns."""
    vlm_cols = [c for c in df.columns if str(c).startswith("vlm_")]
    src_cols = [c for c in df.columns if c not in vlm_cols]
    return src_cols + sorted(vlm_cols)


def output_column_order(
    source_columns: list[str], vlm_columns: list[str]
) -> list[str]:
    """Stable column order: source columns followed by sorted VLM columns."""
    placeholder = pd.DataFrame(columns=source_columns + vlm_columns)
    return sheet_column_order_for_df(placeholder)


# ═════════════════════════════════════════════════════════════════════════════
# Google Sheets helpers
# ═════════════════════════════════════════════════════════════════════════════


def _a1_tab(title: str) -> str:
    if "'" in title:
        raise ValueError("Worksheet title cannot contain a single quote")
    return f"'{title}'"


def ensure_worksheet(sheets_svc: Any, spreadsheet_id: str, title: str) -> None:
    meta = sheets_svc.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
    for s in meta.get("sheets", []):
        if s["properties"]["title"] == title:
            return
    sheets_svc.spreadsheets().batchUpdate(
        spreadsheetId=spreadsheet_id,
        body={"requests": [{"addSheet": {"properties": {"title": title}}}]},
    ).execute()


def _get_header_row(sheets_svc: Any, spreadsheet_id: str, title: str) -> list[str]:
    rng = f"{_a1_tab(title)}!1:1"
    resp = (
        sheets_svc.spreadsheets()
        .values()
        .get(spreadsheetId=spreadsheet_id, range=rng)
        .execute()
    )
    rows = resp.get("values", [])
    return list(rows[0]) if rows and rows[0] else []


def _set_header_row(
    sheets_svc: Any, spreadsheet_id: str, title: str, headers: list[str]
) -> None:
    rng = f"{_a1_tab(title)}!1:1"
    sheets_svc.spreadsheets().values().update(
        spreadsheetId=spreadsheet_id,
        range=rng,
        valueInputOption="USER_ENTERED",
        body={"values": [headers]},
    ).execute()


def sync_sheet_headers(
    sheets_svc: Any, spreadsheet_id: str, title: str, desired: list[str]
) -> list[str]:
    current = _get_header_row(sheets_svc, spreadsheet_id, title)
    if not current:
        _set_header_row(sheets_svc, spreadsheet_id, title, desired)
        return desired
    missing = [c for c in desired if c not in current]
    if missing:
        new_h = current + missing
        _set_header_row(sheets_svc, spreadsheet_id, title, new_h)
        return new_h
    return current


def fetch_processed_keys_from_sheet(
    sheets_svc: Any, spreadsheet_id: str, title: str
) -> set[str]:
    headers = _get_header_row(sheets_svc, spreadsheet_id, title)
    if not headers or "vlm_content_key" not in headers:
        return set()
    col_idx = headers.index("vlm_content_key")
    col_letter = col_index_to_letter(col_idx)
    rng = f"{_a1_tab(title)}!{col_letter}2:{col_letter}"
    resp = (
        sheets_svc.spreadsheets()
        .values()
        .get(spreadsheetId=spreadsheet_id, range=rng)
        .execute()
    )
    keys: set[str] = set()
    for row in resp.get("values", []):
        if row and row[0]:
            keys.add(str(row[0]).strip())
    return keys


def append_values_to_sheet(
    sheets_svc: Any,
    spreadsheet_id: str,
    title: str,
    values: list[list[str]],
    chunk_size: int = DEFAULT_SHEETS_APPEND_CHUNK_SIZE,
) -> None:
    if not values:
        return
    for start in range(0, len(values), chunk_size):
        chunk = values[start : start + chunk_size]
        sheets_svc.spreadsheets().values().append(
            spreadsheetId=spreadsheet_id,
            range=f"{_a1_tab(title)}!A1",
            valueInputOption="USER_ENTERED",
            insertDataOption="INSERT_ROWS",
            body={"values": chunk},
        ).execute()


def append_df_to_sheet(
    sheets_svc: Any,
    spreadsheet_id: str,
    title: str,
    df: pd.DataFrame,
    columns: list[str],
    chunk_size: int = DEFAULT_SHEETS_APPEND_CHUNK_SIZE,
) -> None:
    if df.empty:
        return
    values = [df_row_to_sheet_row(df, i, columns) for i in range(len(df))]
    append_values_to_sheet(sheets_svc, spreadsheet_id, title, values, chunk_size)


def append_sheet_row(
    sheets_svc: Any,
    spreadsheet_id: str,
    title: str,
    row_values: list[str],
    chunk_size: int = DEFAULT_SHEETS_APPEND_CHUNK_SIZE,
) -> None:
    """Append a single row (crash-safe incremental write)."""
    append_values_to_sheet(sheets_svc, spreadsheet_id, title, [row_values], chunk_size)


# ═════════════════════════════════════════════════════════════════════════════
# Local CSV helpers
# ═════════════════════════════════════════════════════════════════════════════


def fetch_processed_keys_from_csv(path: Path) -> set[str]:
    if not path.exists():
        return set()
    prev = pd.read_csv(path)
    if "vlm_content_key" not in prev.columns:
        return set()
    return {
        str(x).strip()
        for x in prev["vlm_content_key"].dropna().astype(str)
        if str(x).strip()
    }


def append_df_to_csv(path: Path, df: pd.DataFrame) -> None:
    if df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists() or path.stat().st_size == 0
    df.to_csv(path, mode="w" if new_file else "a", header=new_file, index=False)


def row_dict_has_annotation_payload(
    row: dict[str, Any], annotation_json_keys: Sequence[str]
) -> bool:
    """True when the row carries annotation data worth copying to a separate sheet."""
    for key in annotation_json_keys:
        v = row.get(key)
        if v is None:
            continue
        if isinstance(v, float) and pd.isna(v):
            continue
        if str(v).strip():
            return True
    return False
