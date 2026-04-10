"""
Generic VLM pipeline orchestration — schema-agnostic.

This module provides the full run lifecycle:

1. :func:`prepare_pipeline` reads configuration from a :class:`ProjectSpec`,
   loads the input CSV, deduplicates, connects to storage, and returns a
   :class:`PipelineContext`.
2. :func:`process_urls_sync` / :func:`process_urls_async` iterate over every
   queued row, calling the first-pass Gemini API and (optionally) the
   annotation pass.
3. :func:`print_run_summary` prints a short completion message.

All schema-specific behaviour is injected via the ``ProjectSpec`` (schemas,
prompts, hooks).  The pipeline auto-derives ``vlm_*`` output columns from
``dataclasses.fields(spec.analysis_schema)``.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import pandas as pd
from google import genai

from .media import setup_api_services_for_credentials_dir, normalize_download_url
from .core import suppress_gemini_sdk_warnings
from .gemini import (
    RetryConfig,
    call_gemini_for_media,
    call_gemini_for_url,
    is_daily_quota_resource_error,
    safe_str,
    send_url_to_api,
)
from .io import (
    append_df_to_csv,
    append_sheet_row,
    compute_content_key,
    df_row_to_sheet_row,
    ensure_worksheet,
    fetch_processed_keys_from_csv,
    fetch_processed_keys_from_sheet,
    output_column_order,
    row_dict_has_annotation_payload,
    sync_sheet_headers,
)
from .project import ProjectSpec

# ═════════════════════════════════════════════════════════════════════════════
# Auto-derived column lists
# ═════════════════════════════════════════════════════════════════════════════


def vlm_result_columns(spec: ProjectSpec) -> list[str]:
    """Derive ``vlm_*`` output columns from the project's analysis schema fields."""
    cols = [
        "vlm_content_key",
        "vlm_url_used",
        "vlm_model_used",
        "vlm_download_error_before_url",
    ]
    for f in dataclasses.fields(spec.analysis_schema):
        cols.append(f"vlm_{f.name}")
    cols += ["vlm_full_response_json", "vlm_error", "vlm_processed_at"]
    if spec.annotation_enabled:
        cols += _annotation_column_names(spec)
    return cols


def _annotation_column_names(spec: ProjectSpec) -> list[str]:
    """Column names added by the annotation pass."""
    if spec.materialize_annotation_fn is not None:
        # Hooks define their own column names — ask the hook via a sentinel call.
        # Convention: the first time the hook sees ``None`` resp, it returns the
        # blank dict whose keys are the column names.
        try:
            blank = spec.materialize_annotation_fn(None, None, None, None, None)
            if isinstance(blank, dict):
                return list(blank.keys())
        except Exception:
            pass
    return ["vlm_annotation_json", "vlm_annotation_error", "vlm_annotation_model_used"]


# ═════════════════════════════════════════════════════════════════════════════
# Row-level helpers
# ═════════════════════════════════════════════════════════════════════════════


def build_row_result_dict(
    row: pd.Series,
    analysis: Any,
    url_used: str,
    error: Optional[str],
    model_used: Optional[str],
    download_err_before_url: Optional[str],
    spec: ProjectSpec,
) -> dict[str, Any]:
    """Merge the original CSV row with first-pass VLM columns.

    Columns are derived from ``spec.analysis_schema`` fields, making this
    function work with any analysis dataclass.
    """
    processed_at = datetime.now(timezone.utc).isoformat()
    row_dict: dict[str, Any] = row.to_dict()

    if analysis is not None:
        try:
            full_json = json.dumps(asdict(analysis), default=str)
        except TypeError:
            d = {}
            for f in dataclasses.fields(analysis):
                v = getattr(analysis, f.name)
                d[f.name] = v.value if isinstance(v, Enum) else v
            full_json = json.dumps(d, default=str)

        row_dict["vlm_url_used"] = url_used
        row_dict["vlm_model_used"] = model_used
        row_dict["vlm_download_error_before_url"] = download_err_before_url
        for f in dataclasses.fields(analysis):
            row_dict[f"vlm_{f.name}"] = getattr(analysis, f.name)
        row_dict["vlm_full_response_json"] = full_json
        row_dict["vlm_error"] = None
        row_dict["vlm_processed_at"] = processed_at
    else:
        api_err: str = error or "Unknown error"
        if download_err_before_url:
            api_err = (
                f"File download failed ({download_err_before_url}). "
                f"URL-only: {api_err}"
            )
        row_dict["vlm_url_used"] = url_used
        row_dict["vlm_model_used"] = model_used
        row_dict["vlm_download_error_before_url"] = download_err_before_url
        for f in dataclasses.fields(spec.analysis_schema):
            row_dict[f"vlm_{f.name}"] = None
        row_dict["vlm_full_response_json"] = None
        row_dict["vlm_error"] = api_err
        row_dict["vlm_processed_at"] = processed_at

    return row_dict


def _check_annotation_field_rules(
    analysis: Any, rules: dict[str, Any]
) -> bool:
    """Generic field-rule gate: every rule must match the analysis object."""
    for field_name, expected in rules.items():
        actual = getattr(analysis, field_name, None)
        if actual is None:
            return False
        if isinstance(actual, Enum):
            if str(actual.value).upper() != str(expected).upper():
                return False
        elif isinstance(expected, bool):
            if bool(actual) != expected:
                return False
        elif actual != expected:
            return False
    return True


def _should_annotate(analysis: Any, spec: ProjectSpec) -> bool:
    """Decide whether the annotation pass should run for this row."""
    if not spec.annotation_enabled:
        return False
    if spec.annotation_criteria_fn is not None:
        return spec.annotation_criteria_fn(analysis)
    if spec.annotation_field_rules:
        return _check_annotation_field_rules(analysis, spec.annotation_field_rules)
    return True


def _default_build_annotation_prompt(
    template: str,
    post_id: str,
    prior: Any,
    row: pd.Series,
) -> str:
    """Fallback annotation-prompt builder using simple ``str.format`` substitutions."""
    try:
        prior_dict = asdict(prior)
    except TypeError:
        prior_dict = {
            f.name: getattr(prior, f.name) for f in dataclasses.fields(prior)
        }
    post_bundle = {"vlm_content_key": post_id, "prior_analysis": prior_dict}
    meta = {k: row.get(k) for k in row.index if not str(k).startswith("vlm_")}
    post_u = normalize_download_url(safe_str(row.get("post_url")))
    media_u = normalize_download_url(safe_str(row.get("media_url")))
    canonical_url = (post_u or media_u or "(none)").strip()

    return template.format(
        post_id_and_prior_json=json.dumps(post_bundle, default=str),
        row_metadata_json=json.dumps(meta, default=str),
        canonical_post_url=canonical_url,
    )


def _blank_annotation_cols(spec: ProjectSpec) -> dict[str, Any]:
    """Empty annotation columns when annotation is skipped."""
    names = _annotation_column_names(spec)
    return {n: None for n in names}


def _run_annotation_pass(
    row_dict: dict[str, Any],
    analysis: Any,
    post_id: str,
    local_media_path: Optional[str],
    client: genai.Client,
    spec: ProjectSpec,
    rc: RetryConfig,
) -> None:
    """Execute the annotation pass and update *row_dict* in place."""
    if not _should_annotate(analysis, spec):
        row_dict.update(_blank_annotation_cols(spec))
        return

    row_series = pd.Series(row_dict)
    assert spec.annotation_schema is not None
    assert spec.annotation_prompt_template is not None or spec.build_annotation_prompt_fn is not None

    if spec.build_annotation_prompt_fn is not None:
        prompt = spec.build_annotation_prompt_fn(post_id, analysis, row_series)
    else:
        assert spec.annotation_prompt_template is not None
        prompt = _default_build_annotation_prompt(
            spec.annotation_prompt_template, post_id, analysis, row_series
        )

    post_u = normalize_download_url(safe_str(row_series.get("post_url")))
    media_u = normalize_download_url(safe_str(row_series.get("media_url")))

    if local_media_path:
        resp, ann_err, ann_model = call_gemini_for_media(
            client, local_media_path, prompt, spec.annotation_schema,
            spec.primary_model, spec.fallback_model,
            original_url=post_u or media_u, rc=rc,
        )
    else:
        url_for_prompt = post_u or media_u
        if not url_for_prompt:
            row_dict.update(_blank_annotation_cols(spec))
            return
        resp, ann_err, ann_model = call_gemini_for_url(
            client, url_for_prompt, prompt, spec.annotation_schema,
            spec.fallback_model, rc=rc,
        )

    if spec.materialize_annotation_fn is not None:
        try:
            ann_cols = spec.materialize_annotation_fn(
                resp, row_series, analysis, post_id, local_media_path
            )
        except Exception as exc:
            ann_cols = _blank_annotation_cols(spec)
            for k in ann_cols:
                if k.endswith("_error"):
                    ann_cols[k] = f"Annotation materialise error: {exc}"
    else:
        if resp is not None:
            try:
                ann_json = json.dumps(asdict(resp), default=str)
            except TypeError:
                ann_json = str(resp)
            ann_cols = {
                "vlm_annotation_json": ann_json,
                "vlm_annotation_error": None,
                "vlm_annotation_model_used": ann_model,
            }
        else:
            ann_cols = {
                "vlm_annotation_json": None,
                "vlm_annotation_error": ann_err,
                "vlm_annotation_model_used": ann_model,
            }

    for k in list(ann_cols):
        if k.endswith("_model_used"):
            ann_cols[k] = ann_model
    if resp is None and ann_err:
        for k in list(ann_cols):
            if k.endswith("_error") and not ann_cols.get(k):
                ann_cols[k] = ann_err

    row_dict.update(ann_cols)


def should_stop_for_daily_quota(
    error: Optional[str],
    model_used: Optional[str],
    primary_model: str,
    stop_flag: bool,
) -> bool:
    """True when the primary model's daily quota is exhausted."""
    return (
        stop_flag
        and error is not None
        and model_used == primary_model
        and is_daily_quota_resource_error(error)
    )


def emit_row_to_callback(
    row_dict: dict[str, Any],
    callback: Optional[Callable[[pd.DataFrame], None]],
    column_order: Optional[list[str]],
) -> None:
    """Wrap *row_dict* in a one-row DataFrame and invoke the persistence callback."""
    if callback is None:
        return
    if column_order is None:
        raise ValueError("column_order required when callback is set")
    one = pd.DataFrame([row_dict]).reindex(columns=column_order)
    callback(one)


# ═════════════════════════════════════════════════════════════════════════════
# Pipeline context
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class PipelineContext:
    """Everything the processing loop needs, returned by :func:`prepare_pipeline`."""

    spec: ProjectSpec
    gemini_client: genai.Client
    batch: pd.DataFrame
    retry_config: RetryConfig
    persist_row: Callable[[pd.DataFrame], None]
    column_order: list[str]
    mirror_path: Optional[Path]


# ═════════════════════════════════════════════════════════════════════════════
# Pipeline setup
# ═════════════════════════════════════════════════════════════════════════════


def prepare_pipeline(spec: ProjectSpec) -> Optional[PipelineContext]:
    """Load config from *spec*, read input CSV, filter to pending rows, set up storage.

    Returns a :class:`PipelineContext` ready for :func:`process_urls_sync` or
    :func:`process_urls_async`, or ``None`` when there are no rows to process.
    """
    suppress_gemini_sdk_warnings()

    rc = RetryConfig(
        max_attempts=spec.retry_max_attempts,
        base_delay_sec=spec.retry_base_delay_sec,
        rate_limit_margin_sec=spec.rate_limit_margin_sec,
        file_active_max_wait_sec=spec.file_active_max_wait_sec,
        file_poll_interval_sec=spec.file_poll_interval_sec,
    )

    # ── API key & Gemini client ──────────────────────────────────────────
    with open(spec.keys_path) as fh:
        gemini_key = json.load(fh)["gemini"]
    gemini_client = genai.Client(api_key=gemini_key)

    # ── Read & validate input CSV ────────────────────────────────────────
    df = pd.read_csv(spec.input_csv)
    missing = [c for c in spec.required_input_columns if c not in df.columns]
    if missing:
        raise SystemExit(
            "Input CSV must include columns "
            + ", ".join(f"'{c}'" for c in spec.required_input_columns)
            + "."
        )

    df["vlm_content_key"] = df.apply(
        lambda r: compute_content_key(r, spec.dedup_key_columns), axis=1
    )

    sort_cols_present = [c for c in spec.sort_columns if c in df.columns]
    sort_asc = spec.sort_ascending[: len(sort_cols_present)]
    if sort_cols_present:
        df_sorted = df.sort_values(by=sort_cols_present, ascending=sort_asc)
    else:
        df_sorted = df

    # ── Storage: Google Sheets or local CSV ──────────────────────────────
    sheets_svc = None
    annotations_worksheet_title = ""
    annotations_header: list[str] = []

    if spec.spreadsheet_id:
        creds_dir = spec.keys_path.parent
        services = setup_api_services_for_credentials_dir(creds_dir)
        sheets_svc = services["sheets"]
        ensure_worksheet(sheets_svc, spec.spreadsheet_id, spec.worksheet)
        processed_keys = fetch_processed_keys_from_sheet(
            sheets_svc, spec.spreadsheet_id, spec.worksheet
        )
        if spec.annotation_enabled:
            annotations_worksheet_title = spec.annotations_worksheet
            ensure_worksheet(
                sheets_svc, spec.spreadsheet_id, annotations_worksheet_title
            )
            if spec.annotation_sheet_headers_fn is not None:
                desired_ann_headers = spec.annotation_sheet_headers_fn()
            else:
                desired_ann_headers = _annotation_column_names(spec)
            annotations_header = sync_sheet_headers(
                sheets_svc,
                spec.spreadsheet_id,
                annotations_worksheet_title,
                desired_ann_headers,
            )
    else:
        processed_keys = fetch_processed_keys_from_csv(spec.local_results_csv)

    # ── Filter to pending rows ───────────────────────────────────────────
    pending = df_sorted[~df_sorted["vlm_content_key"].isin(processed_keys)]
    if spec.rows_per_platform > 0:
        group_col = spec.sort_columns[0] if spec.sort_columns else "platform"
        if group_col in pending.columns:
            batch = (
                pending.groupby(group_col, group_keys=False)
                .head(spec.rows_per_platform)
                .reset_index(drop=True)
            )
        else:
            batch = pending.head(spec.rows_per_platform).reset_index(drop=True)
    else:
        batch = pending.reset_index(drop=True)

    if spec.shuffle_batch:
        batch = batch.sample(frac=1).reset_index(drop=True)

    if batch.empty:
        print("No rows to process (all candidates already recorded).")
        return None

    # ── Summary ──────────────────────────────────────────────────────────
    if spec.rows_per_platform > 0:
        limit_desc = f"up to {spec.rows_per_platform} new row(s) per platform"
    else:
        limit_desc = (
            "until primary model daily quota"
            if spec.stop_on_primary_daily_quota
            else "full pending queue (no daily stop)"
        )
    storage_desc = (
        "Google Sheet" if spec.spreadsheet_id else str(spec.local_results_csv)
    )
    shuffle_note = " [shuffled]" if spec.shuffle_batch else ""
    print(f"Queued {len(batch)} row(s) ({limit_desc}){shuffle_note}; storage={storage_desc}.")

    # ── Column order ─────────────────────────────────────────────────────
    vlm_cols = vlm_result_columns(spec)
    extra = [c for c in vlm_cols if c not in batch.columns]
    all_src = list(batch.columns)
    column_order = output_column_order(all_src, extra)

    if spec.spreadsheet_id:
        assert sheets_svc is not None
        header = sync_sheet_headers(
            sheets_svc, spec.spreadsheet_id, spec.worksheet, column_order
        )
    else:
        header = column_order

    # ── Mirror CSV ───────────────────────────────────────────────────────
    mirror_path = Path(spec.mirror_csv).resolve() if spec.mirror_csv else None

    # ── Annotation JSON keys (for has-payload check) ─────────────────────
    ann_json_keys = [
        k for k in _annotation_column_names(spec)
        if k.endswith("_json") or k.endswith("_error")
    ]

    # ── Persistence callback ─────────────────────────────────────────────
    def persist_row(one_row: pd.DataFrame) -> None:
        if spec.spreadsheet_id:
            assert sheets_svc is not None
            append_sheet_row(
                sheets_svc,
                spec.spreadsheet_id,
                spec.worksheet,
                df_row_to_sheet_row(one_row, 0, header),
                chunk_size=spec.sheets_append_chunk_size,
            )
            if annotations_worksheet_title and row_dict_has_annotation_payload(
                one_row.iloc[0].to_dict(), ann_json_keys
            ):
                if spec.annotation_sheet_row_fn is not None:
                    ann_row_vals = spec.annotation_sheet_row_fn(
                        one_row.iloc[0].to_dict(), annotations_header
                    )
                else:
                    ann_row_vals = df_row_to_sheet_row(
                        one_row, 0, annotations_header
                    )
                append_sheet_row(
                    sheets_svc,
                    spec.spreadsheet_id,
                    annotations_worksheet_title,
                    ann_row_vals,
                    chunk_size=spec.sheets_append_chunk_size,
                )
        else:
            append_df_to_csv(spec.local_results_csv, one_row)
        if mirror_path is not None:
            append_df_to_csv(mirror_path, one_row)

    return PipelineContext(
        spec=spec,
        gemini_client=gemini_client,
        batch=batch,
        retry_config=rc,
        persist_row=persist_row,
        column_order=header,
        mirror_path=mirror_path,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Synchronous processing loop
# ═════════════════════════════════════════════════════════════════════════════


def process_urls_sync(ctx: PipelineContext) -> pd.DataFrame:
    """Process every row in ``ctx.batch`` synchronously.

    Returns a DataFrame of all processed rows (possibly empty if quota
    stopped early).
    """
    spec = ctx.spec
    results: list[dict[str, Any]] = []
    vlm_cols = vlm_result_columns(spec)

    for idx in range(len(ctx.batch)):
        row = ctx.batch.iloc[idx]
        post_id = str(row.get("vlm_content_key", ""))

        analysis, url_used, error, model_used, dl_err, local_path = send_url_to_api(
            row,
            ctx.gemini_client,
            spec.first_pass_prompt,
            spec.analysis_schema,
            spec.primary_model,
            spec.fallback_model,
            spec.media_downloads_dir,
            cookies_dir=spec.cookies_dir,
            rc=ctx.retry_config,
        )

        row_dict = build_row_result_dict(
            row, analysis, url_used, error, model_used, dl_err, spec
        )

        if spec.annotation_enabled:
            _run_annotation_pass(
                row_dict, analysis, post_id, local_path,
                ctx.gemini_client, spec, ctx.retry_config,
            )

        emit_row_to_callback(row_dict, ctx.persist_row, ctx.column_order)
        results.append(row_dict)

        if should_stop_for_daily_quota(
            error, model_used, spec.primary_model,
            spec.stop_on_primary_daily_quota,
        ):
            print(
                f"Primary model ({spec.primary_model}) daily quota reached "
                f"after {idx + 1} row(s). Stopping."
            )
            break

    if not results:
        placeholder = ctx.batch.head(0).copy()
        for c in vlm_cols:
            if c not in placeholder.columns:
                placeholder[c] = pd.Series(dtype=object)
        return placeholder
    return pd.DataFrame(results).reindex(columns=ctx.column_order)


# ═════════════════════════════════════════════════════════════════════════════
# Asynchronous processing loop
# ═════════════════════════════════════════════════════════════════════════════


async def process_urls_async(ctx: PipelineContext) -> pd.DataFrame:
    """Process every row in ``ctx.batch`` with ``asyncio.to_thread`` wrapping.

    The Gemini SDK is synchronous, so each call is off-loaded to a thread.
    Returns a DataFrame of all processed rows.
    """
    spec = ctx.spec
    results: list[dict[str, Any]] = []
    vlm_cols = vlm_result_columns(spec)

    for idx in range(len(ctx.batch)):
        row = ctx.batch.iloc[idx]
        post_id = str(row.get("vlm_content_key", ""))

        analysis, url_used, error, model_used, dl_err, local_path = (
            await asyncio.to_thread(
                send_url_to_api,
                row,
                ctx.gemini_client,
                spec.first_pass_prompt,
                spec.analysis_schema,
                spec.primary_model,
                spec.fallback_model,
                spec.media_downloads_dir,
                cookies_dir=spec.cookies_dir,
                rc=ctx.retry_config,
            )
        )

        row_dict = build_row_result_dict(
            row, analysis, url_used, error, model_used, dl_err, spec
        )

        if spec.annotation_enabled:
            await asyncio.to_thread(
                _run_annotation_pass,
                row_dict, analysis, post_id, local_path,
                ctx.gemini_client, spec, ctx.retry_config,
            )

        await asyncio.to_thread(
            emit_row_to_callback, row_dict, ctx.persist_row, ctx.column_order
        )
        results.append(row_dict)

        if should_stop_for_daily_quota(
            error, model_used, spec.primary_model,
            spec.stop_on_primary_daily_quota,
        ):
            print(
                f"Primary model ({spec.primary_model}) daily quota reached "
                f"after {idx + 1} row(s). Stopping."
            )
            break

    if not results:
        placeholder = ctx.batch.head(0).copy()
        for c in vlm_cols:
            if c not in placeholder.columns:
                placeholder[c] = pd.Series(dtype=object)
        return placeholder
    return pd.DataFrame(results).reindex(columns=ctx.column_order)


# ═════════════════════════════════════════════════════════════════════════════
# Post-run summary
# ═════════════════════════════════════════════════════════════════════════════


def print_run_summary(
    df_out: pd.DataFrame,
    mirror_path: Optional[Path],
) -> None:
    """Print a short completion message."""
    print(f"Finished this run: {len(df_out)} row(s) written.")
    if mirror_path:
        print(f"Also mirrored each row to {mirror_path}")
