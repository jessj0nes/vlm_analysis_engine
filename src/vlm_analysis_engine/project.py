"""
Project specification and TOML-based loader.

A *project* is a directory containing:

* ``project.toml``    — operational settings and pointers to schemas / prompts.
* ``schemas.py``      — one or two ``@dataclass`` types (analysis + optional annotation).
* ``prompts/*.txt``   — prompt text files referenced by the TOML.
* ``hooks.py``        — (optional) custom Python callables the pipeline invokes.

:func:`load_project` reads the TOML, dynamically imports the schema module,
loads prompt files, and (if present) imports hook functions, returning a
fully-resolved :class:`ProjectSpec`.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Optional

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

from .core import (
    DEFAULT_FALLBACK_MODEL,
    DEFAULT_FILE_ACTIVE_MAX_WAIT_SEC,
    DEFAULT_FILE_POLL_INTERVAL_SEC,
    DEFAULT_PRIMARY_MODEL,
    DEFAULT_RATE_LIMIT_MARGIN_SEC,
    DEFAULT_RETRY_BASE_DELAY_SEC,
    DEFAULT_RETRY_MAX_ATTEMPTS,
    DEFAULT_SHEETS_APPEND_CHUNK_SIZE,
)


@dataclass
class ProjectSpec:
    """Everything the generic pipeline needs to execute a project."""

    name: str
    project_dir: Path

    # ── First pass (required) ────────────────────────────────────────────
    analysis_schema: type
    first_pass_prompt: str

    # ── Models ───────────────────────────────────────────────────────────
    primary_model: str = DEFAULT_PRIMARY_MODEL
    fallback_model: str = DEFAULT_FALLBACK_MODEL

    # ── Retry / rate-limit ───────────────────────────────────────────────
    retry_max_attempts: int = DEFAULT_RETRY_MAX_ATTEMPTS
    retry_base_delay_sec: int = DEFAULT_RETRY_BASE_DELAY_SEC
    rate_limit_margin_sec: int = DEFAULT_RATE_LIMIT_MARGIN_SEC
    file_active_max_wait_sec: int = DEFAULT_FILE_ACTIVE_MAX_WAIT_SEC
    file_poll_interval_sec: int = DEFAULT_FILE_POLL_INTERVAL_SEC

    # ── Paths ────────────────────────────────────────────────────────────
    keys_path: Path = field(default_factory=lambda: Path("keys.json"))
    input_csv: Path = field(default_factory=lambda: Path("data/input.csv"))
    local_results_csv: Path = field(default_factory=lambda: Path("data/results.csv"))
    mirror_csv: str = ""
    media_downloads_dir: str = "media_downloads"
    cookies_dir: str = ""

    # ── Google Sheets ────────────────────────────────────────────────────
    spreadsheet_id: str = ""
    worksheet: str = "Results"
    annotations_worksheet: str = "Annotations"
    sheets_append_chunk_size: int = DEFAULT_SHEETS_APPEND_CHUNK_SIZE

    # ── Processing ───────────────────────────────────────────────────────
    rows_per_platform: int = 0
    stop_on_primary_daily_quota: bool = True
    shuffle_batch: bool = False
    required_input_columns: list[str] = field(default_factory=lambda: ["platform"])
    sort_columns: list[str] = field(default_factory=lambda: ["platform"])
    sort_ascending: list[bool] = field(default_factory=lambda: [True])
    dedup_key_columns: list[str] = field(
        default_factory=lambda: ["post_url", "media_url", "platform"]
    )

    # ── Annotation pass (optional — None = skip) ────────────────────────
    annotation_schema: Optional[type] = None
    annotation_prompt_template: Optional[str] = None
    annotation_field_rules: Optional[dict[str, Any]] = None

    # ── Hooks (optional) ─────────────────────────────────────────────────
    annotation_criteria_fn: Optional[Callable] = None
    materialize_annotation_fn: Optional[Callable] = None
    build_annotation_prompt_fn: Optional[Callable] = None
    annotation_sheet_headers_fn: Optional[Callable] = None
    annotation_sheet_row_fn: Optional[Callable] = None

    @property
    def annotation_enabled(self) -> bool:
        return self.annotation_schema is not None


# ═════════════════════════════════════════════════════════════════════════════
# Dynamic module import
# ═════════════════════════════════════════════════════════════════════════════


def _import_module_from_path(filepath: Path, module_name: str) -> ModuleType:
    """Import a Python file by absolute path, registering it as *module_name*."""
    spec = importlib.util.spec_from_file_location(module_name, str(filepath))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {filepath}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# ═════════════════════════════════════════════════════════════════════════════
# TOML loader
# ═════════════════════════════════════════════════════════════════════════════


def load_project(toml_path: str | Path) -> ProjectSpec:
    """Read a ``project.toml`` and return a fully-resolved :class:`ProjectSpec`.

    Path values in the TOML are resolved relative to the TOML file's parent
    directory.  The schema module is imported dynamically, and prompt files are
    read into strings.
    """
    toml_path = Path(toml_path).resolve()
    project_dir = toml_path.parent

    with open(toml_path, "rb") as fh:
        cfg = tomllib.load(fh)

    # ── Project identity ─────────────────────────────────────────────────
    name = cfg.get("project", {}).get("name", project_dir.name)

    # ── Schemas ──────────────────────────────────────────────────────────
    schemas_cfg = cfg.get("schemas", {})
    schema_module_rel = schemas_cfg.get("module", "schemas.py")
    schema_file = (project_dir / schema_module_rel).resolve()
    schema_mod = _import_module_from_path(
        schema_file, f"vlm_project_schemas_{project_dir.name}"
    )

    analysis_class_name = schemas_cfg["analysis_class"]
    analysis_schema = getattr(schema_mod, analysis_class_name)

    annotation_class_name = schemas_cfg.get("annotation_class")
    annotation_schema = (
        getattr(schema_mod, annotation_class_name) if annotation_class_name else None
    )

    # ── Prompts ──────────────────────────────────────────────────────────
    prompts_cfg = cfg.get("prompts", {})
    first_pass_file = (project_dir / prompts_cfg["first_pass"]).resolve()
    first_pass_prompt = first_pass_file.read_text(encoding="utf-8")

    annotation_template_rel = prompts_cfg.get("annotation_template")
    annotation_prompt_template: Optional[str] = None
    if annotation_template_rel:
        annotation_prompt_template = (
            (project_dir / annotation_template_rel).resolve().read_text(encoding="utf-8")
        )

    # ── Hooks ────────────────────────────────────────────────────────────
    hooks_cfg = cfg.get("hooks", {})
    hooks_module_rel = hooks_cfg.get("module")
    annotation_criteria_fn = None
    materialize_annotation_fn = None
    build_annotation_prompt_fn = None
    annotation_sheet_headers_fn = None
    annotation_sheet_row_fn = None

    if hooks_module_rel:
        hooks_file = (project_dir / hooks_module_rel).resolve()
        hooks_mod = _import_module_from_path(
            hooks_file, f"vlm_project_hooks_{project_dir.name}"
        )
        annotation_criteria_fn = getattr(hooks_mod, "annotation_criteria", None)
        materialize_annotation_fn = getattr(hooks_mod, "materialize_annotation", None)
        build_annotation_prompt_fn = getattr(hooks_mod, "build_annotation_prompt", None)
        annotation_sheet_headers_fn = getattr(
            hooks_mod, "annotation_sheet_headers", None
        )
        annotation_sheet_row_fn = getattr(hooks_mod, "annotation_sheet_row", None)

    # ── Resolve paths relative to project dir ────────────────────────────
    def _resolve(rel: str) -> Path:
        return (project_dir / rel).resolve()

    paths_cfg = cfg.get("paths", {})
    models_cfg = cfg.get("models", {})
    retry_cfg = cfg.get("retry", {})
    sheets_cfg = cfg.get("sheets", {})
    proc_cfg = cfg.get("processing", {})
    dedup_cfg = cfg.get("dedup", {})
    ann_cfg = cfg.get("annotation", {})

    return ProjectSpec(
        name=name,
        project_dir=project_dir,
        # Schema & prompts
        analysis_schema=analysis_schema,
        first_pass_prompt=first_pass_prompt,
        annotation_schema=annotation_schema,
        annotation_prompt_template=annotation_prompt_template,
        # Models
        primary_model=models_cfg.get("primary", DEFAULT_PRIMARY_MODEL),
        fallback_model=models_cfg.get("fallback", DEFAULT_FALLBACK_MODEL),
        # Retry
        retry_max_attempts=retry_cfg.get("max_attempts", DEFAULT_RETRY_MAX_ATTEMPTS),
        retry_base_delay_sec=retry_cfg.get(
            "base_delay_sec", DEFAULT_RETRY_BASE_DELAY_SEC
        ),
        rate_limit_margin_sec=retry_cfg.get(
            "rate_limit_margin_sec", DEFAULT_RATE_LIMIT_MARGIN_SEC
        ),
        file_active_max_wait_sec=retry_cfg.get(
            "file_active_max_wait_sec", DEFAULT_FILE_ACTIVE_MAX_WAIT_SEC
        ),
        file_poll_interval_sec=retry_cfg.get(
            "file_poll_interval_sec", DEFAULT_FILE_POLL_INTERVAL_SEC
        ),
        # Paths
        keys_path=_resolve(paths_cfg.get("keys", "keys.json")),
        input_csv=_resolve(paths_cfg.get("input_csv", "data/input.csv")),
        local_results_csv=_resolve(
            paths_cfg.get("local_results_csv", "data/results.csv")
        ),
        mirror_csv=paths_cfg.get("mirror_csv", ""),
        media_downloads_dir=str(
            _resolve(paths_cfg.get("media_downloads", "media_downloads"))
        ),
        cookies_dir=(
            str(_resolve(paths_cfg["cookies_dir"]))
            if paths_cfg.get("cookies_dir")
            else ""
        ),
        # Sheets
        spreadsheet_id=sheets_cfg.get("spreadsheet_id", ""),
        worksheet=sheets_cfg.get("worksheet", "Results"),
        annotations_worksheet=sheets_cfg.get(
            "annotations_worksheet", "Annotations"
        ),
        sheets_append_chunk_size=sheets_cfg.get(
            "append_chunk_size", DEFAULT_SHEETS_APPEND_CHUNK_SIZE
        ),
        # Processing
        rows_per_platform=proc_cfg.get("rows_per_platform", 0),
        stop_on_primary_daily_quota=proc_cfg.get(
            "stop_on_primary_daily_quota", True
        ),
        shuffle_batch=proc_cfg.get("shuffle_batch", False),
        required_input_columns=proc_cfg.get(
            "required_input_columns", ["platform"]
        ),
        sort_columns=proc_cfg.get("sort_columns", ["platform"]),
        sort_ascending=proc_cfg.get("sort_ascending", [True]),
        dedup_key_columns=dedup_cfg.get(
            "key_columns", ["post_url", "media_url", "platform"]
        ),
        # Annotation
        annotation_field_rules=ann_cfg.get("field_rules"),
        # Hooks
        annotation_criteria_fn=annotation_criteria_fn,
        materialize_annotation_fn=materialize_annotation_fn,
        build_annotation_prompt_fn=build_annotation_prompt_fn,
        annotation_sheet_headers_fn=annotation_sheet_headers_fn,
        annotation_sheet_row_fn=annotation_sheet_row_fn,
    )
