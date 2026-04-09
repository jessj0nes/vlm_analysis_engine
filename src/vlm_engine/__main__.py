"""
CLI entry point for the VLM pipeline engine.

Usage::

    vlm-run path/to/project.toml              # synchronous (default)
    vlm-run path/to/project.toml --async       # asynchronous

    # or via module invocation:
    python -m vlm_engine path/to/project.toml --async
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a VLM analysis pipeline from a project.toml configuration."
    )
    parser.add_argument(
        "config",
        help="Path to the project.toml file defining this analysis project.",
    )
    parser.add_argument(
        "--async",
        dest="use_async",
        action="store_true",
        help="Run the processing loop asynchronously (asyncio.to_thread).",
    )
    args = parser.parse_args()

    from .project import load_project
    from .pipeline import prepare_pipeline, print_run_summary

    spec = load_project(args.config)
    ctx = prepare_pipeline(spec)
    if ctx is None:
        return

    if args.use_async:
        import asyncio
        from .pipeline import process_urls_async

        df = asyncio.run(process_urls_async(ctx))
    else:
        from .pipeline import process_urls_sync

        df = process_urls_sync(ctx)

    print_run_summary(df, ctx.mirror_path)


if __name__ == "__main__":
    main()
