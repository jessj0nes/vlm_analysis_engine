# vlm-analysis-engine

Schema-agnostic VLM (Vision-Language Model) analysis pipeline engine powered by Google Gemini.

## Overview

`vlm-analysis-engine` is a generic pipeline for running structured media analysis at scale using the Google Gemini API. You define a **project** — a `@dataclass` schema for what you want Gemini to return, a prompt, and a TOML config — and the engine handles everything else: media download, API calls with retry/fallback, deduplication, and persistence to Google Sheets or local CSV.

## Quick start

### 1. Install

```bash
pip install git+https://github.com/jessj0nes/vlm_analysis_engine.git
```

### 2. Create a project

Copy the template and customise:

```bash
cp -r examples/_template my_project
```

Edit `my_project/schemas.py` with your analysis dataclass, write your prompt in `my_project/prompts/first_pass.txt`, and configure paths/models in `my_project/project.toml`.

### 3. Run

```bash
vlm-run my_project/project.toml              # synchronous
vlm-run my_project/project.toml --async      # asynchronous
```

## Project structure

A project directory contains:

```
my_project/
  project.toml          # operational settings + pointers to schema/prompt
  schemas.py            # @dataclass(es) for Gemini structured output
  prompts/
    first_pass.txt      # the VLM prompt
    annotation.txt      # (optional) second-pass annotation prompt
  hooks.py              # (optional) custom annotation logic
```

The engine reads the TOML, dynamically imports the schema, loads the prompt, and processes every row in your input CSV through the Gemini VLM API.

## Python API

```python
from vlm_analysis_engine import load_project, prepare_pipeline, process_urls_sync, print_run_summary

spec = load_project("my_project/project.toml")
ctx = prepare_pipeline(spec)
if ctx is not None:
    df = process_urls_sync(ctx)
    print_run_summary(df, ctx.mirror_path)
```

## Development

```bash
git clone https://github.com/jessj0nes/vlm_analysis_engine.git
cd vlm_analysis_engine
pip install -e ".[dev]"
```
