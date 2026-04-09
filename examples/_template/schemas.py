"""
Example analysis schema for a VLM pipeline project.

Define one ``@dataclass`` that describes the structured output you want from
the Gemini VLM call.  The field names become ``vlm_<field_name>`` columns in
the output CSV / Google Sheet.

This example checks whether media appears AI-generated and provides a
justification — replace or extend for your own research question.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Confidence(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    NONE = "NONE"


@dataclass
class MyAnalysis:
    """Minimal first-pass analysis schema (customise for your project)."""

    ai_generated: Confidence = field(
        metadata={"description": "Confidence that the media is AI-generated."}
    )
    justification: Optional[str] = field(
        metadata={"description": "Brief reasoning for the AI-generation assessment."}
    )
