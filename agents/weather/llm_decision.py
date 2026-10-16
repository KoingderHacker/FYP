from __future__ import annotations

import json
import os
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel
from google import genai
from google.genai import types

load_dotenv()


class LLMDecision(BaseModel):
    column: str
    issue_type: Literal["sentinel_value", "domain_violation", "statistical_outlier"]
    action: Literal["replace_and_interpolate", "retain", "investigate"]
    reason: str
    confidence: Literal["HIGH", "MEDIUM", "LOW"]


class LLMDecisionResult(BaseModel):
    decisions: list[LLMDecision]


SYSTEM_INSTRUCTION = """
You are a Weather Data Quality Decision Agent.
Analyze diagnostics and recommend actions. Do not modify data.
Sentinel values represent unavailable source data. Statistical outliers are not automatically errors.
Physically impossible values should be investigated. Never delete rows. Long unresolved gaps should be investigated,
not fabricated through extrapolation. Return only the requested structured decision object.
"""


def run_llm_decision(diagnostics: dict) -> dict:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured")
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=[SYSTEM_INSTRUCTION, json.dumps(diagnostics, default=str)],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=LLMDecisionResult,
        ),
    )
    parsed = response.parsed
    if parsed is None:
        raise ValueError("Gemini returned no structured decision")
    return parsed.model_dump()
