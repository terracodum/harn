"""Exceptions shared by the pipeline stages.

Rule of the project (see README, "Никакой тихой деградации"): any failure of an LLM step must
surface as one of these exceptions (or as `LLMError`) and end up in result.json. Heuristic
paths are modes the user selects explicitly, never fallbacks taken inside an `except`.
"""
from __future__ import annotations


class PipelineError(RuntimeError):
    """A stage cannot produce its artefact; the run stops with status=failed."""
