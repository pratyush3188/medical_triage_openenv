"""
Scalar / OpenEnv: each task score must satisfy 0 < score < 1 (reject exactly 0.0 and 1.0).

We centralize normalization so graders, inference, and the HTTP server cannot drift.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

# Wider than 1e-4 so float32 JSON round-trips and Python rounding cannot snap to 0.0 / 1.0
_EPS = 1e-3

STRICT_SCORE_MIN = _EPS
STRICT_SCORE_MAX = 1.0 - _EPS


def strict_open_unit_score(value: object) -> float:
    """Map any numeric-ish input to a float strictly inside (0, 1)."""
    try:
        v = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        v = 0.5
    if not math.isfinite(v):
        v = 0.5

    lo, hi = STRICT_SCORE_MIN, STRICT_SCORE_MAX
    v = min(hi, max(lo, v))
    # `round(x, 4)` can produce exactly 1.0 or 0.0 (e.g. round(0.99995, 4) == 1.0).
    v = round(v, 6)
    v = min(hi, max(lo, v))
    # Last-resort guard: binary float edge cases must not surface as 0.0 or 1.0
    if v <= 0.0:
        v = lo
    elif v >= 1.0:
        v = hi
    return float(v)


def normalize_task_scores(scores: Mapping[str, Any]) -> dict[str, float]:
    """Apply strict_open_unit_score to every value (e.g. OPENENV_TASK_SCORES_JSON payloads)."""
    return {str(k): strict_open_unit_score(v) for k, v in scores.items()}
