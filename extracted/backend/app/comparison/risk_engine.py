"""
Real-Time Risk Scoring Engine.

Interpretable, rule-based scorer using literature-informed reference ranges:
natural human speech shows more pitch/amplitude micro-variation than most
neural TTS/vocoder output, so unusually smooth = higher risk. Fully
explainable to judges, works with zero training data. Combined with an
optional ML score and contextual boosts.
"""

# Reasonable starting reference ranges for natural human speech.
# NOT calibrated on ground-truth labeled data yet -- recalibrate once you
# have real ASVspoof / your-own-recorded genuine-vs-cloned samples.
NATURAL_RANGES = {
    "jitter_proxy": (0.015, 0.09),          # too low -> unnaturally smooth pitch
    "shimmer_proxy": (0.12, 0.55),          # too low -> unnaturally smooth amplitude
    "spectral_flatness_mean": (0.0, 0.35),  # too high -> noise-like/synthetic texture
    "pitch_std_hz": (8.0, 60.0),            # too low -> monotone/synthetic
}

WEIGHTS = {
    "jitter_proxy": 0.30,
    "shimmer_proxy": 0.30,
    "spectral_flatness_mean": 0.20,
    "pitch_std_hz": 0.20,
}


def _range_penalty(value: float, low: float, high: float) -> float:
    """Returns 0..1, higher = more suspicious (further below the natural range)."""
    if value >= low:
        return 0.0
    span = max(high - low, 1e-6)
    return min(1.0, (low - value) / span)


def heuristic_score(features: dict) -> tuple[float, list]:
    reasons = []
    total = 0.0
    for key, (low, high) in NATURAL_RANGES.items():
        val = features.get(key, 0.0)
        penalty = _range_penalty(val, low, high)
        total += penalty * WEIGHTS[key]
        if penalty > 0.4:
            reasons.append(
                f"{key}={val:.4f} is below the natural human range "
                f"[{low}-{high}], suggesting artificial smoothness"
            )
    score_0_100 = round(total * 100, 1)
    if not reasons:
        reasons.append("All measured acoustic/prosodic features fall within natural human ranges")
    return score_0_100, reasons


def context_boost(context: dict) -> float:
    boost = 0.0
    if context.get("unknown_caller"):
        boost += 8.0
    if context.get("high_value_transaction"):
        boost += 10.0
    if context.get("privileged_request"):
        boost += 7.0
    return boost


def level_for_score(score: float) -> str:
    if score < 40:
        return "LOW"
    if score < 65:
        return "MEDIUM"
    return "HIGH"


def compute_risk(features: dict, ml_score: float | None = None, context: dict | None = None) -> dict:
    context = context or {}
    h_score, reasons = heuristic_score(features)

    if ml_score is not None:
        # Blend heuristic (explainable) and ML (learned) signal.
        blended = 0.5 * h_score + 0.5 * (ml_score * 100)
    else:
        blended = h_score

    boost = context_boost(context)
    final = min(100.0, blended + boost)
    if boost > 0:
        reasons.append(f"Context risk boost applied (+{boost:.1f}) based on call-context flags")

    return {
        "heuristic_score": h_score,
        "ml_score": round(ml_score * 100, 1) if ml_score is not None else None,
        "context_boost": boost,
        "final_score": round(final, 1),
        "level": level_for_score(final),
        "reasons": reasons,
    }
