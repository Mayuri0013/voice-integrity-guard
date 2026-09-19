"""
privacy.py
----------
"Privacy and Compliance Module" from the problem statement:
  - Raw audio is NEVER persisted to disk or long-term storage. It is
    decoded in memory, features are extracted, and the raw waveform is
    discarded immediately after feature extraction.
  - Only derived, non-reversible numeric features + risk scores are
    logged (feature-only logging), which cannot be played back as audio.
  - Session data auto-expires (simple TTL sweep) to support data
    minimization requirements (e.g. India's DPDP Act 2023, GDPR-style
    retention limits).
"""

from __future__ import annotations
import time
from collections import defaultdict

SESSION_TTL_SECONDS = 60 * 30  # 30 minutes

# session_id -> {"created": ts, "history": [ {ts, features, risk_result} ] }
_sessions: dict[str, dict] = defaultdict(lambda: {"created": time.time(), "history": []})


def log_feature_event(session_id: str, features: dict, risk_result: dict):
    """Stores ONLY derived features + risk score -- never raw audio bytes."""
    _sweep_expired()
    sess = _sessions[session_id]
    sess["history"].append({
        "timestamp": time.time(),
        "features": features,      # numeric, non-reversible summary stats
        "risk_result": risk_result,
    })
    # cap history length to avoid unbounded memory growth
    if len(sess["history"]) > 500:
        sess["history"] = sess["history"][-500:]


def get_session_history(session_id: str) -> list[dict]:
    return _sessions.get(session_id, {}).get("history", [])


def delete_session(session_id: str):
    _sessions.pop(session_id, None)


def _sweep_expired():
    now = time.time()
    expired = [sid for sid, s in _sessions.items() if now - s["created"] > SESSION_TTL_SECONDS]
    for sid in expired:
        _sessions.pop(sid, None)
