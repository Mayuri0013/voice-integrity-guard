"""
alerts.py
---------
"Alerting and User Interaction Layer" + "Configurable workflows" bullets
from the problem statement. Keeps an in-memory log of alerts per session
and exposes notification stubs (wire these to real SMS/email/Slack APIs
in production -- e.g. Twilio, AWS SES).
"""

from __future__ import annotations
import time
from collections import defaultdict

# in-memory alert log: session_id -> list[alert dict]
_alert_log: dict[str, list[dict]] = defaultdict(list)


def raise_alert(session_id: str, risk_result: dict) -> dict:
    alert = {
        "timestamp": time.time(),
        "session_id": session_id,
        "risk_score": risk_result["risk_score"],
        "risk_level": risk_result["risk_level"],
        "recommendation": risk_result["recommendation"],
    }
    if risk_result["risk_level"] in ("HIGH", "MEDIUM"):
        _alert_log[session_id].append(alert)
        _dispatch_notification(alert)
    return alert


def _dispatch_notification(alert: dict):
    """Notification stub. In production, plug in:
      - SMS/email: Twilio, AWS SNS/SES
      - In-app: WebSocket push (already used for the live dashboard)
      - Enterprise: Slack/Teams webhook, SIEM/ticketing system
    For this demo we just print + keep in the in-memory log, which the
    dashboard polls/streams.
    """
    print(f"[ALERT][{alert['risk_level']}] session={alert['session_id']} "
          f"score={alert['risk_score']} -> {alert['recommendation']}")


def get_alerts(session_id: str) -> list[dict]:
    return _alert_log.get(session_id, [])


def clear_alerts(session_id: str):
    _alert_log.pop(session_id, None)
