"""
main.py
-------
FastAPI application exposing the "Platform and Integration APIs" from
the problem statement:

  POST /api/analyze          -> one-shot analysis of an uploaded audio file
  WS   /ws/stream/{session}   -> near-real-time streaming analysis
                                  (client sends audio chunks, server pushes
                                  back live risk scores as they compute)
  GET  /api/alerts/{session}  -> retrieve alert history for a session
  GET  /api/health            -> health check

Run with:  uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations
import io
import json
import time
import uuid

from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import numpy as np

from .feature_extraction import extract_features, load_audio_from_bytes, SAMPLE_RATE
from .risk_engine import compute_risk
from .alerts import raise_alert, get_alerts
from .privacy import log_feature_event, get_session_history

app = FastAPI(title="Voice Integrity Guard", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok", "time": time.time()}


@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    unknown_caller: bool = Form(False),
    high_value_transaction: bool = Form(False),
    privileged_request: bool = Form(False),
):
    """One-shot analysis: upload a short audio clip, get back a risk score.
    Raw audio bytes exist only for the duration of this request (see
    privacy.py docstring) and are discarded immediately after feature
    extraction -- never written to disk."""
    session_id = str(uuid.uuid4())
    audio_bytes = await file.read()

    try:
        y = load_audio_from_bytes(audio_bytes)
    except Exception as e:
        return {"error": f"Could not decode audio: {e}"}
    finally:
        del audio_bytes  # explicitly drop raw bytes reference

    features = extract_features(y, SAMPLE_RATE)
    context = {
        "unknown_caller": unknown_caller,
        "high_value_transaction": high_value_transaction,
        "privileged_request": privileged_request,
    }
    risk_result = compute_risk(features, context, raw_audio=y, sr=SAMPLE_RATE)

    log_feature_event(session_id, features, risk_result)
    alert = raise_alert(session_id, risk_result)

    return {
        "session_id": session_id,
        "risk_result": risk_result,
        "features_extracted": len(features),
        "alert_raised": alert if risk_result["risk_level"] != "LOW" else None,
    }


@app.get("/api/alerts/{session_id}")
def alerts(session_id: str):
    return {"session_id": session_id, "alerts": get_alerts(session_id)}

 
@app.get("/api/history/{session_id}")
def history(session_id: str):
    return {"session_id": session_id, "history": get_session_history(session_id)}


# ---------------- Real-time streaming endpoint ----------------

CHUNK_WINDOW_SECONDS = 2.0  # analyze every ~2 seconds of audio, like a live call


@app.websocket("/ws/stream/{session_id}")
async def stream_endpoint(websocket: WebSocket, session_id: str):
    """Client streams raw PCM float32 mono audio chunks (binary frames,
    16kHz) OR JSON control messages. Server buffers ~2s windows, runs the
    detection pipeline on each window, and pushes back a live risk update.

    This simulates what a telephony/VoIP integration would do: tap the
    live audio stream, analyze in near-real-time, and surface a running
    risk score to the frontline agent/user before any sensitive action
    is approved -- matching the 'Real-Time Risk Scoring Engine' bullet.
    """
    await websocket.accept()
    buffer = np.array([], dtype=np.float32)
    context = {}
    running_scores = []

    try:
        while True:
            message = await websocket.receive()

            if "text" in message and message["text"] is not None:
                try:
                    payload = json.loads(message["text"])
                except json.JSONDecodeError:
                    continue
                if payload.get("type") == "context":
                    context = payload.get("context", {})
                elif payload.get("type") == "end":
                    break
                continue

            if "bytes" in message and message["bytes"] is not None:
                chunk = np.frombuffer(message["bytes"], dtype=np.float32)
                buffer = np.concatenate([buffer, chunk])

                window_len = int(CHUNK_WINDOW_SECONDS * SAMPLE_RATE)
                if len(buffer) >= window_len:
                    window = buffer[:window_len]
                    buffer = buffer[window_len:]  # slide forward (non-overlapping for simplicity)

                    features = extract_features(window, SAMPLE_RATE)
                    risk_result = compute_risk(features, context, raw_audio=window, sr=SAMPLE_RATE)
                    running_scores.append(risk_result["risk_score"])

                    log_feature_event(session_id, features, risk_result)
                    alert = None
                    if risk_result["risk_level"] != "LOW":
                        alert = raise_alert(session_id, risk_result)

                    await websocket.send_json({
                        "type": "risk_update",
                        "timestamp": time.time(),
                        "risk_result": risk_result,
                        "running_avg_score": round(float(np.mean(running_scores)), 1),
                        "alert": alert,
                    })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


# ---------------- Serve the dashboard frontend ----------------
import os
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "frontend")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/")
    def index():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
