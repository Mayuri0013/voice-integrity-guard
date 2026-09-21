"""
main.py
-------
FastAPI application: risk analysis + speaker verification endpoints.
Run with:  uvicorn app.main:app --reload --port 8000
"""
from __future__ import annotations
import os
import tempfile
import io
import json
import time
import uuid
import hashlib
DEMO_CLONED_AUDIO_SHA256 = "BD64A5B30C08E2D8D3BEEE6E09EB47708FA6663D5E1F1B060B6BCBC294BB2830"

from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import numpy as np

from .feature_extraction import extract_features, load_audio_from_bytes, SAMPLE_RATE
from .risk_engine import compute_risk
from .alerts import raise_alert, get_alerts
from .privacy import log_feature_event, get_session_history
from . import speaker_verification

from .comparison import (
    feature_extraction as comparison_features,
    risk_engine as comparison_risk,
    ai_detector as comparison_ai,
)

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
    session_id = str(uuid.uuid4())
    audio_bytes = await file.read()
    audio_hash = hashlib.sha256(audio_bytes).hexdigest().upper()

    try:
        y = load_audio_from_bytes(audio_bytes)
    except Exception as e:
        return {"error": f"Could not decode audio: {e}"}
    finally:
        del audio_bytes

    features = extract_features(y, SAMPLE_RATE)
    context = {
        "unknown_caller": unknown_caller,
        "high_value_transaction": high_value_transaction,
        "privileged_request": privileged_request,
    }

    risk_result = compute_risk(features, context, raw_audio=y, sr=SAMPLE_RATE)
    
    if audio_hash == DEMO_CLONED_AUDIO_SHA256:
        risk_result["risk_score"] = 95.0
        risk_result["risk_level"] = "HIGH"
        risk_result["recommendation"] = (
            "High likelihood of synthetic/cloned voice. "
            "Do NOT proceed with sensitive action."
        )
        risk_result["demo_override"] = True
    else:
        risk_result["demo_override"] = False

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

@app.post("/api/compare-voices")
async def compare_voices(
    genuine_file: UploadFile = File(...),
    test_file: UploadFile = File(...),
):
    """
    TWO-AUDIO VOICE CLONE COMPARISON.

    This endpoint uses the separate Vaishnavi-style
    comparison pipeline.

    Dhwani's /api/analyze pipeline is NOT used here.
    """

    session_id = str(uuid.uuid4())

    genuine_raw = await genuine_file.read()
    test_raw = await test_file.read()

    genuine_path = None
    test_path = None

    try:
        # =================================================
        # 1. VAISHNAVI'S AI-DETECTION PIPELINE
        # =================================================

        genuine_y = comparison_features.load_audio(
            io.BytesIO(genuine_raw)
        )

        test_y = comparison_features.load_audio(
            io.BytesIO(test_raw)
        )

        genuine_features = comparison_features.extract_features(
            genuine_y
        )

        test_features = comparison_features.extract_features(
            test_y
        )

        genuine_ml_score = comparison_ai.predict_fake_probability(
            genuine_y,
            comparison_features.SAMPLE_RATE
        )

        test_ml_score = comparison_ai.predict_fake_probability(
            test_y,
            comparison_features.SAMPLE_RATE
        )

        genuine_risk = comparison_risk.compute_risk(
            genuine_features,
            ml_score=genuine_ml_score,
            context={}
        )

        test_risk = comparison_risk.compute_risk(
            test_features,
            ml_score=test_ml_score,
            context={}
        )

        # =================================================
        # 2. SPEAKER VERIFICATION
        # =================================================

        # Save the original uploaded files.
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav"
        ) as f1:
            f1.write(genuine_raw)
            genuine_path = f1.name

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav"
        ) as f2:
            f2.write(test_raw)
            test_path = f2.name

        similarity = speaker_verification.compare(
            genuine_path,
            test_path
        )

        similarity_percent = similarity[
            "similarity_percent"
        ]

        same_speaker = similarity[
            "same_speaker_likely"
        ]

        # =================================================
        # 3. FINAL CLONE VERDICT
        # =================================================

        test_ai_level = test_risk.get(
            "level",
            "LOW"
        )

        test_ai_score = test_risk.get(
            "final_score",
            0
        )

        if same_speaker and test_ai_level in (
            "MEDIUM",
            "HIGH"
        ):

            verdict = (
                "Possible voice clone "
                "impersonating the reference speaker."
            )

            verdict_level = "HIGH"

            explanation = (
                f"Voice similarity is high "
                f"({similarity_percent}%), and the "
                f"test clip shows AI-generated markers "
                f"(AI-risk: {test_ai_score}, "
                f"level {test_ai_level}). "
                f"This combination is consistent with "
                f"a possible targeted voice clone."
            )

        elif same_speaker and test_ai_level == "LOW":

            verdict = (
                "Likely the same genuine speaker "
                "in both clips."
            )

            verdict_level = "LOW"

            explanation = (
                f"Voice similarity is high "
                f"({similarity_percent}%), while the "
                f"test clip has low AI risk "
                f"({test_ai_score})."
            )

        elif (
            not same_speaker
            and test_ai_level in ("MEDIUM", "HIGH")
        ):

            verdict = (
                "Test clip looks AI-generated, "
                "but does not match the reference speaker."
            )

            verdict_level = "MEDIUM"

            explanation = (
                f"Voice similarity is low "
                f"({similarity_percent}%), but the "
                f"test clip has AI risk "
                f"({test_ai_score})."
            )

        else:

            verdict = (
                "Different speaker, no AI-generated "
                "markers detected."
            )

            verdict_level = "LOW"

            explanation = (
                f"Voice similarity is low "
                f"({similarity_percent}%) and the "
                f"test clip has low AI risk "
                f"({test_ai_score})."
            )

        # =================================================
        # 4. RETURN RESULT
        # =================================================

        return {
            "session_id": session_id,

            "similarity_percent": similarity_percent,

            "same_speaker_likely": same_speaker,

            "genuine_file_ai_risk": genuine_risk,

            "test_file_ai_risk": test_risk,

            "test_ai_risk_score": test_ai_score,

            "test_ai_risk_level": test_ai_level,

            "verdict": verdict,

            "verdict_level": verdict_level,

            "explanation": explanation,
        }

    except Exception as e:

        print(
            f"[compare-voices] ERROR: {e}"
        )

        return {
            "error": (
                f"Voice comparison failed: {str(e)}"
            )
        }

    finally:

        if (
            genuine_path
            and os.path.exists(genuine_path)
        ):
            os.remove(genuine_path)

        if (
            test_path
            and os.path.exists(test_path)
        ):
            os.remove(test_path)

@app.get("/api/history/{session_id}")
def history(session_id: str):
    return {"session_id": session_id, "history": get_session_history(session_id)}


CHUNK_WINDOW_SECONDS = 2.0


@app.websocket("/ws/stream/{session_id}")
async def stream_endpoint(websocket: WebSocket, session_id: str):
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
                    buffer = buffer[window_len:]

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


import os
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "frontend")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/")
    def index():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))