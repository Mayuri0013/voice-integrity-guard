# Voice Integrity Guard
### Real-time AI voice cloning / impersonation detection — SIH 2026, Cyber Security domain

A working end-to-end prototype that analyzes a voice call (uploaded clip or
simulated live stream) and produces a real-time **impersonation risk score**,
with alerting and secondary-verification recommendations — before a
sensitive action (fund transfer, OTP disclosure, privileged approval) is
taken.

This is a **fully functional prototype**, tested end-to-end. Read the
"What's real vs. what's a placeholder" section before you demo it — it will
help you answer judges' questions honestly and confidently.

---

## 1. Quick start

```bash
cd backend
pip install -r requirements.txt
python train_baseline.py        # trains & saves the baseline ML model (~10 sec)
uvicorn app.main:app --reload --port 8000
```

Then open **http://localhost:8000** in a browser. That's it — the dashboard,
API, and WebSocket streaming endpoint are all served from this one command.

Two demo audio files are in `demo/` — one engineered to have natural pitch/
amplitude micro-variation, one engineered to be artificially smooth (mimics
a key TTS/vocoder signature). Upload either one in the dashboard and try
both "Run one-shot analysis" and "Simulate live call".

Verified test results from this build:
| Sample | Risk score | Level |
|---|---|---|
| `sample_genuine_like.wav` | 30.7 | LOW |
| `sample_synthetic_like.wav` | 69.5–72.7 | MEDIUM/HIGH |

---

## 2. Architecture

```
┌─────────────┐     upload / stream audio      ┌──────────────────────┐
│  Dashboard  │ ───────────────────────────────▶│   FastAPI backend    │
│ (HTML/JS)   │◀─────── live risk updates ──────│                       │
└─────────────┘         (WebSocket / REST)       │  1. feature_extraction│
                                                  │  2. risk_engine       │
                                                  │  3. ml_model (RF)     │
                                                  │  4. alerts             │
                                                  │  5. privacy             │
                                                  └──────────────────────┘
```

| File | Problem-statement component it implements |
|---|---|
| `feature_extraction.py` | Multi-Layer Voice Authenticity Analysis (spectral + prosody) |
| `risk_engine.py` | Real-Time Risk Scoring Engine + contextual enrichment |
| `ml_model.py` / `train_baseline.py` | Pluggable ML classifier slot |
| `alerts.py` | Alerting and User Interaction Layer / configurable workflows |
| `privacy.py` | Privacy and Compliance Module (feature-only logging, no raw audio storage) |
| `main.py` | Platform and Integration APIs (REST + WebSocket + CORS for enterprise integration) |
| `frontend/` | Dashboard for frontline staff / end users |

---

## 3. What's real vs. what's a placeholder (be upfront about this in your demo)

**Fully real and working:**
- Audio decoding, resampling, and **real acoustic/prosodic feature extraction**
  (MFCC, spectral flatness/centroid/bandwidth/rolloff, pitch tracking via
  `librosa.pyin`, jitter/shimmer proxies, pause ratio) — these are genuine
  signal-processing computations on the actual waveform, not mocked.
- The **heuristic risk engine** — an interpretable, rule-based scorer using
  literature-informed reference ranges (natural human speech shows more
  pitch/amplitude micro-variation than most neural TTS output; unusually
  smooth = higher risk). This works with zero training data and is fully
  explainable to judges.
- **Real-time streaming**: the WebSocket endpoint genuinely buffers,
  analyzes, and scores ~2-second audio windows as they arrive, the same
  way a live telephony tap would.
- **Contextual risk boosting**, alerting, alert logging, and the
  privacy-preserving session store (raw audio is never written to disk;
  only derived numeric features + scores are retained, and sessions
  auto-expire).

**Placeholder / needs real data before production claims:**
- The **ML classifier** (`baseline_model.joblib`) is trained on a
  **synthetically generated proxy dataset** (see `train_baseline.py`'s
  docstring), not real labeled human-vs-cloned recordings. It demonstrates
  a fully working train → save → load → predict pipeline, but its
  "accuracy" numbers are only meaningful on that synthetic proxy set —
  don't present them as real-world detection accuracy.
- To make this production-real: get a labeled dataset (ASVspoof 2019/2021
  LA partition is the standard academic benchmark, or record your own
  genuine vs. cloned samples using Coqui-TTS/RVC), write a loader that
  calls `feature_extraction.extract_features()` on each file, and retrain.
  No other code changes needed — `ml_model.py` just loads whatever
  `train_baseline.py` produces.
- Heuristic thresholds in `risk_engine.py`'s `NATURAL_RANGES` are
  reasonable starting points, not calibrated on ground truth — recalibrate
  once you have labeled data.

**How to talk about this to judges:** frame it as "a working, retrainable
detection architecture with a real signal-processing pipeline, currently
running on a demonstration baseline — the exact integration point for
ASVspoof-trained weights is already built and wired in." That's honest and
still a strong, working demo.

---

## 4. Roadmap to strengthen this before final submission

1. **Swap in real training data.** Highest-impact step. Use ASVspoof
   2019/2021, or record ~30 min of your own team's voices + generate
   cloned versions with a free TTS tool, extract features, retrain.
2. **Upgrade the model.** Once you have real data, a small CNN on
   mel-spectrograms (or fine-tuning a pretrained wav2vec2/RawNet2-style
   model) will beat the RandomForest baseline — architecture in
   `ml_model.py` can host any `.predict_proba`-compatible classifier.
3. **Multilingual/accent coverage.** Add Indian-language speech samples
   (Mozilla Common Voice Hindi/Tamil/Bengali, IIT-Madras Indic-TTS) to the
   training set — the feature extractor is language-agnostic already.
4. **Edge/on-device inference.** For the privacy story, mention exporting
   the trained model to ONNX for on-device inference (ties into the
   "minimal retention" bullet of the problem statement).
5. **Real telephony/VoIP integration.** Currently the "live call" is
   simulated by streaming an uploaded file. For a stronger pitch, mention
   integrating with Twilio Media Streams or a SIP/RTP tap as the real
   audio source feeding the same WebSocket pipeline — the backend code
   doesn't change, only the audio source.
6. **Speaker verification cross-check.** Add a pretrained speaker
   embedding model (e.g. SpeechBrain's ECAPA-TDNN) to compare the live
   caller against a known genuine reference sample, per the "Cross-session
   consistency checks" bullet — this is a natural next module.

---

## 5. Project structure

```
voice-integrity-guard/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app: REST + WebSocket endpoints
│   │   ├── feature_extraction.py
│   │   ├── risk_engine.py
│   │   ├── ml_model.py
│   │   ├── alerts.py
│   │   ├── privacy.py
│   │   ├── schemas.py
│   │   └── models/
│   │       └── baseline_model.joblib   (generated by train_baseline.py)
│   ├── train_baseline.py
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── demo/
│   ├── sample_genuine_like.wav
│   └── sample_synthetic_like.wav
└── README.md
```

## 6. API reference

- `GET /api/health` — health check
- `POST /api/analyze` — multipart form upload (`file`, plus optional
  `unknown_caller` / `high_value_transaction` / `privileged_request`
  booleans) → returns risk score + breakdown
- `WS /ws/stream/{session_id}` — send binary Float32 PCM chunks (16kHz
  mono) or JSON control messages (`{"type":"context","context":{...}}`,
  `{"type":"end"}`); receives `{"type":"risk_update", ...}` JSON pushes
- `GET /api/alerts/{session_id}` — alert history for a session
- `GET /api/history/{session_id}` — full feature/score history for a session
