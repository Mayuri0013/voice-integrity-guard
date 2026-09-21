const API_BASE = window.location.origin;
const WS_BASE = API_BASE.replace(/^http/, "ws");

const fileInput = document.getElementById("fileInput");
const dropzone = document.getElementById("dropzone");
const dropzoneLabel = document.getElementById("dropzoneLabel");
const analyzeBtn = document.getElementById("analyzeBtn");
 
const micBtn = document.getElementById("micBtn");
const micTimer = document.getElementById("micTimer");
const statusLine = document.getElementById("statusLine");
const sessionPill = document.getElementById("sessionPill");

const gaugeFill = document.getElementById("gaugeFill");
const gaugeNumber = document.getElementById("gaugeNumber");
const gaugeLevel = document.getElementById("gaugeLevel");
const recommendation = document.getElementById("recommendation");
const featureList = document.getElementById("featureList");
const alertList = document.getElementById("alertList");
const scoreBars = document.getElementById("scoreBars");

const GAUGE_CIRC = 251.2;
let currentFile = null;
let ws = null;
let scoreHistory = [];

let micStream = null;
let micAudioCtx = null;
let micProcessor = null;
let micSource = null;
let micWs = null;
let micBuffer = [];
let micRecording = false;
let micChunkInterval = null;
let micTimerInterval = null;
let micStartTime = null;

function levelColor(level) {
  if (level === "HIGH") return getCSS("--high");
  if (level === "MEDIUM") return getCSS("--medium");
  return getCSS("--low");
}
function getCSS(varName) {
  return getComputedStyle(document.documentElement).getPropertyValue(varName).trim();
}

function setGauge(score, level) {
  const clamped = Math.max(0, Math.min(100, score));
  const offset = GAUGE_CIRC - (clamped / 100) * GAUGE_CIRC;
  gaugeFill.style.strokeDashoffset = offset;
  gaugeFill.style.stroke = levelColor(level);
  gaugeNumber.textContent = Math.round(clamped);
  gaugeLevel.textContent = level ? `${level} RISK` : "no data";
}

function pushScoreBar(score, level) {
  scoreHistory.push({ score, level });
  if (scoreHistory.length > 40) scoreHistory.shift();
  scoreBars.innerHTML = "";
  scoreHistory.forEach(({ score, level }) => {
    const bar = document.createElement("div");
    bar.className = "bar";
    bar.style.height = `${Math.max(4, (score / 100) * 54)}px`;
    bar.style.background = levelColor(level);
    scoreBars.appendChild(bar);
  });
}

function renderFeatureBreakdown(breakdown) {
  featureList.innerHTML = "";
  const entries = Object.entries(breakdown || {});
  if (!entries.length) {
    featureList.innerHTML = '<div class="muted small">No sample analyzed yet.</div>';
    return;
  }
  entries
    .sort((a, b) => b[1] - a[1])
    .forEach(([name, val]) => {
      const row = document.createElement("div");
      row.className = "feature-row";
      row.innerHTML = `<span class="fname">${prettyName(name)}</span><span class="fval">${val}</span>`;
      featureList.appendChild(row);
    });
}

function prettyName(name) {
  return name
    .replace(/_/g, " ")
    .replace("mean", "")
    .replace("proxy", "")
    .trim()
    .replace(/^\w/, (c) => c.toUpperCase());
}

function renderRecommendation(risk) {
  recommendation.textContent = risk.recommendation;
}

function addAlert(alert) {
  if (!alert) return;
  if (alertList.querySelector(".muted")) alertList.innerHTML = "";
  const row = document.createElement("div");
  row.className = `alert-row ${alert.risk_level ? alert.risk_level.toLowerCase() : ""}`;
  const time = new Date((alert.timestamp || Date.now() / 1000) * 1000).toLocaleTimeString();
  row.innerHTML = `<div class="alert-head"><span>${alert.risk_level}</span><span>${time}</span></div>${alert.recommendation}`;
  alertList.prepend(row);
}

dropzone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => {
  if (fileInput.files.length) {
    currentFile = fileInput.files[0];
    dropzoneLabel.textContent = currentFile.name;
    dropzone.classList.add("has-file");
    analyzeBtn.disabled = false;
 
    statusLine.textContent = "Ready. Choose an audio file to analyze.";
  }
});

analyzeBtn.addEventListener("click", async () => {
  if (!currentFile) return;
  statusLine.textContent = "Analyzing…";
  analyzeBtn.disabled = true;

  const form = new FormData();
  form.append("file", currentFile);
   

  try {
    const res = await fetch(`${API_BASE}/api/analyze`, { method: "POST", body: form });
    const data = await res.json();
    if (data.error) {
      statusLine.textContent = `Error: ${data.error}`;
      analyzeBtn.disabled = false;
      return;
    }
    sessionPill.textContent = `session ${data.session_id.slice(0, 8)}`;
    const risk = data.risk_result;
    setGauge(risk.risk_score, risk.risk_level);
    pushScoreBar(risk.risk_score, risk.risk_level);
    renderFeatureBreakdown(risk.feature_breakdown);
    renderRecommendation(risk);
    if (data.alert_raised) addAlert(data.alert_raised);
    statusLine.textContent = `Analysis complete — ${data.features_extracted} features extracted.`;
  } catch (e) {
    statusLine.textContent = `Request failed: ${e}`;
  } finally {
    analyzeBtn.disabled = false;
  }
});
 
function handleStreamMessage(event) {
  const msg = JSON.parse(event.data);
  if (msg.type === "risk_update") {
    const risk = msg.risk_result;
    setGauge(risk.risk_score, risk.risk_level);
    pushScoreBar(risk.risk_score, risk.risk_level);
    renderFeatureBreakdown(risk.feature_breakdown);
    renderRecommendation(risk);
    if (msg.alert) addAlert(msg.alert);
    statusLine.textContent = `Live — running average risk: ${msg.running_avg_score}`;
  } else if (msg.type === "error") {
    statusLine.textContent = `Stream error: ${msg.message}`;
  }
}
 

micBtn.addEventListener("click", async () => {
  if (!micRecording) {
    await startMicStream();
  } else {
    stopMicStream();
  }
});

async function startMicStream() {
  try {
    micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (e) {
    statusLine.textContent = "Microphone access denied or unavailable in this browser.";
    return;
  }

  micRecording = true;
  micBtn.classList.add("recording");
  micBtn.textContent = "⏹ Stop recording";
  analyzeBtn.disabled = true;
  
  scoreHistory = [];
  scoreBars.innerHTML = "";

  const sessionId = crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}`;
  sessionPill.textContent = `session ${sessionId.slice(0, 8)} · microphone`;

  micAudioCtx = new (window.AudioContext || window.webkitAudioContext)();
  micSource = micAudioCtx.createMediaStreamSource(micStream);
  micProcessor = micAudioCtx.createScriptProcessor(4096, 1, 1);
  micBuffer = [];

  micProcessor.onaudioprocess = (e) => {
    micBuffer.push(new Float32Array(e.inputBuffer.getChannelData(0)));
  };

  const silentGain = micAudioCtx.createGain();
  silentGain.gain.value = 0;
  micSource.connect(micProcessor);
  micProcessor.connect(silentGain);
  silentGain.connect(micAudioCtx.destination);

  micWs = new WebSocket(`${WS_BASE}/ws/stream/${sessionId}`);
  micWs.binaryType = "arraybuffer";

  micWs.onopen = () => {
    micWs.send(JSON.stringify({
      type: "context",
      context: { },
    }));
  };
  micWs.onmessage = handleStreamMessage;
  micWs.onclose = () => {
    if (micRecording) statusLine.textContent = "Microphone stream disconnected.";
  };

  const chunkSeconds = 2.0;
  const targetRate = 16000;

  micChunkInterval = setInterval(async () => {
    if (!micBuffer.length) return;
    const chunks = micBuffer;
    micBuffer = [];

    const totalLength = chunks.reduce((sum, c) => sum + c.length, 0);
    const merged = new Float32Array(totalLength);
    let off = 0;
    for (const c of chunks) { merged.set(c, off); off += c.length; }

    const nativeRate = micAudioCtx.sampleRate;
    const offlineCtx = new OfflineAudioContext(
      1, Math.ceil((merged.length * targetRate) / nativeRate), targetRate
    );
    const buf = offlineCtx.createBuffer(1, merged.length, nativeRate);
    buf.copyToChannel(merged, 0);
    const src = offlineCtx.createBufferSource();
    src.buffer = buf;
    src.connect(offlineCtx.destination);
    src.start();
    const resampled = await offlineCtx.startRendering();
    const pcm16k = resampled.getChannelData(0);

    if (micWs && micWs.readyState === WebSocket.OPEN) {
      micWs.send(pcm16k.buffer);
    }
  }, chunkSeconds * 1000);

  micStartTime = Date.now();
  micTimerInterval = setInterval(() => {
    const elapsed = Math.floor((Date.now() - micStartTime) / 1000);
    const m = String(Math.floor(elapsed / 60)).padStart(2, "0");
    const s = String(elapsed % 60).padStart(2, "0");
    micTimer.textContent = `● Recording ${m}:${s}`;
  }, 500);

  statusLine.textContent = "Recording from your microphone — speak naturally, risk updates every ~2s.";
}

function stopMicStream() {
  micRecording = false;
  micBtn.classList.remove("recording");
  micBtn.textContent = "🎙 Test with your microphone";
  analyzeBtn.disabled = !currentFile;
   
  micTimer.textContent = "";

  if (micChunkInterval) clearInterval(micChunkInterval);
  if (micTimerInterval) clearInterval(micTimerInterval);

  if (micProcessor) {
    micProcessor.disconnect();
    micProcessor.onaudioprocess = null;
  }
  if (micSource) micSource.disconnect();
  if (micAudioCtx) micAudioCtx.close();
  if (micStream) micStream.getTracks().forEach((t) => t.stop());

  if (micWs) {
    if (micWs.readyState === WebSocket.OPEN) {
      micWs.send(JSON.stringify({ type: "end" }));
      micWs.close();
    }
  }
  statusLine.textContent = "Microphone recording stopped.";
}

const genuineDropzone = document.getElementById("genuineDropzone");
const genuineFileInput = document.getElementById("genuineFileInput");
const genuineDropzoneLabel = document.getElementById("genuineDropzoneLabel");
const testDropzone = document.getElementById("testDropzone");
const testFileInput = document.getElementById("testFileInput");
const testDropzoneLabel = document.getElementById("testDropzoneLabel");
const compareBtn = document.getElementById("compareBtn");
const compareStatus = document.getElementById("compareStatus");
const compareResult = document.getElementById("compareResult");
const compareVerdictBanner = document.getElementById("compareVerdictBanner");
const compareVerdictText = document.getElementById("compareVerdictText");
const compareExplanation = document.getElementById("compareExplanation");
const compareSimilarity = document.getElementById("compareSimilarity");
const compareSameSpeaker = document.getElementById("compareSameSpeaker");
const compareGenuineRisk = document.getElementById("compareGenuineRisk");
const compareTestRisk = document.getElementById("compareTestRisk");

let genuineFile = null;
let compareTestFile = null;

genuineDropzone.addEventListener("click", () => genuineFileInput.click());
genuineFileInput.addEventListener("change", () => {
  if (genuineFileInput.files.length) {
    genuineFile = genuineFileInput.files[0];
    genuineDropzoneLabel.textContent = genuineFile.name;
    genuineDropzone.classList.add("has-file");
    updateCompareBtnState();
  }
});

testDropzone.addEventListener("click", () => testFileInput.click());
testFileInput.addEventListener("change", () => {
  if (testFileInput.files.length) {
    compareTestFile = testFileInput.files[0];
    testDropzoneLabel.textContent = compareTestFile.name;
    testDropzone.classList.add("has-file");
    updateCompareBtnState();
  }
});

function updateCompareBtnState() {
  compareBtn.disabled = !(genuineFile && compareTestFile);
  if (genuineFile && compareTestFile) {
    compareStatus.textContent = "Ready to compare.";
  }
}

compareBtn.addEventListener("click", async () => {
  if (!genuineFile || !compareTestFile) return;
  compareBtn.disabled = true;
  compareStatus.textContent = "Comparing voices — this can take a few seconds…";

  const form = new FormData();
  form.append("genuine_file", genuineFile);
  form.append("test_file", compareTestFile);

  try {
    const res = await fetch(`${API_BASE}/api/compare-voices`, { method: "POST", body: form });
    const data = await res.json();

    if (data.error) {
      compareStatus.textContent = `Error: ${data.error}`;
      compareBtn.disabled = false;
      return;
    }

    compareResult.style.display = "block";

    const levelClass = data.verdict_level ? data.verdict_level.toLowerCase() : "low";
    compareVerdictBanner.className = `compare-verdict-banner ${levelClass}`;
    compareVerdictText.textContent = data.verdict;
    compareExplanation.textContent = data.explanation;

    compareSimilarity.textContent = `${data.similarity_percent}%`;
    compareSameSpeaker.textContent = data.same_speaker_likely ? "Same speaker likely" : "Different speaker likely";

    const gRisk = data.genuine_file_ai_risk || {};
    const tRisk = data.test_file_ai_risk || {};
    compareGenuineRisk.textContent = gRisk.risk_level ? `${gRisk.risk_score} · ${gRisk.risk_level}` : "--";
    compareTestRisk.textContent = tRisk.risk_level ? `${tRisk.risk_score} · ${tRisk.risk_level}` : "--";

    compareStatus.textContent = "Comparison complete.";
  } catch (e) {
    compareStatus.textContent = `Request failed: ${e}`;
  } finally {
    compareBtn.disabled = false;
  }
});