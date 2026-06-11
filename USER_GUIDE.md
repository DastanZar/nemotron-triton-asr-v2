# Nemotron 3.5 ASR Streaming — User Guide

## Prerequisites
- Linux server with **NVIDIA GPU** (tested on L40S, 48GB VRAM)
- **Docker + Docker Compose** (v2+)
- **NVIDIA Container Toolkit** (`nvidia-ctk` installed)
- Python 3.10+ on host (for client scripts)
- Test audio files (WAV, 16kHz mono)

## Quick Start

### 1. Clone and Build
```bash
git clone https://github.com/DastanZar/nemotron-triton-asr-v2.git
cd nemotron-triton-asr-v2
docker compose build --no-cache   # ~30 min first time (NeMo from source)
```

### 2. Prepare Test Audio
Place audio files in:
```
test_audio/english/  → *.wav (16kHz mono preferred)
test_audio/hindi/    → *.wav
```
Or copy from an existing setup:
```bash
cp -r /path/to/existing/test_audio/* test_audio/
```

### 3. Launch
```bash
docker compose up -d
# Watch logs:
docker compose logs -f triton   # wait for "server is ready"
docker compose logs -f gateway  # wait for "Application startup complete"
```

### 4. Verify
```bash
curl http://localhost:8000/healthz
# → {"ok": true}

curl http://localhost:8000/v1/files
# → {"english": ["file1.wav", ...], "hindi": ["file1.wav", ...]}
```

### 5. Run Demos

**Interactive transcription:**
```bash
pip3 install websockets numpy soundfile scipy
python3 client/interactive_demo.py
```
Prompts for: streams, iterations, chunk size (80/160/320ms), language (en/hi)

**Per-stream metrics:**
```bash
python3 client/metrics_demo.py
```
Prompts for: streams, iterations, chunk size, language
Outputs per-stream RTF, Speed, TTFT, TTBL, p50/p95 latency

**Web dashboard:**
Open `http://localhost:8000` in a browser.

**Automated benchmark:**
```bash
python3 client/benchmark.py
```
Sweeps concurrency from 1 to 32, outputs results table + JSON.

## Troubleshooting

### "No module named 'websockets'"
```bash
pip3 install websockets numpy soundfile scipy
```

### Gateway can't connect to Triton
```bash
docker compose logs triton | tail -20
docker compose logs gateway | tail -20
```
Ensure both containers are running: `docker compose ps`

### Hindi returns empty text at 80ms
This is a known model limitation — use 160ms or 320ms for Hindi.

### OOM / GPU memory errors
Reduce `max_batch_size` in `model_repository/nemotron_streaming/config.pbtxt` (line 3).

### "Triton unavailable" in health check
Wait for Triton to finish downloading the model (first run only, ~2GB download).

## File Layout
```
nemotron-triton-asr-v2/
├── docker-compose.yml          ← Service orchestration (entry point)
├── deploy/
│   ├── Dockerfile.triton       ← Triton + NeMo image build
│   └── Dockerfile.gateway      ← FastAPI gateway image build
├── gateway/
│   ├── app.py                  ← FastAPI server (routes, WebSocket, Triton client)
│   └── dashboard.html          ← Web dashboard UI
├── model_repository/
│   └── nemotron_streaming/
│       ├── config.pbtxt        ← Model configuration (inputs, outputs, batching)
│       └── 1/
│           └── model.py        ← Triton Python backend (model load & inference)
├── client/
│   ├── interactive_demo.py     ← Live transcription viewer
│   ├── metrics_demo.py         ← Per-stream metrics (TTFT, TTBL, RTF, latency)
│   └── benchmark.py            ← Automated concurrency sweep benchmark
├── test_audio/
│   ├── english/                ← English FLEURS WAV clips
│   └── hindi/                  ← Hindi FLEURS WAV clips
├── CONTEXT_CARRY.md            ← AI assistant context (full project history)
└── USER_GUIDE.md               ← This file
```

## Chunk Size Guide
| Chunk | Use Case |
|-------|----------|
| 80ms  | English only, lower latency (~500ms TTFT) |
| 160ms | Hindi & English, good balance (~800ms TTFT) |
| 320ms | Default, best transcription quality (~1.5s TTFT) |
