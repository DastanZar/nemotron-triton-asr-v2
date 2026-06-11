# Nemotron 3.5 ASR Streaming on Triton

Real-time streaming ASR using `nvidia/nemotron-3.5-asr-streaming-0.6b` behind NVIDIA Triton Inference Server.

Two containers:
- **`nemotron-triton`** — Triton 25.05 + NeMo Python backend (ports 8001/8002/8003)
- **`nemotron-gateway`** — FastAPI gateway + WebSocket endpoint (port 8000)

## Quick Start

```bash
git clone https://github.com/DastanZar/nemotron-triton-asr-v2.git
cd nemotron-triton-asr-v2
docker compose build
docker compose up -d
```

Full setup guide → [USER_GUIDE.md](USER_GUIDE.md)

## Client Scripts

| Script | Purpose |
|--------|---------|
| `client/interactive_demo.py` | Live per-stream transcription view |
| `client/metrics_demo.py` | Per-stream RTF, TTFT, TTBL, latency breakdown |
| `client/benchmark.py` | Automated concurrency sweep benchmark |

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/stream` | WebSocket | Real-time streaming per chunk |
| `/v1/transcribe` | POST | File-based transcription |
| `/v1/files` | GET | List available test audio files |
| `/healthz` | GET | Server health check |
| `/` | GET | Web dashboard |

## Architecture

```
Client (WebSocket/Python) → Gateway (FastAPI :8000) → Triton (:8001) → NeMo (GPU)
```

Triton uses a Python backend model that loads the NeMo checkpoint and maintains per-stream cache state. Dynamic batching coalesces chunk requests from concurrent streams for GPU efficiency.
