# Context Carry — Nemotron 3.5 ASR Streaming on Triton

## Project Identity
- **Repo**: DastanZar/nemotron-triton-asr-v2 (fresh repo, no history)
- **Branch**: main
- **Deploy path on server**: /home/ubuntu/nemotron-triton/
- **Server**: AWS EC2 (L40S GPU, 48GB VRAM)
- **User accesses**: VS Code Remote-SSH from laptop

## Architecture
```
Client (Python/WebSocket)  →  Gateway (FastAPI :8000)  →  Triton (Python backend :8001)
                                ↑                               ↓
                            dashboard.html              model_repository/nemotron_streaming/1/model.py
```

Two containers:
- `nemotron-triton` — Triton 25.05 + NeMo from source, ports 8001/8002/8003
- `nemotron-gateway` — Python 3.12 + FastAPI + tritonclient[http], port 8000

## Model
- **Model**: nvidia/nemotron-3.5-asr-streaming-0.6b
- **Att context size**: [56, 3] (56 frames left context, 3 frames right lookahead)
- **Subsampling factor**: 8
- **drop_extra_pre_encoded**: 2

## Critical Fixes (model.py lines 208-234)
- `drop_extra_pre_encoded=2` collapses feature frames to 0 for short sequences
- **Fix**: Skip `drop_extra` for:
  - First chunks (is_start=True)
  - Last chunks (is_end=True)
  - Any chunk with <20 frames after preprocessor
- Added INFO logging for preprocessor output shapes and drop_extra decisions

## Streaming Protocol (WebSocket /v1/stream)
Client sends JSON per chunk:
```json
{"stream_id": "...", "target_lang": "auto", "sample_rate": 16000,
 "encoding": "pcm_s16le", "is_start": true, "is_end": false,
 "audio_b64": "<base64 pcm16>"}
```
Server responds per chunk:
```json
{"stream_id": "...", "text": "...", "is_final": false, "language": "en"}
```

## Chunk Size Behavior
| Chunk | English | Hindi |
|-------|---------|-------|
| 80ms  | Works | Empty text (model limitation) |
| 160ms | Works | Works (minimum for Hindi) |
| 320ms | Works | Works (default, best quality) |

## Performance (Hindi 160ms, 8 concurrent)
- Per-stream RTF: 0.45–0.61 (all <1, fast)
- Per-stream Speed: 1.63x–2.23x
- Overall throughput: ~9.5x (8 streams)
- Chunk latency p50: ~90ms, p95: ~150ms
- TTFT: 447–1176ms (5-13 chunks of context needed)

## Client Scripts
1. `client/interactive_demo.py` — Live per-stream transcription view
2. `client/metrics_demo.py` — Per-stream RTF/TTFT/TTBL/latency breakdown
3. `client/benchmark.py` — Automated concurrency sweep (DO NOT MODIFY)

## Key Files
- `docker-compose.yml` — Service orchestration
- `deploy/Dockerfile.triton` — Triton + NeMo from source build
- `deploy/Dockerfile.gateway` — FastAPI gateway build
- `gateway/app.py` — FastAPI routes, thread-local Triton clients
- `gateway/dashboard.html` — Web dashboard with manual WAV parsing
- `model_repository/nemotron_streaming/config.pbtxt` — Model config (dynamic batching, params)
- `model_repository/nemotron_streaming/1/model.py` — Triton Python backend (core logic)

## Important Decisions
- **Thread-local Triton clients**: Single shared client serialized requests. Thread-local via `threading.local()` enables parallelism.
- **CUDA graph decoder disabled**: `fused_batch_size=-1`, `use_cuda_graph_decoder=False` in model.py.
- **No artificial sleep in metrics_demo**: Sends chunks back-to-back for true server processing measurement.
- **RTF convention**: RTF = wall_time / audio_duration (< 1 = fast). Speed = audio_duration / wall_time (> 1 = fast).
- **Dynamic batching**: pref [4, 8, 16, 32], max_queue_delay 2000μs.
- **No secrets in git**: PAT stored in git remote URL, NGC key in env vars.

## Test Audio
- /home/ubuntu/nemotron-triton/test_audio/english/ — 32 English FLEURS WAV clips
- /home/ubuntu/nemotron-triton/test_audio/hindi/ — 15 Hindi FLEURS WAV clips
- To generate: run client/download_fleurs.py or copy from benchmark project
