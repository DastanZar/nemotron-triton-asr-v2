#!/usr/bin/env python3
"""
Interactive concurrent streaming demo.
Pick number of streams, iterations, chunk size — watch them all process simultaneously.
"""

import asyncio
import base64
import json
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf
import websockets

TARGET_SR = 16000
print_lock = asyncio.Lock()


def load_audio(path):
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sr != TARGET_SR:
        from scipy.signal import resample_poly
        audio = resample_poly(audio, TARGET_SR, sr).astype(np.float32)
    return np.clip(audio, -1.0, 1.0)


def iter_chunks(audio, chunk_ms):
    chunk_samples = max(1, int(TARGET_SR * chunk_ms / 1000))
    return [audio[i:i + chunk_samples] for i in range(0, len(audio), chunk_samples)]


async def log(msg):
    async with print_lock:
        print(msg, flush=True)


async def stream_loop(stream_id, audio_path, target_lang, chunk_ms, iterations, server):
    audio = load_audio(audio_path)
    audio_duration = len(audio) / TARGET_SR
    chunks = iter_chunks(audio, chunk_ms)
    short_id = stream_id[:14]
    start = time.perf_counter()
    first_text_chunk = None
    last_text = ""
    total_chunks = 0

    async with websockets.connect(server, max_size=8 * 1024 * 1024) as ws:
        for iteration in range(iterations):
            for i, chunk in enumerate(chunks):
                is_start = (iteration == 0 and i == 0)
                is_end = (iteration == iterations - 1 and i == len(chunks) - 1)

                pcm16 = (chunk * 32767).astype(np.int16).tobytes()
                payload = {
                    "stream_id": stream_id,
                    "target_lang": target_lang,
                    "sample_rate": TARGET_SR,
                    "encoding": "pcm_s16le",
                    "is_start": is_start,
                    "is_end": is_end,
                    "audio_b64": base64.b64encode(pcm16).decode(),
                }

                chunk_start = time.perf_counter()
                await ws.send(json.dumps(payload))
                resp = json.loads(await ws.recv())
                latency_ms = (time.perf_counter() - chunk_start) * 1000
                total_chunks += 1

                text = resp.get("text", "")
                if text and first_text_chunk is None:
                    first_text_chunk = total_chunks
                    await log(f"  [{short_id}] FIRST TEXT at chunk {total_chunks}: {text[:80]}")

                if text and text != last_text:
                    await log(f"  [{short_id}] iter={iteration+1} chunk={i+1}/{len(chunks)} {latency_ms:.0f}ms | {text[:80]}")
                    last_text = text

                if is_end:
                    await log(f"  [{short_id}] STREAM END | final: {text[:80]}")

                await asyncio.sleep(chunk_ms / 1000.0)

    elapsed = time.perf_counter() - start
    return {
        "id": short_id,
        "elapsed": elapsed,
        "chunks": total_chunks,
        "audio_dur": audio_duration,
        "first_text_at": first_text_chunk,
        "final_text": last_text,
    }


async def run(num_streams, num_iterations, chunk_ms, lang, server):
    test_dir = Path(__file__).parent.parent / "test_audio"
    if lang == "hi":
        files = sorted((test_dir / "hindi").glob("*.wav"))
    else:
        files = sorted((test_dir / "english").glob("*.wav"))

    chunks_per_file = len(iter_chunks(load_audio(files[0]), chunk_ms))

    await log(f"\n{'='*70}")
    await log(f"  CONCURRENT STREAMING DEMO")
    await log(f"  Streams: {num_streams} | Iterations: {num_iterations} | Chunk: {chunk_ms}ms")
    await log(f"  Chunks/stream: {chunks_per_file} x {num_iterations} = {chunks_per_file * num_iterations}")
    await log(f"  Files: {len(files)} available, cycling through them")
    await log(f"{'='*70}\n")
    await log("  Streaming... (text appears after model accumulates enough audio)\n")

    coros = []
    for s in range(num_streams):
        audio_file = files[s % len(files)]
        stream_id = f"stream-{s+1:02d}-{uuid.uuid4().hex[:6]}"
        coros.append(stream_loop(stream_id, str(audio_file), lang, chunk_ms, num_iterations, server))

    wall_start = time.perf_counter()
    results = await asyncio.gather(*coros, return_exceptions=True)
    wall_elapsed = time.perf_counter() - wall_start

    await log(f"\n{'='*70}")
    await log(f"  RESULTS")
    await log(f"{'='*70}")
    await log(f"  {'Stream':<16} {'Audio':>6} {'Chunks':>7} {'Time':>8} {'RTF':>7} {'1st Text':>10}")
    await log(f"  {'-'*58}")

    total_audio = 0
    for r in results:
        if isinstance(r, Exception):
            await log(f"  ERROR: {r}")
        else:
            total_audio += r["audio_dur"] * num_iterations
            stream_rtf = (r["audio_dur"] * num_iterations) / r["elapsed"] if r["elapsed"] > 0 else 0
            first = f"ch={r['first_text_at']}" if r["first_text_at"] else "none"
            await log(f"  {r['id']:<16} {r['audio_dur']:>5.1f}s {r['chunks']:>7} {r['elapsed']:>7.2f}s {stream_rtf:>6.2f}x {first:>10}")

    overall_rtf = total_audio / wall_elapsed if wall_elapsed > 0 else 0
    await log(f"\n  Wall time: {wall_elapsed:.2f}s")
    await log(f"  Total audio: {total_audio:.1f}s | Throughput: {overall_rtf:.2f}x real-time")
    await log(f"  Streams: {num_streams} | Chunks/sec: {sum(r['chunks'] for r in results if not isinstance(r, Exception)) / wall_elapsed:.0f}")
    await log(f"{'='*70}\n")


def main():
    print("\n--- Concurrent Streaming Demo ---\n")

    num_streams = int(input("Number of concurrent streams [1-32]: ") or "4")
    num_iterations = int(input("Iterations per stream (loops) [1-100]: ") or "3")
    chunk_ms = int(input("Chunk size in ms [80/160/320]: ") or "320")
    lang = input("Language (en/hi) [en]: ").strip() or "en"
    lang_code = "hi" if lang == "hi" else "auto"

    num_streams = max(1, min(32, num_streams))
    num_iterations = max(1, min(100, num_iterations))
    chunk_ms = max(80, min(1000, chunk_ms))

    server = "ws://localhost:8000/v1/stream"

    asyncio.run(run(num_streams, num_iterations, chunk_ms, lang_code, server))


if __name__ == "__main__":
    main()
