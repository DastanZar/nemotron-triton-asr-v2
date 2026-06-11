#!/usr/bin/env python3
"""
Per-stream metrics demo.
N concurrent streams with TTFT, TTBL, per-stream RTF, chunk latency breakdown.
"""

import asyncio
import base64
import json
import statistics
import time
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf
import websockets

TARGET_SR = 16000
PRINT_LOCK = asyncio.Lock()


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
    async with PRINT_LOCK:
        print(msg, flush=True)


class StreamMetrics:
    def __init__(self, stream_id, audio_path, iterations):
        self.stream_id = stream_id[:14]
        self.file = str(Path(audio_path).name)
        single_duration = len(load_audio(audio_path)) / TARGET_SR
        self.audio_duration = single_duration * iterations
        self.chunk_count = 0
        self.chunk_latencies = []
        self.first_text_at_chunk = None
        self.first_text_time = None
        self.last_response_time = None
        self.first_chunk_sent_time = None
        self.final_text = ""
        self.error = None
        self.start_time = None

    @property
    def total_time(self):
        if self.first_chunk_sent_time is None or self.last_response_time is None:
            return None
        return self.last_response_time - self.first_chunk_sent_time

    @property
    def rtf(self):
        t = self.total_time
        if t is None or t <= 0 or self.audio_duration <= 0:
            return None
        return round(t / self.audio_duration, 3)

    @property
    def throughput(self):
        t = self.total_time
        if t is None or t <= 0 or self.audio_duration <= 0:
            return None
        return round(self.audio_duration / t, 2)

    @property
    def ttft_ms(self):
        if self.first_text_time is None or self.first_chunk_sent_time is None:
            return None
        return round((self.first_text_time - self.first_chunk_sent_time) * 1000, 1)

    @property
    def ttbl_ms(self):
        if self.last_response_time is None or self.first_chunk_sent_time is None:
            return None
        return round((self.last_response_time - self.first_chunk_sent_time) * 1000, 1)

    @property
    def chunk_latency_p50(self):
        if not self.chunk_latencies:
            return None
        return round(statistics.median(self.chunk_latencies) * 1000, 1)

    @property
    def chunk_latency_p95(self):
        if not self.chunk_latencies:
            return None
        return round(statistics.quantiles(self.chunk_latencies, n=20)[-1] * 1000, 1)

    @property
    def chunk_latency_avg(self):
        if not self.chunk_latencies:
            return None
        return round(statistics.mean(self.chunk_latencies) * 1000, 1)


async def stream_loop(stream_id, audio_path, target_lang, chunk_ms, iterations, server):
    m = StreamMetrics(stream_id, audio_path, iterations)
    audio = load_audio(audio_path)
    chunks = iter_chunks(audio, chunk_ms)

    try:
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

                    if m.first_chunk_sent_time is None:
                        m.first_chunk_sent_time = time.perf_counter()

                    t0 = time.perf_counter()
                    await ws.send(json.dumps(payload))
                    resp = json.loads(await ws.recv())
                    dt = time.perf_counter() - t0
                    m.chunk_latencies.append(dt)
                    m.chunk_count += 1
                    m.last_response_time = time.perf_counter()

                    text = resp.get("text", "")
                    if text and m.first_text_at_chunk is None:
                        m.first_text_at_chunk = m.chunk_count
                        m.first_text_time = time.perf_counter()
                        await log(
                            f"  [{m.stream_id}] TTFT={m.ttft_ms}ms at chunk {m.chunk_count}"
                        )

                    if text and text != m.final_text:
                        m.final_text = text
    except Exception as e:
        m.error = str(e)

    return m


async def run(num_streams, num_iterations, chunk_ms, lang, server):
    test_dir = Path(__file__).parent.parent / "test_audio"
    if lang == "hi":
        files = sorted((test_dir / "hindi").glob("*.wav"))
    else:
        files = sorted((test_dir / "english").glob("*.wav"))

    chunks_per_file = len(iter_chunks(load_audio(files[0]), chunk_ms))

    await log(f"\n{'='*72}")
    await log(f"  PER-STREAM METRICS")
    await log(f"  Streams: {num_streams} | Iterations: {num_iterations} | Chunk: {chunk_ms}ms")
    await log(f"  Chunks/stream: {chunks_per_file} x {num_iterations} = {chunks_per_file * num_iterations}")
    await log(f"{'='*72}\n")

    coros = []
    for s in range(num_streams):
        audio_file = files[s % len(files)]
        stream_id = f"stream-{s+1:02d}-{uuid.uuid4().hex[:6]}"
        coros.append(stream_loop(stream_id, str(audio_file), lang, chunk_ms, num_iterations, server))

    wall_start = time.perf_counter()
    results = await asyncio.gather(*coros, return_exceptions=True)
    wall_elapsed = time.perf_counter() - wall_start

    await log(f"\n{'='*72}")
    await log(f"  RESULTS — per stream")
    await log(f"{'='*72}")

    header = (
        f"  {'Stream':<18} {'Audio':>6} {'Chunks':>6} "
        f"{'RTF':>6} {'Speed':>6} {'TTFT':>8} {'TTBL':>8} "
        f"{'p50':>8} {'p95':>8} {'avg':>8}"
    )
    await log(header)
    await log(f"  {'-'*78}")

    total_audio = 0
    valid_results = []
    for r in results:
        if isinstance(r, Exception):
            await log(f"  {'ERROR':<18} {str(r)[:60]}")
        elif r.error:
            await log(f"  {r.stream_id:<18} {r.audio_duration:>5.1f}s {r.chunk_count:>6} {'ERR':>6} {'--':>8} {'--':>8} {'--':>8} {'--':>8} {'--':>8} {r.error[:10]:>10}")
        else:
            valid_results.append(r)
            total_audio += r.audio_duration
            ttft = f"{r.ttft_ms}ms" if r.ttft_ms is not None else "None"
            ttbl = f"{r.ttbl_ms}ms" if r.ttbl_ms is not None else "None"
            rtf = f"{r.rtf:.2f}" if r.rtf is not None else "None"
            speed = f"{r.throughput:.2f}x" if r.throughput is not None else "None"
            p50 = f"{r.chunk_latency_p50}ms" if r.chunk_latency_p50 is not None else "None"
            p95 = f"{r.chunk_latency_p95}ms" if r.chunk_latency_p95 is not None else "None"
            avg = f"{r.chunk_latency_avg}ms" if r.chunk_latency_avg is not None else "None"
            text_len = len(r.final_text)
            await log(
                f"  {r.stream_id:<18} {r.audio_duration:>5.1f}s {r.chunk_count:>6} "
                f"{rtf:>6} {speed:>6} {ttft:>8} {ttbl:>8} "
                f"{p50:>8} {p95:>8} {avg:>8}"
            )

    if valid_results:
        rtfs = [r.rtf for r in valid_results if r.rtf is not None]
        ttfts = [r.ttft_ms for r in valid_results if r.ttft_ms is not None]
        ttbls = [r.ttbl_ms for r in valid_results if r.ttbl_ms is not None]
        p50s = [r.chunk_latency_p50 for r in valid_results if r.chunk_latency_p50 is not None]
        p95s = [r.chunk_latency_p95 for r in valid_results if r.chunk_latency_p95 is not None]

        overall_rtf = wall_elapsed / total_audio if total_audio > 0 else 0
        overall_speed = total_audio / wall_elapsed if wall_elapsed > 0 else 0
        await log(f"\n  {'Batch':<18} {'Audio':>6} {'Chunks':>8} {'RTF':>7} {'Speed':>7} {'TTFT':>8} {'TTBL':>8} {'p50':>8} {'p95':>8}")
        await log(f"  {'-'*74}")
        await log(
            f"  {'OVERALL':<18} {total_audio:>5.1f}s {sum(r.chunk_count for r in valid_results):>8} "
            f"{overall_rtf:>7.2f} {overall_speed:>7.2f}x {statistics.mean(ttfts):>7.0f}ms {statistics.mean(ttbls):>7.0f}ms "
            f"{statistics.mean(p50s):>7.1f}ms {statistics.mean(p95s):>7.1f}ms"
        )

    await log(f"\n  Wall time: {wall_elapsed:.2f}s")
    await log(f"  Streams: {len(results)} ({len(valid_results)} ok)")
    await log(f"{'='*72}\n")


def main():
    print("\n--- Per-Stream Metrics Demo ---\n")

    num_streams = int(input("Number of concurrent streams [1-32]: ") or "4")
    num_iterations = int(input("Iterations per stream [1-20]: ") or "2")
    chunk_ms = int(input("Chunk size in ms [80/160/320]: ") or "160")
    lang = input("Language (en/hi) [en]: ").strip() or "en"
    lang_code = "hi" if lang == "hi" else "auto"

    num_streams = max(1, min(32, num_streams))
    num_iterations = max(1, min(20, num_iterations))
    chunk_ms = max(80, min(1000, chunk_ms))

    server = "ws://localhost:8000/v1/stream"
    asyncio.run(run(num_streams, num_iterations, chunk_ms, lang_code, server))


if __name__ == "__main__":
    main()
