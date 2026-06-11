import asyncio
import base64
import io
import json
import os
import threading
import uuid
from typing import Generator

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from scipy.signal import resample_poly
from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput


TRITON_URL = os.environ.get("TRITON_URL", "localhost:8001")
MODEL_NAME = os.environ.get("MODEL_NAME", "nemotron_streaming")
DEFAULT_CHUNK_MS = int(os.environ.get("DEFAULT_CHUNK_MS", "320"))
TARGET_SR = 16000

app = FastAPI(title="Nemotron ASR Gateway")

_thread_local = threading.local()
CONCURRENCY = int(os.environ.get("TRITON_CLIENT_CONCURRENCY", "32"))


def _get_client() -> InferenceServerClient:
    client = getattr(_thread_local, "client", None)
    if client is None:
        client = InferenceServerClient(url=TRITON_URL, concurrency=CONCURRENCY)
        _thread_local.client = client
    return client


def _to_mono_float32(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    audio = audio.astype(np.float32)
    if sample_rate != TARGET_SR:
        audio = resample_poly(audio, TARGET_SR, sample_rate).astype(np.float32)
    return np.clip(audio, -1.0, 1.0)


def _iter_chunks(audio: np.ndarray, chunk_ms: int) -> Generator[np.ndarray, None, None]:
    chunk_samples = max(1, int(TARGET_SR * chunk_ms / 1000))
    for start in range(0, len(audio), chunk_samples):
        yield audio[start : start + chunk_samples]


async def _infer_chunk_async(
    stream_id: str,
    chunk: np.ndarray,
    target_lang: str,
    is_start: bool,
    is_end: bool,
) -> dict:
    infer_inputs = []

    audio_chunk = InferInput("AUDIO_CHUNK", [1, len(chunk)], "FP32")
    audio_chunk.set_data_from_numpy(chunk.reshape(1, -1).astype(np.float32))
    infer_inputs.append(audio_chunk)

    audio_length = InferInput("AUDIO_LENGTH", [1, 1], "INT32")
    audio_length.set_data_from_numpy(np.array([[len(chunk)]], dtype=np.int32))
    infer_inputs.append(audio_length)

    stream_id_tensor = InferInput("STREAM_ID", [1, 1], "BYTES")
    stream_id_tensor.set_data_from_numpy(np.array([[stream_id.encode("utf-8")]], dtype=object))
    infer_inputs.append(stream_id_tensor)

    lang_tensor = InferInput("TARGET_LANG", [1, 1], "BYTES")
    lang_tensor.set_data_from_numpy(np.array([[target_lang.encode("utf-8")]], dtype=object))
    infer_inputs.append(lang_tensor)

    start_tensor = InferInput("IS_START", [1, 1], "BOOL")
    start_tensor.set_data_from_numpy(np.array([[is_start]], dtype=bool))
    infer_inputs.append(start_tensor)

    end_tensor = InferInput("IS_END", [1, 1], "BOOL")
    end_tensor.set_data_from_numpy(np.array([[is_end]], dtype=bool))
    infer_inputs.append(end_tensor)

    outputs = [
        InferRequestedOutput("TRANSCRIPT"),
        InferRequestedOutput("IS_FINAL"),
        InferRequestedOutput("LANGUAGE"),
    ]

    def _sync_infer():
        return _get_client().infer(MODEL_NAME, infer_inputs, outputs=outputs)

    response = await asyncio.to_thread(_sync_infer)

    text = response.as_numpy("TRANSCRIPT").reshape(-1)[0]
    language = response.as_numpy("LANGUAGE").reshape(-1)[0]
    is_final_resp = bool(response.as_numpy("IS_FINAL").reshape(-1)[0])
    return {
        "stream_id": stream_id,
        "text": text.decode("utf-8") if isinstance(text, bytes) else str(text),
        "is_final": is_final_resp,
        "language": language.decode("utf-8") if isinstance(language, bytes) else str(language),
    }


AUDIO_DIR = os.environ.get("AUDIO_DIR", "/data/test_audio")

DASHBOARD_PATH = os.path.join(os.path.dirname(__file__), "dashboard.html")


@app.get("/", response_class=HTMLResponse)
def dashboard():
    try:
        with open(DASHBOARD_PATH, "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dashboard not found") from None


if os.path.isdir(AUDIO_DIR):
    app.mount("/test_audio", StaticFiles(directory=AUDIO_DIR), name="test_audio")


@app.get("/v1/files")
def list_files():
    result = {}
    if not os.path.isdir(AUDIO_DIR):
        return JSONResponse(result)
    for lang in sorted(os.listdir(AUDIO_DIR)):
        lang_dir = os.path.join(AUDIO_DIR, lang)
        if os.path.isdir(lang_dir):
            files = sorted(f for f in os.listdir(lang_dir) if f.endswith((".wav", ".flac", ".mp3", ".ogg")))
            result[lang] = files
    return JSONResponse(result)


@app.get("/healthz")
def healthz():
    try:
        return {"ok": _get_client().is_server_live()}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Triton unavailable: {exc}") from exc


@app.post("/v1/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    target_lang: str = Form("auto"),
    chunk_ms: int = Form(DEFAULT_CHUNK_MS),
):
    payload = await file.read()
    audio, sr = sf.read(io.BytesIO(payload), dtype="float32")
    audio = _to_mono_float32(audio, sr)
    stream_id = str(uuid.uuid4())
    last_result = None
    chunks = list(_iter_chunks(audio, chunk_ms))
    try:
        for index, chunk in enumerate(chunks):
            last_result = await _infer_chunk_async(
                stream_id=stream_id,
                chunk=chunk,
                target_lang=target_lang,
                is_start=index == 0,
                is_end=index == len(chunks) - 1,
            )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Inference backend unavailable: {exc}") from exc
    return JSONResponse(last_result or {"stream_id": stream_id, "text": "", "is_final": True, "language": target_lang})


@app.websocket("/v1/stream")
async def stream(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            raw_message = await websocket.receive_text()
            message = json.loads(raw_message)
            stream_id = message.get("stream_id") or str(uuid.uuid4())
            target_lang = message.get("target_lang", "auto")
            encoding = message.get("encoding", "pcm_s16le")
            sample_rate = int(message.get("sample_rate", TARGET_SR))
            audio_bytes = base64.b64decode(message["audio_b64"])

            if encoding == "pcm_s16le":
                audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            elif encoding == "float32le":
                audio = np.frombuffer(audio_bytes, dtype=np.float32)
            else:
                await websocket.send_json({"error": f"unsupported encoding: {encoding}"})
                continue

            audio = _to_mono_float32(audio, sample_rate)
            try:
                result = await _infer_chunk_async(
                    stream_id=stream_id,
                    chunk=audio,
                    target_lang=target_lang,
                    is_start=bool(message.get("is_start", False)),
                    is_end=bool(message.get("is_end", False)),
                )
                await websocket.send_json(result)
            except Exception as exc:
                await websocket.send_json({"error": f"inference backend unavailable: {exc}"})
    except WebSocketDisconnect:
        return
