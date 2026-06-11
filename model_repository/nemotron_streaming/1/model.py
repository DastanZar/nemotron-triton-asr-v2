import ast
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import triton_python_backend_utils as pb_utils

from nemo.collections.asr.parts.utils.transcribe_utils import setup_model


LOGGER = logging.getLogger("nemotron_streaming")


def _as_bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "y", "on"}


def _decode_scalar_string(tensor) -> str:
    value = tensor.as_numpy().reshape(-1)[0]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _decode_scalar_bool(tensor) -> bool:
    value = tensor.as_numpy().reshape(-1)[0]
    return bool(value)


@dataclass
class StreamState:
    stream_id: str
    target_lang: str
    cache_last_channel: torch.Tensor | None = None
    cache_last_time: torch.Tensor | None = None
    cache_last_channel_len: torch.Tensor | None = None
    previous_hypotheses: Any = None
    previous_pred_out: Any = None
    chunk_count: int = 0
    last_text: str = ""
    updated_at: float = 0.0


class TritonPythonModel:
    def initialize(self, args):
        if isinstance(args, str):
            args = json.loads(args)
        model_config = args["model_config"]
        if isinstance(model_config, str):
            model_config = json.loads(model_config)
        params = model_config.get("parameters", {})

        self.model_name = self._parameter(params, "MODEL_NAME", "nvidia/nemotron-3.5-asr-streaming-0.6b")
        self.default_target_lang = self._parameter(params, "TARGET_LANG_DEFAULT", "auto")
        self.att_context_size = ast.literal_eval(self._parameter(params, "ATT_CONTEXT_SIZE", "[56,3]"))
        self.strip_lang_tags = _as_bool(self._parameter(params, "STRIP_LANG_TAGS", "true"))
        self.max_session_idle_sec = int(self._parameter(params, "MAX_SESSION_IDLE_SEC", "900"))

        logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        LOGGER.info("Loading model %s on %s", self.model_name, self.device)

        cfg = type("Cfg", (), {"model_path": None, "pretrained_name": self.model_name})()
        self.asr_model, _ = setup_model(cfg=cfg, map_location=self.device)
        if hasattr(self.asr_model.encoder, "set_default_att_context_size"):
            self.asr_model.encoder.set_default_att_context_size(att_context_size=self.att_context_size)
        if hasattr(self.asr_model, "set_inference_prompt"):
            self.asr_model.set_inference_prompt(self.default_target_lang)
            self.asr_model.decoding.set_strip_lang_tags(self.strip_lang_tags)

        if hasattr(self.asr_model, "joint"):
            from nemo.collections.asr.parts.submodules.rnnt_decoding import (
                RNNTDecodingConfig,
            )
            dec_cfg = RNNTDecodingConfig(fused_batch_size=-1)
            dec_cfg.greedy.use_cuda_graph_decoder = False
            self.asr_model.change_decoding_strategy(dec_cfg)

        self.asr_model = self.asr_model.to(device=self.device, dtype=torch.float32)
        self.asr_model.eval()

        self.drop_extra = self.asr_model.encoder.streaming_cfg.drop_extra_pre_encoded
        LOGGER.info("Model loaded on %s (drop_extra=%d)", self.device, self.drop_extra)

        self.sessions: dict[str, StreamState] = {}

    def execute(self, requests):
        self._expire_idle_sessions()
        responses = []
        grouped = {}

        for index, request in enumerate(requests):
            stream_id = _decode_scalar_string(
                pb_utils.get_input_tensor_by_name(request, "STREAM_ID")
            )
            target_lang = _decode_scalar_string(
                pb_utils.get_input_tensor_by_name(request, "TARGET_LANG")
            ) or self.default_target_lang
            is_end = _decode_scalar_bool(
                pb_utils.get_input_tensor_by_name(request, "IS_END")
            )
            group_key = (target_lang, is_end)
            grouped.setdefault(group_key, []).append(
                (index, request, stream_id, target_lang, is_end)
            )

        output_map = {}
        for (target_lang, is_end), items in grouped.items():
            batch_result = self._run_group(
                items=items, target_lang=target_lang, is_end=is_end
            )
            output_map.update(batch_result)

        for index in range(len(requests)):
            transcript, is_final, language = output_map[index]
            outputs = [
                pb_utils.Tensor(
                    "TRANSCRIPT", np.array([transcript.encode("utf-8")], dtype=object)
                ),
                pb_utils.Tensor("IS_FINAL", np.array([is_final], dtype=bool)),
                pb_utils.Tensor(
                    "LANGUAGE",
                    np.array([language.encode("utf-8")], dtype=object),
                ),
            ]
            responses.append(pb_utils.InferenceResponse(output_tensors=outputs))

        return responses

    def finalize(self):
        self.sessions.clear()

    def _run_group(self, items, target_lang: str, is_end: bool):
        if hasattr(self.asr_model, "set_inference_prompt"):
            self.asr_model.set_inference_prompt(target_lang)
            self.asr_model.decoding.set_strip_lang_tags(self.strip_lang_tags)

        for _, request, stream_id, _, _ in items:
            is_start = _decode_scalar_bool(
                pb_utils.get_input_tensor_by_name(request, "IS_START")
            )
            if is_start or stream_id not in self.sessions:
                self.sessions[stream_id] = self._new_state(
                    stream_id=stream_id, target_lang=target_lang
                )

        states = []
        for _, _, stream_id, _, _ in items:
            states.append(self.sessions[stream_id])

        batch_audio = []
        for _, request, stream_id, _, _ in items:
            chunk = (
                pb_utils.get_input_tensor_by_name(request, "AUDIO_CHUNK")
                .as_numpy()
                .astype(np.float32)
                .reshape(-1)
            )
            length = int(
                pb_utils.get_input_tensor_by_name(request, "AUDIO_LENGTH")
                .as_numpy()
                .reshape(-1)[0]
            )
            batch_audio.append(chunk[:length])

        with torch.inference_mode(), torch.no_grad():
            max_samples = max(len(a) for a in batch_audio)
            batch_size = len(batch_audio)
            stacked = torch.zeros(
                batch_size, max_samples, device=self.device, dtype=torch.float32
            )
            audio_lengths = torch.zeros(
                batch_size, dtype=torch.long, device=self.device
            )
            for i, audio in enumerate(batch_audio):
                stacked[i, : len(audio)] = torch.from_numpy(audio).to(self.device)
                audio_lengths[i] = len(audio)

            processed, processed_lengths = self.asr_model.preprocessor(
                input_signal=stacked, length=audio_lengths
            )

            LOGGER.info(
                "preprocessor output: shape=%s, lengths=%s",
                tuple(processed.shape), [pl.item() for pl in processed_lengths]
            )

            max_time = processed.size(-1)
            feat_dim = processed.size(-2)

            cache_channel = torch.cat(
                [s.cache_last_channel for s in states if s.cache_last_channel is not None],
                dim=1,
            )
            cache_time = torch.cat(
                [s.cache_last_time for s in states if s.cache_last_time is not None],
                dim=1,
            )
            cache_channel_len = torch.cat(
                [
                    s.cache_last_channel_len
                    for s in states
                    if s.cache_last_channel_len is not None
                ],
                dim=0,
            )

            prev_hyps = [s.previous_hypotheses for s in states]
            is_first = all(s.chunk_count == 0 for s in states)
            is_last = all(_decode_scalar_bool(
                pb_utils.get_input_tensor_by_name(request, "IS_END")
            ) for _, request, _, _, _ in items)

            min_frames_for_attention = 20
            pl_vals = [pl.item() for pl in processed_lengths]
            # Don't apply drop_extra if: first chunk, last chunk, or too few frames for attention
            apply_drop_extra = (not is_first and not is_last and 
                               all(pl >= self.drop_extra + min_frames_for_attention for pl in pl_vals))
            use_drop_extra = self.drop_extra if apply_drop_extra else 0
            if not apply_drop_extra and not is_first:
                LOGGER.info(
                    "Skipping drop_extra=%d: is_first=%s, is_last=%s, processed_lengths=%s, min_needed=%d",
                    self.drop_extra, is_first, is_last, pl_vals, min_frames_for_attention
                )

            LOGGER.info(
                "is_first=%s, is_last=%s, drop_extra=%d, processed_lengths=%s, use_drop_extra=%d",
                is_first, is_last, self.drop_extra, pl_vals, use_drop_extra
            )

            pred_out, transcribed_texts, new_ch, new_tm, new_cl, new_hyps = (
                self.asr_model.conformer_stream_step(
                    processed_signal=processed,
                    processed_signal_length=processed_lengths,
                    cache_last_channel=cache_channel,
                    cache_last_time=cache_time,
                    cache_last_channel_len=cache_channel_len,
                    keep_all_outputs=True,
                    previous_hypotheses=prev_hyps,
                    previous_pred_out=states[0].previous_pred_out
                    if states[0].previous_pred_out is not None
                    else None,
                    drop_extra_pre_encoded=use_drop_extra,
                    return_transcription=True,
                )
            )

        for i, state in enumerate(states):
            state.cache_last_channel = new_ch[:, i : i + 1, :]
            state.cache_last_time = new_tm[:, i : i + 1, :]
            state.cache_last_channel_len = new_cl[i : i + 1]
            state.previous_hypotheses = (
                new_hyps[i] if i < len(new_hyps) else None
            )
            state.previous_pred_out = pred_out

            if isinstance(transcribed_texts, (list, tuple)) and i < len(
                transcribed_texts
            ):
                hyp = transcribed_texts[i]
                text = hyp.text if hasattr(hyp, "text") else str(hyp)
                state.last_text = text

            state.chunk_count += 1

        results = {}
        for idx, (request_index, _, stream_id, _, _) in enumerate(items):
            state = self.sessions[stream_id]
            results[request_index] = (state.last_text, is_end, target_lang)
            if is_end:
                self.sessions.pop(stream_id, None)
        return results

    def _new_state(self, stream_id: str, target_lang: str) -> StreamState:
        cache_last_channel, cache_last_time, cache_last_channel_len = (
            self.asr_model.encoder.get_initial_cache_state(batch_size=1)
        )
        return StreamState(
            stream_id=stream_id,
            target_lang=target_lang,
            cache_last_channel=cache_last_channel.to(self.device),
            cache_last_time=cache_last_time.to(self.device),
            cache_last_channel_len=cache_last_channel_len.to(self.device),
            updated_at=time.time(),
        )

    def _expire_idle_sessions(self):
        now = time.time()
        stale_ids = [
            stream_id
            for stream_id, state in self.sessions.items()
            if now - state.updated_at > self.max_session_idle_sec
        ]
        for stream_id in stale_ids:
            self.sessions.pop(stream_id, None)

    @staticmethod
    def _parameter(params, key: str, default: str) -> str:
        value = params.get(key)
        if not value:
            return default
        return value["string_value"]
