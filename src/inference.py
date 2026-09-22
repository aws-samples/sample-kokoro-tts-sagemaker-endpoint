"""Amazon SageMaker AI inference handler for Kokoro TTS (PyTorch backend)."""

from __future__ import annotations

import io
import json
import os
from typing import Any

import numpy as np
import soundfile as sf
from kokoro import KPipeline

SAMPLE_RATE = 24_000
DEFAULT_LANG_CODE = "b"
DEFAULT_VOICE = "af_bella"
DEFAULT_SPEED = 1.0
DEFAULT_SPLIT_PATTERN = r"\n\n+"
USE_TRF = True

# Voices baked into the image, read from the KOKORO_VOICES env var that the
# Dockerfile sets (the single source of truth, also consumed by prime.py at
# build time). Empty when unset (e.g. unit tests), in which case voice
# validation is skipped.
ALLOWED_VOICES = frozenset(
    v.strip() for v in os.environ.get("KOKORO_VOICES", "").split(",") if v.strip()
)


def model_fn(_model_dir: str) -> KPipeline:
    """Load the Kokoro pipeline once at container startup.

    Return the pipeline directly because Kokoro weights
    and the spaCy transformer are baked into the image.
    """
    return KPipeline(lang_code=DEFAULT_LANG_CODE, trf=USE_TRF)


def input_fn(request_body: bytes | str, content_type: str) -> dict[str, Any]:
    """Parse the incoming request into a dict of pipeline kwargs.

    Accepts a single content type, `application/json`, with shape:
    {"text": str, "voice"?: str, "speed"?: float, "split_pattern"?: str}
    """
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type: {content_type}")
    body = (
        request_body.decode("utf-8")
        if isinstance(request_body, bytes)
        else request_body
    )
    payload = json.loads(body)
    if "text" not in payload:
        raise ValueError("JSON payload must include a 'text' field")
    voice = payload.get("voice")
    if voice is not None and ALLOWED_VOICES and voice not in ALLOWED_VOICES:
        raise ValueError(
            f"Unknown voice {voice!r}; expected one of: "
            f"{', '.join(sorted(ALLOWED_VOICES))}"
        )
    return payload


def predict_fn(inputs: dict[str, Any], pipeline: KPipeline) -> np.ndarray:
    """Run the Kokoro pipeline and concatenate streamed audio chunks.

    The pipeline yields `(graphemes, phonemes, audio)` tuples per chunk; we keep
    only the audio. Returns an empty float32 array when the pipeline produces
    no chunks (e.g. empty or whitespace-only input).
    """
    text = inputs["text"]
    voice = inputs.get("voice", DEFAULT_VOICE)
    speed = float(inputs.get("speed", DEFAULT_SPEED))
    split_pattern = inputs.get("split_pattern", DEFAULT_SPLIT_PATTERN)

    audio_parts: list[np.ndarray] = []
    for _gs, _ps, audio in pipeline(
        text, voice=voice, speed=speed, split_pattern=split_pattern
    ):
        if audio is None:
            continue
        audio_parts.append(np.asarray(audio))
    if not audio_parts:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(audio_parts)


def output_fn(prediction: np.ndarray, accept: str) -> tuple[bytes, str]:
    """Encode the audio array as a 24 kHz WAV byte stream.

    SageMaker async writes whatever bytes we return to the configured S3
    output location. `accept` is honoured loosely: `audio/wav`, `*/*`, and
    `application/octet-stream` all yield WAV.
    """
    if accept not in ("audio/wav", "*/*", "application/octet-stream"):
        raise ValueError(f"Unsupported accept type: {accept}")
    buf = io.BytesIO()
    sf.write(buf, prediction, SAMPLE_RATE, format="WAV")
    return buf.getvalue(), "audio/wav"
