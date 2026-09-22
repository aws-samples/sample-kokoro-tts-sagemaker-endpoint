"""Unit tests for the Amazon SageMaker AI Kokoro inference handler contract.

Validates input parsing, voice validation, and output encoding without
loading the model or requiring kokoro as a dependency.
"""

import io
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest
import soundfile as sf

sys.modules.setdefault("kokoro", MagicMock())

from src import inference  # noqa: E402

SAMPLE_RATE = 24_000


# --- input_fn -----------------------------------------------------------------


def test_input_fn_parses_json():
    got = inference.input_fn(
        b'{"text": "hi", "voice": "bf_emma", "speed": 1.1}', "application/json"
    )
    assert got == {"text": "hi", "voice": "bf_emma", "speed": 1.1}


def test_input_fn_accepts_str_body():
    got = inference.input_fn('{"text": "hi"}', "application/json")
    assert got == {"text": "hi"}


def test_input_fn_rejects_missing_text():
    with pytest.raises(ValueError, match="text"):
        inference.input_fn(b'{"voice": "af_bella"}', "application/json")


def test_input_fn_rejects_content_type():
    with pytest.raises(ValueError, match="content type"):
        inference.input_fn(b"hello", "text/plain")


# --- voice validation ---------------------------------------------------------


def test_input_fn_accepts_known_voice():
    got = inference.input_fn(b'{"text": "hi", "voice": "af_bella"}', "application/json")
    assert got == {"text": "hi", "voice": "af_bella"}


def test_input_fn_rejects_unknown_voice():
    with pytest.raises(ValueError, match="Unknown voice"):
        inference.input_fn(b'{"text": "hi", "voice": "xx_typo"}', "application/json")


def test_input_fn_allows_missing_voice_with_allowlist():
    got = inference.input_fn(b'{"text": "hi"}', "application/json")
    assert got == {"text": "hi"}


# --- output_fn ----------------------------------------------------------------


def test_output_fn_encodes_wav():
    audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
    wav_bytes, content_type = inference.output_fn(audio, "audio/wav")
    assert content_type == "audio/wav"
    data, sr = sf.read(io.BytesIO(wav_bytes))
    assert sr == SAMPLE_RATE
    assert len(data) == SAMPLE_RATE


@pytest.mark.parametrize("accept", ["audio/wav", "*/*", "application/octet-stream"])
def test_output_fn_accept_variants(accept):
    wav_bytes, content_type = inference.output_fn(
        np.zeros(10, dtype=np.float32), accept
    )
    assert content_type == "audio/wav"
    assert wav_bytes[:4] == b"RIFF"


def test_output_fn_rejects_accept():
    with pytest.raises(ValueError, match="accept type"):
        inference.output_fn(np.zeros(10, dtype=np.float32), "image/png")
