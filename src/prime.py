"""Pre-warm Kokoro caches at image build time.

Runs once during `docker build` to download Kokoro-82M weights, the spaCy
transformer state, and all English voice tensors into the HF cache baked into
the image. After build, runtime never needs network access — `HF_HUB_OFFLINE=1`
is set in the Dockerfile to fail loudly on any accidental download attempt.

The voice list is read from the KOKORO_VOICES env var, set in the Dockerfile so
it is the single source of truth shared with the runtime handler (inference.py).
"""

import os

from kokoro import KPipeline

VOICES = [
    v.strip() for v in os.environ.get("KOKORO_VOICES", "").split(",") if v.strip()
]


def main() -> None:
    if not VOICES:
        raise SystemExit("KOKORO_VOICES is empty; set it in the Dockerfile")
    pipeline = KPipeline(lang_code="b", trf=True)
    for voice in VOICES:
        print(f"warming voice: {voice}")
        pipeline.load_voice(voice)
    print(f"primed {len(VOICES)} voices")


if __name__ == "__main__":
    main()
