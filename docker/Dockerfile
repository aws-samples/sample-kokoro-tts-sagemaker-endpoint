# Extended SageMaker PyTorch inference DLC for Kokoro TTS.
#
# Inherits TorchServe + SageMaker inference toolkit + CUDA stack from the base
# image, then layers in:
#   - Kokoro and its transitive misaki/spaCy deps
#   - the en_core_web_trf spaCy transformer (for misaki trf=True)
#   - Kokoro-82M weights and the English voice tensors, baked into the HF cache
#
# Runtime is fully offline: HF_HUB_OFFLINE=1 makes any accidental download fail
# loudly rather than silently re-fetching at request time.

FROM 763104351884.dkr.ecr.eu-west-2.amazonaws.com/pytorch-inference:2.6.0-gpu-py312-cu124-ubuntu22.04-sagemaker

ENV HF_HOME=/opt/ml/hf-cache

COPY docker/requirements.txt /opt/ml/requirements.txt
RUN pip install --no-cache-dir -r /opt/ml/requirements.txt

ARG KOKORO_VOICES
ENV KOKORO_VOICES=${KOKORO_VOICES}

COPY src/prime.py /opt/ml/prime.py
RUN python /opt/ml/prime.py

# Bake the inference handler into the image. With no model.tar.gz the toolkit
# won't auto-extract a code/ dir, so we put /opt/ml/code on PYTHONPATH directly:
# the serving toolkit resolves the handler via find_spec(SAGEMAKER_PROGRAM
# without the .py), which honours PYTHONPATH. Verified that find_spec("inference")
# resolves and all four handler functions bind.
COPY src/inference.py /opt/ml/code/inference.py
ENV SAGEMAKER_PROGRAM=inference.py
ENV PYTHONPATH=/opt/ml/code

# Raise TorchServe's request/response frame limits and the model-server timeout.
# Defaults (~6.5 MB frame, 60s timeout) are far below this workload: text expands
# to audio at roughly 3,000:1, so even a modest input produces tens of MB of WAV,
# which trips "io.netty.handler.codec.TooLongFrameException" and terminates the worker.
# With enable_envvars_config=true, TorchServe maps TS_MAX_*_SIZE -> max_*_size.
#
# Sizing (bytes; 24 kHz mono PCM-16 WAV = 48 KB per audio-second):
#   - request  16 MB : text is tiny and never binds.
#   - response 512 MB: the binding limit. Caps output at ~2.96 hr of audio, which
#                      corresponds to ~165k input characters.
#   - timeout  3600s : SAGEMAKER_MODEL_SERVER_TIMEOUT (TorchServe
#                      default_response_timeout). The 60s default terminates longer
#                      inferences; 3600s matches the async InvocationTimeoutSeconds
#                      max so the response cap stays the binding constraint.
ENV TS_MAX_REQUEST_SIZE=16777216
ENV TS_MAX_RESPONSE_SIZE=536870912
ENV SAGEMAKER_MODEL_SERVER_TIMEOUT=3600

ENV HF_HUB_OFFLINE=1
