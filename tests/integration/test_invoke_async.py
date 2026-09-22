"""Integration test for the deployed Kokoro TTS Amazon SageMaker AI async endpoint.

Invokes the live endpoint and validates the response is a valid WAV file.
Run manually after deployment:

    uv run pytest tests/integration/ -v

To target a different stack name, set the STACK_NAME environment variable:

    STACK_NAME=my-stack uv run pytest tests/integration/ -v
"""

import io
import json
import time
from urllib.parse import urlparse

import boto3
import pytest
import soundfile as sf
from botocore.exceptions import ClientError

import os

REGION = "eu-west-2"
STACK_NAME = os.environ.get("STACK_NAME", "kokoro-async")
SAMPLE_RATE = 24_000


def stack_outputs(region: str, stack_name: str) -> dict[str, str]:
    cfn = boto3.client("cloudformation", region_name=region)
    stacks = cfn.describe_stacks(StackName=stack_name)["Stacks"]
    return {o["OutputKey"]: o["OutputValue"] for o in stacks[0].get("Outputs", [])}


@pytest.fixture(scope="module")
def endpoint():
    outputs = stack_outputs(REGION, STACK_NAME)
    return {
        "name": outputs["EndpointName"],
        "bucket": outputs["BucketName"],
    }


def invoke_and_wait(endpoint, text, voice="af_bella", speed=1.0, timeout=600) -> bytes:
    s3 = boto3.client("s3", region_name=REGION)
    runtime = boto3.client("sagemaker-runtime", region_name=REGION)

    payload = json.dumps({"text": text, "voice": voice, "speed": speed}).encode()

    resp = runtime.invoke_endpoint_async(
        EndpointName=endpoint["name"],
        Body=payload,
        ContentType="application/json",
        Accept="audio/wav",
        InvocationTimeoutSeconds=3600,
    )

    parsed = urlparse(resp["OutputLocation"])
    out_bucket, out_key = parsed.netloc, parsed.path.lstrip("/")

    parsed_fail = urlparse(resp["FailureLocation"])
    fail_bucket, fail_key = parsed_fail.netloc, parsed_fail.path.lstrip("/")

    deadline = time.time() + timeout
    while time.time() < deadline:
        # Check if the output WAV has landed in S3
        try:
            obj = s3.get_object(Bucket=out_bucket, Key=out_key)
            return obj["Body"].read()
        except ClientError as e:
            if e.response["Error"]["Code"] not in ("NoSuchKey", "404"):
                raise  # unexpected S3 error

        # Check if the endpoint wrote a failure record instead
        try:
            obj = s3.get_object(Bucket=fail_bucket, Key=fail_key)
            detail = obj["Body"].read().decode("utf-8", "replace")
            pytest.fail(f"Inference failed: {detail}")
        except ClientError as e:
            if e.response["Error"]["Code"] not in ("NoSuchKey", "404"):
                raise  # unexpected S3 error

        # Neither exists yet — inference is still processing
        time.sleep(5)

    pytest.fail(f"Timed out after {timeout}s waiting for result")


def test_endpoint_returns_valid_wav(endpoint):
    wav_bytes = invoke_and_wait(endpoint, "Hello from the integration test.")
    assert wav_bytes[:4] == b"RIFF"
    data, sr = sf.read(io.BytesIO(wav_bytes))
    assert sr == SAMPLE_RATE
    assert len(data) > SAMPLE_RATE * 0.5


def test_endpoint_rejects_invalid_voice(endpoint):
    # invoke_and_wait calls pytest.fail() when the failure S3 path appears,
    # which is the expected outcome here. Catch it and verify the message.
    with pytest.raises(pytest.fail.Exception, match="Unknown voice"):
        invoke_and_wait(endpoint, "This should fail.", voice="xx_invalid")
