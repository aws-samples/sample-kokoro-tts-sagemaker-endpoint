#!/usr/bin/env bash
# Build and push the Kokoro TTS inference image to Amazon ECR.
#
# Steps:
#   1. ensure the Amazon ECR repo exists in the target account/region
#   2. docker login to that Amazon ECR registry (12h STS token)
#   3. cross-build linux/amd64 from Apple Silicon via buildx, push to Amazon ECR
#
# Run from the sagemaker/ directory:
#   ./scripts/build_image.sh
#
# Override the tag with: TAG=2026-06-05 ./scripts/build_image.sh

set -euo pipefail

REGION="eu-west-2"
REPO="oss-tts/kokoro-inference"
TAG="${TAG:-latest}"

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"

REGISTRY="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
IMAGE_URI="${REGISTRY}/${REPO}:${TAG}"

# AWS-published Deep Learning Containers live in this account; pulls require ECR auth
DLC_REGISTRY="763104351884.dkr.ecr.${REGION}.amazonaws.com"

cd "$(dirname "$0")/.."

# shellcheck source-path=SCRIPTDIR source=../docker/voices.env
source docker/voices.env

echo "==> ensuring ECR repo ${REPO} exists in ${REGION}"
aws ecr describe-repositories \
    --repository-names "${REPO}" \
    --region "${REGION}" \
    >/dev/null 2>&1 \
  || aws ecr create-repository \
       --repository-name "${REPO}" \
       --region "${REGION}" \
       --image-scanning-configuration scanOnPush=true \
       >/dev/null

echo "==> logging in to ${DLC_REGISTRY} (for base image pull)"
aws ecr get-login-password --region "${REGION}" \
  | docker login --username AWS --password-stdin "${DLC_REGISTRY}"

echo "==> logging in to ${REGISTRY}"
aws ecr get-login-password --region "${REGION}" \
  | docker login --username AWS --password-stdin "${REGISTRY}"

echo "==> building and pushing ${IMAGE_URI}"
# --provenance=false: without it, buildx pushes an OCI image index
# (application/vnd.oci.image.index.v1+json) which SageMaker's image puller
# rejects. Disabling provenance yields a plain Docker v2 manifest.
docker buildx build \
    --platform linux/amd64 \
    --provenance=false \
    --file docker/Dockerfile \
    --build-arg "KOKORO_VOICES=${KOKORO_VOICES}" \
    --tag "${IMAGE_URI}" \
    --push \
    .

echo
echo "image: ${IMAGE_URI}"
