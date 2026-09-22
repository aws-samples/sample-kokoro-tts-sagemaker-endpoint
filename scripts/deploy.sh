#!/usr/bin/env bash
# Deploy the Kokoro TTS async endpoint via AWS CloudFormation.
#
# Assumes the image has already been built and pushed (see build_image.sh) and
# that AWS credentials are in the environment (e.g. `source ../.env`).
#
# Run from the sagemaker/ directory:
#   ./scripts/deploy.sh
#
# Override the image tag with: TAG=2026-06-05 ./scripts/deploy.sh

set -euo pipefail

REGION="eu-west-2"
STACK_NAME="${STACK_NAME:-kokoro-async}"
TAG="${TAG:-latest}"

cd "$(dirname "$0")/.."

echo "==> deploying stack ${STACK_NAME} to ${REGION} (image tag: ${TAG})"
aws cloudformation deploy \
    --region "${REGION}" \
    --stack-name "${STACK_NAME}" \
    --template-file template.yaml \
    --capabilities CAPABILITY_IAM \
    --parameter-overrides "ImageTag=${TAG}"

echo
echo "==> stack outputs"
aws cloudformation describe-stacks \
    --region "${REGION}" \
    --stack-name "${STACK_NAME}" \
    --query "Stacks[0].Outputs[].{Key:OutputKey,Value:OutputValue}" \
    --output table
