#!/usr/bin/env bash
# Tear down the Kokoro async endpoint stack.
#
# AWS CloudFormation cannot delete a non-empty S3 bucket, so we empty both buckets
# first, then delete the stack. Versioning is enabled, so we delete all object versions.

set -euo pipefail

REGION="eu-west-2"
STACK_NAME="${STACK_NAME:-kokoro-async}"

empty_bucket() {
    local bucket="$1"
    local delete_payload
    local object_count

    if [[ -n "${bucket}" && "${bucket}" != "None" ]]; then
        echo "==> emptying s3://${bucket} (including all versions)"
        while true; do
            object_count="$(aws s3api list-object-versions \
                --region "${REGION}" \
                --bucket "${bucket}" \
                --max-keys 1000 \
                --query 'length([Versions, DeleteMarkers][][])' \
                --output text)"

            if (( object_count == 0 )); then
                break
            fi

            # Backticks below delimit a JMESPath boolean literal.
            # shellcheck disable=SC2016
            delete_payload="$(aws s3api list-object-versions \
                --region "${REGION}" \
                --bucket "${bucket}" \
                --max-keys 1000 \
                --query '{Objects: [Versions, DeleteMarkers][][].{Key:Key,VersionId:VersionId}, Quiet: `true`}' \
                --output json)"

            aws s3api delete-objects \
                --region "${REGION}" \
                --bucket "${bucket}" \
                --delete "${delete_payload}" \
                >/dev/null
        done
    fi
}

echo "==> looking up buckets from stack ${STACK_NAME}"
BUCKET="$(aws cloudformation describe-stacks \
    --region "${REGION}" \
    --stack-name "${STACK_NAME}" \
    --query "Stacks[0].Outputs[?OutputKey=='BucketName'].OutputValue" \
    --output text 2>/dev/null || true)"

LOGS_BUCKET="$(aws cloudformation describe-stacks \
    --region "${REGION}" \
    --stack-name "${STACK_NAME}" \
    --query "Stacks[0].Outputs[?OutputKey=='LogsBucketName'].OutputValue" \
    --output text 2>/dev/null || true)"

empty_bucket "${BUCKET}"
empty_bucket "${LOGS_BUCKET}"

echo "==> deleting stack ${STACK_NAME}"
aws cloudformation delete-stack --region "${REGION}" --stack-name "${STACK_NAME}"

echo "==> waiting for delete to complete..."
aws cloudformation wait stack-delete-complete --region "${REGION}" --stack-name "${STACK_NAME}"

echo "done: stack ${STACK_NAME} deleted"
