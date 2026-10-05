#!/usr/bin/env python3
"""Demonstrates trust boundary: assumes an IAM role with ExternalId, then performs a probe action.

Parent lab: iam-assume-role-101.
AWS Services: STS (assume_role) + configurable target service (bedrock or s3).
Generated artifacts: iam/assume-role-audit-log.json (identity, expiration, probe result).
Mode: live (assume_role + probe call to target service).
"""
import json
import os
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError, EndpointConnectionError, NoCredentialsError

# Distinguishable failure modes for the learner when an AWS call fails.
AWS_ERROR_HINTS = {
    "AccessDenied": "AssumeRole denied: role trust policy must allow this principal, and ExternalId must match.",
    "AccessDeniedException": "Insufficient permissions: check the IAM policy of the role or current identity.",
    "ThrottlingException": "Rate limit exceeded: reduce call frequency or retry with exponential backoff.",
    "ValidationError": "Invalid parameter: verify the role ARN format.",
}


def explain_aws_error(exc: ClientError) -> str:
    """Translates an AWS ClientError into an actionable diagnostic message for the learner."""
    error = exc.response["Error"]
    code = error.get("Code", "Unknown")
    hint = AWS_ERROR_HINTS.get(code, "Unexpected AWS error: read details below.")
    return f"[{code}] {hint}\nAWS Details: {error.get('Message', '')}"


ROLE_ARN = os.environ.get(
    "TRAINING_ASSUME_ROLE_ARN",
    "arn:aws:iam::123456789012:role/nablu-training-assume-role",
)
EXTERNAL_ID = os.environ.get("TRAINING_ASSUME_ROLE_EXTERNAL_ID", "training-check-123")
SESSION_NAME = "nablu-training-assume-role-check"
AWS_REGION = os.environ.get("AWS_REGION", "eu-west-1")
SERVICE_PROBE = os.environ.get("TRAINING_PROBE_SERVICE", "bedrock")
OUTPUT_PATH = Path("iam/assume-role-audit-log.json")


def assume_role() -> tuple[dict, dict]:
    # assume_role is the core API: capture a summary (never write credentials to disk).
    sts = boto3.client("sts", region_name=AWS_REGION)
    response = sts.assume_role(
        RoleArn=ROLE_ARN,
        RoleSessionName=SESSION_NAME,
        ExternalId=EXTERNAL_ID,
    )
    identity = {
        "assumedRoleId": response["AssumedRoleUser"]["AssumedRoleId"],
        "arn": response["AssumedRoleUser"]["Arn"],
        "credentialsExpiration": response["Credentials"]["Expiration"].isoformat(),
        "sessionTokenPrefix": response["Credentials"]["SessionToken"][:8],
    }
    return identity, response["Credentials"]


def probe_service(credentials: dict) -> dict:
    # Temporary credentials injected into an ephemeral Boto3 session.
    session = boto3.Session(
        aws_access_key_id=credentials["AccessKeyId"],
        aws_secret_access_key=credentials["SecretAccessKey"],
        aws_session_token=credentials["SessionToken"],
        region_name=AWS_REGION,
    )
    client = session.client(SERVICE_PROBE)
    if SERVICE_PROBE == "bedrock":
        # Default probe: list Nova models to verify Bedrock inference permissions.
        models = client.list_foundation_models(
            byOutputModality="TEXT",
            byInferenceType="ON_DEMAND",
        ).get("modelSummaries", [])
        return {
            "service": "bedrock",
            "ok": True,
            "novaModels": [m["modelId"] for m in models if "nova" in m["modelId"]][:5],
        }
    if SERVICE_PROBE == "s3":
        buckets = client.list_buckets().get("Buckets", [])
        return {
            "service": "s3",
            "ok": True,
            "buckets": [b["Name"] for b in buckets][:5],
        }
    return {"service": SERVICE_PROBE, "ok": True}


def main() -> None:
    # AssumeRole + probe: temporary credentials are never stored on disk.
    identity, credentials = assume_role()
    probe = probe_service(credentials)
    audit_log = {
        "roleArn": ROLE_ARN,
        "externalId": EXTERNAL_ID,
        "sessionName": SESSION_NAME,
        "region": AWS_REGION,
        "assumed": identity,
        "probe": probe,
        "result": "ok" if probe.get("ok") else "blocked",
    }
    OUTPUT_PATH.write_text(json.dumps(audit_log, indent=2), encoding="utf-8")
    print(json.dumps(audit_log, indent=2))


if __name__ == "__main__":
    try:
        main()
    except ClientError as exc:
        sys.exit(explain_aws_error(exc))
    except NoCredentialsError:
        sys.exit("No AWS credentials found: run `aws configure` or export AWS_PROFILE.")
    except EndpointConnectionError:
        sys.exit("AWS endpoint unreachable: verify AWS_REGION and network access.")
    except FileNotFoundError as exc:
        sys.exit(f"Expected file not found: {exc.filename}. Run the previous lab step first.")
