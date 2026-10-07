"""Tests the Bedrock Guardrail using sample prompt payloads and logs the policy verdict.

Parent lab: bedrock-guardrails-101.
AWS Services: Amazon Bedrock Runtime (converse + guardrailConfig).
Generated artifacts: guardrails/guardrail-audit-log.json (risk, control, prompt, stopReason).
Mode: live (consomme des tokens et applique le guardrail).
"""
import json
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError, EndpointConnectionError, NoCredentialsError

# Distinguishable failure modes when the call fails.
AWS_ERROR_HINTS = {
    "AccessDeniedException": "Access denied: either the IAM policy lacks bedrock:InvokeModel, or model access is not enabled (Bedrock console > Model access).",
    "ModelTimeoutException": "Model timed out: reduce prompt size or max tokens.",
    "ResourceNotFoundException": "Resource not found in this region: not all models are available in all regions.",
    "ThrottlingException": "Quota exceeded: reduce call rate or retry with exponential backoff.",
    "ValidationException": "Request validation error: invalid model ID or model requires an inference profile ARN.",
}


def explain_aws_error(exc: ClientError) -> str:
    """Translates an AWS ClientError into an actionable diagnostic message for the learner."""
    error = exc.response["Error"]
    code = error.get("Code", "Unknown")
    hint = AWS_ERROR_HINTS.get(code, "Unexpected AWS error: read details below.")
    return f"[{code}] {hint}\nAWS Details: {error.get('Message', '')}"


RUNTIME_CONFIG_PATH = Path("guardrails/guardrail-runtime-config.json")


def load_runtime_config() -> dict:
    return json.loads(RUNTIME_CONFIG_PATH.read_text(encoding="utf-8"))


def main() -> None:
    # Read configuration exported by `create_guardrail.py` (build vs run separation).
    runtime_config = load_runtime_config()
    active_control = runtime_config["activeControl"]
    client = boto3.client(
        "bedrock-runtime",
        region_name=runtime_config["region"],
    )
    # Converse API call attaches guardrailConfig to enforce version pinning.
    response = client.converse(
        modelId=runtime_config["modelId"],
        messages=[
            {
                "role": "user",
                "content": [{"text": runtime_config["testPrompt"]}],
            }
        ],
        guardrailConfig={
            "guardrailIdentifier": runtime_config["guardrailIdentifier"],
            "guardrailVersion": runtime_config["guardrailVersion"],
        },
    )

    # Audit: record risk, control, and proof for compliance matrix.
    audit_log = {
        "risk": active_control["risk"],
        "control": active_control["control"],
        "controlOwner": active_control["owner"],
        "proofExpectation": active_control["proof"],
        "guardrailIdentifier": runtime_config["guardrailIdentifier"],
        "guardrailVersion": runtime_config["guardrailVersion"],
        "stopReason": response.get("stopReason"),
        "requestId": response["ResponseMetadata"]["RequestId"],
        "prompt": runtime_config["testPrompt"],
    }
    Path("guardrails/guardrail-audit-log.json").write_text(
        json.dumps(audit_log, indent=2),
        encoding="utf-8",
    )
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
