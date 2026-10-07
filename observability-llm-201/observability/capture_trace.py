"""Captures an observability trace (latency, request_id, token preview) following a Bedrock call.

Parent lab: observability-llm-201.
AWS Services: Amazon Bedrock Runtime (converse).
Generated artifacts: observability/redacted-trace.json (résumé d'appel anonymisé).
Mode: live (un seul appel Bedrock).
"""
import json
import sys
from pathlib import Path
from time import perf_counter

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



client = boto3.client("bedrock-runtime", region_name="eu-west-1")


def main() -> None:
    # Measure end-to-end roundtrip latency for dashboard metrics.
    started_at = perf_counter()
    response = client.converse(
        modelId="amazon.nova-lite-v1:0",
        messages=[{"role": "user", "content": [{"text": "Resume la politique d'acces en 2 points."}]}],
    )
    # Minimal trace: AWS request ID, stop reason, and truncated content snippet.
    trace = {
        "latency_ms": round((perf_counter() - started_at) * 1000, 2),
        "request_id": response["ResponseMetadata"]["RequestId"],
        "stop_reason": response.get("stopReason"),
        "output_preview": response["output"]["message"]["content"][0]["text"][:120],
    }
    # Write execution trace for consumption by the training UI.
    Path("observability/redacted-trace.json").write_text(
        json.dumps(trace, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(trace, indent=2))


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
