"""Queries Amazon Q Business chat API using applicationId and userId.

Parent lab: amazon-q-101.
AWS Services: Amazon Q Business (chat).
Generated artifacts: q/audit/qbusiness-query-audit.json (réponse, systemMessage, latence).
Mode: live si APPLICATION_ID/USER_ID fournis, sinon audit "skipped" sans appel.
"""
import json
import os
import sys
import time
from pathlib import Path

import boto3
from botocore.exceptions import ClientError, EndpointConnectionError, NoCredentialsError

# Distinguishable failure modes when the call fails.
AWS_ERROR_HINTS = {
    "AccessDeniedException": "Insufficient permissions: check the IAM policy of the role or current identity.",
    "ResourceNotFoundException": "Q Business application not found: verify TRAINING_QBIZ_APPLICATION_ID and region.",
    "ThrottlingException": "Quota exceeded: reduce call rate or retry with exponential backoff.",
    "ValidationException": "Request validation error: verify parameter format in details below.",
}


def explain_aws_error(exc: ClientError) -> str:
    """Translates an AWS ClientError into an actionable diagnostic message for the learner."""
    error = exc.response["Error"]
    code = error.get("Code", "Unknown")
    hint = AWS_ERROR_HINTS.get(code, "Unexpected AWS error: read details below.")
    return f"[{code}] {hint}\nAWS Details: {error.get('Message', '')}"


AWS_REGION = os.environ.get("AWS_REGION", "eu-west-1")
APPLICATION_ID = os.environ.get("TRAINING_QBIZ_APPLICATION_ID", "REPLACE_WITH_QBIZ_APP_ID")
USER_ID = os.environ.get("TRAINING_QBIZ_USER_ID", "REPLACE_WITH_QBIZ_USER_ID")
QUESTION = os.environ.get("TRAINING_QBIZ_QUESTION", "Quel est le delai de carence ?")
OUTPUT_PATH = Path("q/audit/qbusiness-query-audit.json")


def call_qbusiness() -> dict:
    # Q Business s'appelle via l'API `chat` (≠ `qbusiness:chat` côté IAM).
    client = boto3.client("qbusiness", region_name=AWS_REGION)
    started = time.perf_counter()
    response = client.chat(
        applicationId=APPLICATION_ID,
        userId=USER_ID,
        userMessage=QUESTION,
    )
    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    return {
        "applicationId": APPLICATION_ID,
        "userId": USER_ID,
        "question": QUESTION,
        "systemMessage": response.get("systemMessage", ""),
        "latencyMs": latency_ms,
        "requestId": response["ResponseMetadata"]["RequestId"],
        # `skipped` status allows the UI to display a "provision required" state without crashing.
        "skipped": APPLICATION_ID.startswith("REPLACE_") or USER_ID.startswith("REPLACE_"),
    }


def main() -> None:
    # If the application is not configured, attach an educational note to the audit artifact.
    result = call_qbusiness()
    if result.get("skipped"):
        result["note"] = (
            "Aucun APPLICATION_ID fourni: configuration requise (IAM Identity Center + create_application)."
        )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


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
