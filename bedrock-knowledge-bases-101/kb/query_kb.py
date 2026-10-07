"""Queries a Bedrock Knowledge Base (Retrieve API) with a metadata filter.

Parent lab: bedrock-knowledge-bases-101.
AWS Services: Amazon Bedrock Agent Runtime (retrieve + vectorSearchConfiguration).
Generated artifacts: kb/retrieve-response.json (réponse brute) + résumé console.
Mode: live (Retrieve consomme des crédits d'indexation et de recherche).
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
    "ResourceNotFoundException": "Agent, alias, or Knowledge Base not found: check identifier and region.",
    "ThrottlingException": "Quota exceeded: reduce call rate or retry with exponential backoff.",
    "ValidationException": "Request validation error: invalid model ID or model requires an inference profile ARN.",
}


def explain_aws_error(exc: ClientError) -> str:
    """Translates an AWS ClientError into an actionable diagnostic message for the learner."""
    error = exc.response["Error"]
    code = error.get("Code", "Unknown")
    hint = AWS_ERROR_HINTS.get(code, "Unexpected AWS error: read details below.")
    return f"[{code}] {hint}\nAWS Details: {error.get('Message', '')}"


AWS_REGION = "eu-west-1"
KB_ID = "KB_ID_PLACEHOLDER"


def main() -> None:
    # Retrieve knowledge chunks using metadata filter `sensitivity = internal`.
    client = boto3.client("bedrock-agent-runtime", region_name=AWS_REGION)
    response = client.retrieve(
        knowledgeBaseId=KB_ID,
        retrievalQuery={"text": "Quel est le mode de validation CLI proof ?"},
        retrievalConfiguration={
            "vectorSearchConfiguration": {
                "numberOfResults": 3,
                "filter": {
                    "equals": {
                        "key": "sensitivity",
                        "value": "internal",
                    }
                },
            }
        },
    )
    # Raw serialization for UI debugging.
    Path("kb/retrieve-response.json").write_text(
        json.dumps(response, indent=2, default=str),
        encoding="utf-8",
    )

    results = response.get("retrievalResults", [])
    first_result = results[0] if results else {}
    # Console summary: match count plus location of the top hit.
    print(
        json.dumps(
            {
                "matches": len(results),
                "first_location": first_result.get("location"),
            },
            indent=2,
            default=str,
        )
    )


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
