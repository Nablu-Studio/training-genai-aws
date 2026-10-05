"""Demonstrates a Bedrock converse call with a `create_ticket` tool and logs the raw response.

Parent lab: agents-tooluse-101.
AWS Services: Amazon Bedrock Runtime (converse + toolConfig).
Generated artifacts: agent/tooluse-response.json (réponse Bedrock complète).
Mode: live (un appel converse avec schéma d'input).
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
    "ResourceNotFoundException": "Ressource introuvable dans cette région : tous les modèles ne sont pas disponibles partout.",
    "ThrottlingException": "Quota exceeded: reduce call rate or retry with exponential backoff.",
    "ValidationException": "Request validation error: invalid model ID or model requires an inference profile./us./global.) plutôt que l'ID direct.",
}


def explain_aws_error(exc: ClientError) -> str:
    """Translates an AWS ClientError into an actionable diagnostic message for the learner."""
    error = exc.response["Error"]
    code = error.get("Code", "Unknown")
    hint = AWS_ERROR_HINTS.get(code, "Unexpected AWS error: read details below.")
    return f"[{code}] {hint}\nAWS Details: {error.get('Message', '')}"



client = boto3.client("bedrock-runtime", region_name="eu-west-1")


def main() -> None:
    # Tool unique `create_ticket` avec schéma JSON : title + severity obligatoires.
    response = client.converse(
        modelId="amazon.nova-lite-v1:0",
        messages=[{"role": "user", "content": [{"text": "Cree un ticket pour une regression de prompt."}]}],
        toolConfig={
            "tools": [
                {
                    "toolSpec": {
                        "name": "create_ticket",
                        "description": "Cree un ticket dans le systeme de support",
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {
                                    "title": {"type": "string"},
                                    "severity": {"type": "string"},
                                },
                                "required": ["title", "severity"],
                            }
                        },
                    }
                }
            ]
        },
    )
    # Sérialisation brute pour analyse offline par la UI.
    Path("agent/tooluse-response.json").write_text(
        json.dumps(response, indent=2, default=str),
        encoding="utf-8",
    )
    print(response.get("stopReason"))


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
