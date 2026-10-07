"""Demonstrates Bedrock function calling with a `lookup_genai_service` tool and validates the returned payload.

Parent lab: prompt-engineering-techniques-101.
AWS Services: Amazon Bedrock Runtime (converse + toolConfig).
Generated artifacts: prompting/audit/function-calling-audit.json (réponse, tool, validation).
Mode: live (un appel converse par exécution).
"""
import json
import os
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


AWS_REGION = os.environ.get("AWS_REGION", "eu-west-1")
MODEL_ID = os.environ.get("TRAINING_PROMPTING_MODEL", "amazon.nova-lite-v1:0")
OUTPUT_PATH = Path("prompting/audit/function-calling-audit.json")

TOOL_SPEC = {
    "toolSpec": {
        "name": "lookup_genai_service",
        "description": "Renvoie la fiche synthetique d un service AWS GenAI.",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "service": {
                        "type": "string",
                        "enum": ["bedrock", "sagemaker", "q", "kendra", "comprehend"],
                    }
                },
                "required": ["service"],
            }
        },
    }
}

ALLOWED_SERVICES = TOOL_SPEC["toolSpec"]["inputSchema"]["json"]["properties"]["service"]["enum"]

client = boto3.client("bedrock-runtime", region_name=AWS_REGION)


def validate_payload(payload: dict) -> dict:
    # Enum values defined once in TOOL_SPEC: reference it directly for validation.
    issues: list[str] = []
    service = payload.get("service")
    if not isinstance(service, str):
        issues.append("service manquant ou non string")
    elif service not in ALLOWED_SERVICES:
        issues.append(f"service '{service}' hors enum")
    return {"valid": not issues, "issues": issues}


def call_once(question: str) -> dict:
    response = client.converse(
        modelId=MODEL_ID,
        system=[{"text": "Tu es un assistant AWS GenAI expert. Utilise l outil fourni quand c est adapte."}],
        messages=[{"role": "user", "content": [{"text": question}]}],
        toolConfig={"tools": [TOOL_SPEC]},
    )
    output = response["output"]["message"]
    # Isolate only `toolUse` content blocks (remaining blocks contain freeform text).
    tool_calls = [
        block for block in output.get("content", []) if isinstance(block, dict) and "toolUse" in block
    ]
    if not tool_calls:
        return {
            "stopReason": response.get("stopReason"),
            "toolCalled": False,
            "answer": "".join(
                block.get("text", "") for block in output.get("content", []) if isinstance(block, dict)
            ).strip(),
        }
    use = tool_calls[0]["toolUse"]
    return {
        "stopReason": response.get("stopReason"),
        "toolCalled": True,
        "toolName": use.get("name"),
        "input": use.get("input"),
        "requestId": response["ResponseMetadata"]["RequestId"],
    }


def main() -> None:
    # Single demo query + audit log of invoked tool name and input parameter validity.
    question = "Donne moi la fiche du service Bedrock."
    result = call_once(question)
    validation = (
        validate_payload(result.get("input", {}))
        if result.get("toolCalled")
        else {"valid": False, "issues": ["no tool call"]}
    )
    audit = {
        "modelId": MODEL_ID,
        "region": AWS_REGION,
        "question": question,
        "toolName": TOOL_SPEC["toolSpec"]["name"],
        "toolCalled": result.get("toolCalled"),
        "stopReason": result.get("stopReason"),
        "input": result.get("input"),
        "answer": result.get("answer"),
        "validation": validation,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


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
