"""Generates a PNG image from a text prompt using Amazon Titan Image Generator.

Parent lab: multimodal-101.
AWS Services: Amazon Bedrock Runtime (Titan Image Generator v1).
Generated artifacts: multimodal/output/generated.png + multimodal/audit/image-gen-audit.json.
Mode: live (image générée facturée).
"""
import base64
import json
import os
import sys
import time
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
IMAGE_GEN_MODEL = os.environ.get("TRAINING_IMAGE_GEN_MODEL", "amazon.titan-image-generator-v1")
PROMPT = os.environ.get(
    "TRAINING_IMAGE_PROMPT",
    "Diagramme d architecture Bedrock, low poly, style blueprint",
)
AUDIT_PATH = Path("multimodal/audit/image-gen-audit.json")
GEN_OUTPUT = Path("multimodal/output/generated.png")


def generate_image(prompt: str) -> dict:
    # invoke_model with taskType=TEXT_IMAGE: conforms to Titan Image Generator schema.
    bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    started = time.perf_counter()
    response = bedrock.invoke_model(
        modelId=IMAGE_GEN_MODEL,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(
            {
                "taskType": "TEXT_IMAGE",
                "textToImageParams": {"text": prompt},
                "imageGenerationConfig": {
                    "numberOfImages": 1,
                    "height": 512,
                    "width": 512,
                    "cfgWeight": 8.0,
                },
            }
        ),
    )
    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    payload = json.loads(response["body"].read())
    # Titan returns base64: decode bytes and write the PNG image to disk.
    image_bytes = base64.b64decode(payload["images"][0])
    GEN_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    GEN_OUTPUT.write_bytes(image_bytes)
    return {
        "modelId": IMAGE_GEN_MODEL,
        "prompt": prompt,
        "latencyMs": latency_ms,
        "bytesWritten": len(image_bytes),
        "requestId": response["ResponseMetadata"]["RequestId"],
    }


def main() -> None:
    # Single execution: outputs JSON audit log plus binary image for the UI.
    audit = generate_image(PROMPT)
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.write_text(json.dumps(audit, indent=2), encoding="utf-8")
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
