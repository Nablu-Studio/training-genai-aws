"""Prompts a Bedrock multimodal vision model to describe a PNG image.

Parent lab: multimodal-101.
AWS Services: Amazon Bedrock Runtime (converse multimodal).
Generated artifacts: multimodal/audit/vision-audit.json (description, latence, requestId).
Mode: live (appel facturé sur l'image fournie).
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


AWS_REGION = os.environ.get("AWS_REGION", "eu-west-1")
VISION_MODEL = os.environ.get("TRAINING_VISION_MODEL", "amazon.nova-lite-v1:0")
IMAGE_PATH = os.environ.get("TRAINING_MULTIMODAL_IMAGE_PATH", "multimodal/data/sample-architecture.png")
OUTPUT_PATH = Path("multimodal/audit/vision-audit.json")


def describe_image(image_path: Path) -> dict:
    # L'image est transmise en base64 dans le bloc `image` du payload converse.
    runtime = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    started = time.perf_counter()
    response = runtime.converse(
        modelId=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"image": {"format": "png", "source": {"bytes": image_b64}}},
                    {"text": "Decris l architecture en 3 phrases."},
                ],
            }
        ],
        inferenceConfig={"temperature": 0.0, "maxTokens": 256},
    )
    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    # Concaténation défensive des blocs texte retournés.
    description = "".join(
        block.get("text", "")
        for block in response["output"]["message"].get("content", [])
    ).strip()
    return {
        "modelId": VISION_MODEL,
        "inputImage": str(image_path),
        "latencyMs": latency_ms,
        "description": description,
        "requestId": response["ResponseMetadata"]["RequestId"],
    }


def main() -> None:
    # Point d'entrée : invocation unique puis sérialisation de l'artefact d'audit.
    audit = describe_image(Path(IMAGE_PATH))
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
