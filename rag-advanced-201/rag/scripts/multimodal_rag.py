"""Multimodal RAG pipeline: embeds an image with Titan Multimodal and describes it with a vision model.

Parent lab: rag-advanced-201.
AWS Services: Amazon Bedrock Runtime (Titan Multimodal Embeddings G1 + Amazon Nova Lite vision).
Generated artifacts: rag/audit/multimodal-rag-audit.json (vecteur, latences, description).
Mode: live (appels Bedrock facturés).
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
EMBED_MODEL = os.environ.get("TRAINING_MM_EMBED_MODEL", "amazon.titan-embed-image-v1")
VISION_MODEL = os.environ.get("TRAINING_MM_VISION_MODEL", "amazon.nova-lite-v1:0")
IMAGE_PATH = os.environ.get("TRAINING_RAG_IMAGE_PATH", "rag/data/sample-diagram.png")
QUESTION = os.environ.get("TRAINING_RAG_MM_QUESTION", "Explique le diagramme en 2 phrases.")
OUTPUT_PATH = Path("rag/audit/multimodal-rag-audit.json")


def main() -> None:
    # Read image bytes and encode to base64 required by Bedrock multimodal APIs.
    image_bytes = Path(IMAGE_PATH).read_bytes()
    image_b64 = base64.b64encode(image_bytes).decode("ascii")
    runtime = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    # Step 1: multimodal embedding generation for semantic search indexing.
    embed_started = time.perf_counter()
    embed_response = runtime.invoke_model(
        modelId=EMBED_MODEL,
        contentType="application/json",
        accept="application/json",
        body=json.dumps({"inputImage": image_b64, "embeddingConfig": {"outputEmbeddingLength": 1024}}),
    )
    embed_latency_ms = round((time.perf_counter() - embed_started) * 1000, 2)
    embedding = json.loads(embed_response["body"].read())["embedding"]

    # Step 2: multimodal Converse API call to generate natural language description.
    vision_started = time.perf_counter()
    vision_response = runtime.converse(
        modelId=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"image": {"format": "png", "source": {"bytes": image_b64}}},
                    {"text": QUESTION},
                ],
            }
        ],
        inferenceConfig={"temperature": 0.0, "maxTokens": 256},
    )
    vision_latency_ms = round((time.perf_counter() - vision_started) * 1000, 2)
    # Defensive concatenation across response content blocks.
    answer = "".join(
        block.get("text", "")
        for block in vision_response["output"]["message"].get("content", [])
    ).strip()
    audit = {
        "embedModelId": EMBED_MODEL,
        "visionModelId": VISION_MODEL,
        "imagePath": IMAGE_PATH,
        "question": QUESTION,
        "embeddingDimension": len(embedding),
        "embedLatencyMs": embed_latency_ms,
        "visionLatencyMs": vision_latency_ms,
        "answer": answer,
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
