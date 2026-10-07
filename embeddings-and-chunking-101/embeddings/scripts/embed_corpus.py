#!/usr/bin/env python3
"""Generates a reference vector embedding for the first document in the corpus.

Parent lab: embeddings-and-chunking-101.
AWS Services: Amazon Bedrock Runtime (Titan Embeddings v2).
Generated artifacts: embeddings/reference-vector.json (vecteur, dimension, L2 norm, head).
Mode: live (un appel Bedrock sur le premier document).
"""
import json
import os
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError, EndpointConnectionError, NoCredentialsError
import numpy as np

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

MODEL_ID = os.environ.get("BEDROCK_EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0")
DIMENSIONS = int(os.environ.get("BEDROCK_EMBED_DIMENSIONS", "512"))
AWS_REGION = os.environ.get("AWS_REGION", "eu-west-1")
CORPUS_PATH = Path("embeddings/data/sample-corpus.json")
OUTPUT_PATH = Path("embeddings/reference-vector.json")


def load_corpus() -> list[dict]:
    return json.loads(CORPUS_PATH.read_text(encoding="utf-8"))


def embed(text: str) -> np.ndarray:
    # L'API Titan renvoie déjà un vecteur normalisé, on re-normalise par sécurité.
    client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    response = client.invoke_model(
        modelId=MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(
            {"inputText": text, "dimensions": DIMENSIONS, "normalize": True}
        ),
    )
    payload = json.loads(response["body"].read())
    vector = np.asarray(payload["embedding"], dtype=np.float32)
    return vector / np.linalg.norm(vector)


def main() -> None:
    # Démo : on n'embarque que le premier doc pour un artefact léger.
    corpus = load_corpus()
    reference = corpus[0]
    vector = embed(reference["content"])
    OUTPUT_PATH.write_text(
        json.dumps(
            {
                "docId": reference["id"],
                "modelId": MODEL_ID,
                "dimensions": DIMENSIONS,
                "rawDimension": int(vector.shape[0]),
                "l2Norm": float(np.linalg.norm(vector)),
                "head": vector[:8].tolist(),
                "requestId": "n/a",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "docId": reference["id"],
                "modelId": MODEL_ID,
                "rawDimension": int(vector.shape[0]),
                "l2Norm": round(float(np.linalg.norm(vector)), 4),
            },
            indent=2,
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
