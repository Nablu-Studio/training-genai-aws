#!/usr/bin/env python3
"""Applies a fixed-size chunking strategy with overlap and evaluates intra-chunk coherence via cosine similarity.

Parent lab: embeddings-and-chunking-101.
AWS Services: Amazon Bedrock Runtime (Titan Embeddings v2, 1 appel par phrase).
Generated artifacts: embeddings/chunk-audit.json (chunks, scores, chunks à faible cohérence).
Mode: live (coûteux : un appel Bedrock par phrase par chunk).
"""
import json
import os
import statistics
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
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "200"))
OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "40"))
COHERENCE_THRESHOLD = 0.5

CORPUS_PATH = Path("embeddings/data/sample-corpus.json")
OUTPUT_PATH = Path("embeddings/chunk-audit.json")


def load_corpus() -> list[dict]:
    return json.loads(CORPUS_PATH.read_text(encoding="utf-8"))


def embed(text: str) -> np.ndarray:
    # Vecteur Titan déjà normalisé, exploitable directement pour un cosinus.
    client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    response = client.invoke_model(
        modelId=MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(
            {"inputText": text, "dimensions": DIMENSIONS, "normalize": True}
        ),
    )
    return np.asarray(
        json.loads(response["body"].read())["embedding"], dtype=np.float32
    )


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    # Découpage par fenêtre glissante de tokens (mots) avec recouvrement paramétrable.
    tokens = text.split()
    if not tokens:
        return []
    chunks: list[str] = []
    cursor = 0
    while cursor < len(tokens):
        piece = tokens[cursor : cursor + size]
        if not piece:
            break
        chunks.append(" ".join(piece))
        if cursor + size >= len(tokens):
            break
        cursor += size - overlap
    return chunks


def intra_chunk_coherence(text: str) -> float:
    # Cosinus moyen entre toutes les paires de phrases d'un chunk.
    sentences = [sentence.strip() for sentence in text.split(".") if sentence.strip()]
    if len(sentences) < 2:
        return 1.0
    vectors = [embed(sentence) for sentence in sentences]
    pairs = []
    for index in range(len(vectors)):
        for other in range(index + 1, len(vectors)):
            pairs.append(float(np.dot(vectors[index], vectors[other])))
    return round(statistics.fmean(pairs), 4)


def main() -> None:
    # Pipeline complet : chunking -> scoring -> audit JSON récapitulatif.
    corpus = load_corpus()
    chunks: list[dict] = []
    coherences: list[float] = []
    for doc in corpus:
        for index, piece in enumerate(chunk_text(doc["content"], CHUNK_SIZE, OVERLAP)):
            score = intra_chunk_coherence(piece)
            chunks.append(
                {
                    "docId": doc["id"],
                    "chunkIndex": index,
                    "tokens": len(piece.split()),
                    "coherence": score,
                    "preview": piece[:60] + ("..." if len(piece) > 60 else ""),
                }
            )
            coherences.append(score)
    audit = {
        "modelId": MODEL_ID,
        "dimensions": DIMENSIONS,
        "strategy": {"name": "fixed_size", "size": CHUNK_SIZE, "overlap": OVERLAP},
        "corpusSize": len(corpus),
        "chunks": chunks,
        "summary": {
            "chunkCount": len(chunks),
            "meanCoherence": round(statistics.fmean(coherences), 4),
            "lowCoherenceChunks": [
                f"{chunk['docId']}#{chunk['chunkIndex']}"
                for chunk in chunks
                if chunk["coherence"] < COHERENCE_THRESHOLD
            ],
        },
    }
    OUTPUT_PATH.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit["summary"], indent=2))


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
