"""Re-ranks document chunks using Cohere Rerank v3 hosted on Amazon Bedrock.

Parent lab: rag-advanced-201.
AWS Services: Amazon Bedrock Runtime (modèle Cohere Rerank v3-5).
Generated artifacts: rag/audit/rerank-audit.json (scores, ordre, latence).
Mode: live (consomme des tokens Bedrock facturés).
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
RERANK_MODEL = os.environ.get("TRAINING_RERANK_MODEL", "cohere.rerank-v3-5:0")
CORPUS_PATH = Path("rag/data/tech-corpus.json")
QUERY = os.environ.get("TRAINING_RERANK_QUERY", "Comment fonctionne le re-ranking Bedrock ?")
OUTPUT_PATH = Path("rag/audit/rerank-audit.json")


def rerank(client: object, query: str, documents: list[str]) -> dict:
    # Mesure de la latence d'invocation du modèle de re-ranking.
    started = time.perf_counter()
    response = client.invoke_model(
        modelId=RERANK_MODEL,
        contentType="application/json",
        accept="application/json",
        body=json.dumps({"query": query, "documents": documents, "top_n": min(3, len(documents))}),
    )
    payload = json.loads(response["body"].read())
    # Projection des champs Cohere vers un format neutre consommé par l'UI.
    return {
        "modelId": RERANK_MODEL,
        "latencyMs": round((time.perf_counter() - started) * 1000, 2),
        "results": [
            {"index": item.get("index"), "relevanceScore": item.get("relevance_score")}
            for item in payload.get("results", [])
        ],
        "requestId": response["ResponseMetadata"]["RequestId"],
    }


def main() -> None:
    # Load corpus, flatten document chunks, and call Bedrock embeddings.
    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    documents = [chunk["content"] for chunk in corpus]
    runtime = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    audit = rerank(runtime, QUERY, documents)
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
