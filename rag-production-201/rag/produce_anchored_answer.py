#!/usr/bin/env python3
"""Produces a grounded RAG answer anchored on explicit citations, with fallback `no_answer` when data is stale.

Parent lab: rag-production-201.
AWS Services: Amazon Bedrock Runtime (converse), réutilise le reranker local du module `rerank.py`.
Generated artifacts: rag/anchored-answers.json (réponses, sources, latences, refus).
Mode: live (appels Bedrock facturés).
"""
import json
import os
import sys
from pathlib import Path
from time import perf_counter

import boto3
from botocore.exceptions import ClientError, EndpointConnectionError, NoCredentialsError

from rerank import rerank

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
MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "amazon.nova-lite-v1:0")
KB_CHUNKS_PATH = Path("rag/sample-chunks.json")
QUESTIONS_PATH = Path("rag/questions.json")
FRESHNESS_PATH = Path("rag/freshness-policy.json")
OUTPUT_PATH = Path("rag/anchored-answers.json")


def load_chunks() -> list[dict]:
    return json.loads(KB_CHUNKS_PATH.read_text(encoding="utf-8"))


def load_questions() -> list[dict]:
    return json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))


def load_policy() -> dict:
    return json.loads(FRESHNESS_PATH.read_text(encoding="utf-8"))


def select_chunks(question: str, chunks: list[dict], top_k: int = 3) -> list[dict]:
    # Re-ranking local puis conservation des top-k chunks les plus pertinents.
    return [
        {**chunk, "score": rank}
        for rank, chunk in enumerate(rerank(chunks, question)[:top_k], start=1)
    ]


def build_context(chunks: list[dict]) -> str:
    # Format `[id] contenu` pour faciliter l'extraction des citations côté modèle.
    return "\n\n".join(f"[{chunk['id']}] {chunk['content']}" for chunk in chunks)


def is_fresh(chunks: list[dict], policy: dict) -> bool:
    # Tous les chunks doivent être plus récents que le seuil métier.
    stale_after = policy.get("staleAfterDays", 30)
    return all(chunk.get("freshnessDays", 0) <= stale_after for chunk in chunks)


def run_question(client, question: str, chunks: list[dict], policy: dict) -> dict:
    started = perf_counter()
    selected = select_chunks(question, chunks)
    # Fallback branch: refuse to answer rather than hallucinate if data is stale.
    if policy.get("fallbackWhenStale") == "no_answer" and not is_fresh(selected, policy):
        return {
            "question": question,
            "answer": "No answer: source data is stale.",
            "sources": [chunk["id"] for chunk in selected],
            "refused": True,
            "latencyMs": round((perf_counter() - started) * 1000.0, 2),
        }
    context = build_context(selected)
    response = client.converse(
        modelId=MODEL_ID,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "text": (
                            f"Contexte:\n{context}\n\nQuestion: {question}\n"
                            "Reponds UNIQUEMENT en t'appuyant sur le contexte. "
                            "Cite les ids utilises."
                        )
                    }
                ],
            }
        ],
    )
    answer = "".join(
        block.get("text", "")
        for block in response["output"]["message"].get("content", [])
        if isinstance(block, dict)
    ).strip()
    # Extraction des IDs cités dans la réponse pour auditabilité.
    cited = [token for token in answer.split() if token.startswith("[") and token.endswith("]")]
    return {
        "question": question,
        "answer": answer,
        "sources": [chunk["id"] for chunk in selected],
        "citedIds": cited,
        "refused": False,
        "latencyMs": round((perf_counter() - started) * 1000.0, 2),
        "requestId": response["ResponseMetadata"]["RequestId"],
    }


def main() -> None:
    # Loop over all evaluation questions and serialize audit report.
    client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    policy = load_policy()
    chunks = load_chunks()
    questions = load_questions()
    answers = [run_question(client, item["q"], chunks, policy) for item in questions]
    audit_log = {
        "region": AWS_REGION,
        "modelId": MODEL_ID,
        "policy": policy,
        "chunks": len(chunks),
        "answers": answers,
    }
    OUTPUT_PATH.write_text(json.dumps(audit_log, indent=2), encoding="utf-8")
    print(json.dumps(audit_log, indent=2))


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
