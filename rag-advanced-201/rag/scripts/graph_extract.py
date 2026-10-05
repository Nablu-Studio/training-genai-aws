"""Extracts knowledge graph triplets (entity, relation, entity) from a corpus for GraphRAG.

Parent lab: rag-advanced-201.
AWS Services: Amazon Bedrock Runtime (converse avec sortie JSON contrainte).
Generated artifacts: rag/audit/graph-audit.json (triplets, entités, contexte injectable).
Mode: live (un appel Bedrock par chunk du corpus).

Environment variables:
  AWS_REGION            région Bedrock (default eu-west-1)
  TRAINING_GRAPH_MODEL  modèle appelé (default amazon.nova-lite-v1:0)
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
MODEL_ID = os.environ.get("TRAINING_GRAPH_MODEL", "amazon.nova-lite-v1:0")
CORPUS_PATH = Path("rag/data/tech-corpus.json")
OUTPUT_PATH = Path("rag/audit/graph-audit.json")

INSTRUCTION = (
    "Extrais les relations factuelles du passage sous forme de triplets. "
    "Réponds uniquement par un tableau JSON d'objets "
    '{"sujet": "...", "relation": "...", "objet": "..."}, sans texte autour. '
    "N'invente aucune relation absente du passage ; renvoie [] si le passage n'en contient pas."
)


def parse_triplets(answer: str) -> list[dict]:
    """Model may wrap JSON with Markdown text: parse and isolate the JSON array."""
    start, end = answer.find("["), answer.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        parsed = json.loads(answer[start : end + 1])
    except json.JSONDecodeError:
        return []
    # On ne garde que les triplets complets : un champ manquant casse le graphe.
    return [
        {"sujet": str(item["sujet"]), "relation": str(item["relation"]), "objet": str(item["objet"])}
        for item in parsed
        if isinstance(item, dict) and {"sujet", "relation", "objet"} <= set(item)
    ]


def extract_from_chunk(client, chunk: dict) -> tuple[list[dict], dict]:
    response = client.converse(
        modelId=MODEL_ID,
        messages=[{"role": "user", "content": [{"text": f"{INSTRUCTION}\n\nPassage :\n{chunk['content']}"}]}],
        # Température nulle : deux exécutions doivent donner le même graphe.
        inferenceConfig={"maxTokens": 500, "temperature": 0},
    )
    answer = response["output"]["message"]["content"][0]["text"]
    triplets = [dict(triplet, source=chunk["id"]) for triplet in parse_triplets(answer)]
    return triplets, response.get("usage", {})


def build_context(triplets: list[dict]) -> str:
    """Relational context injected into RAG prompt, formatted with one line per triplet."""
    return "\n".join(f"({t['sujet']}) -[{t['relation']}]-> ({t['objet']})" for t in triplets)


def main() -> None:
    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    client = boto3.client("bedrock-runtime", region_name=AWS_REGION)

    started = time.perf_counter()
    triplets: list[dict] = []
    input_tokens = output_tokens = 0
    for chunk in corpus:
        chunk_triplets, usage = extract_from_chunk(client, chunk)
        triplets += chunk_triplets
        input_tokens += usage.get("inputTokens", 0)
        output_tokens += usage.get("outputTokens", 0)

    entities = sorted({t["sujet"] for t in triplets} | {t["objet"] for t in triplets})
    audit = {
        "modelId": MODEL_ID,
        "chunks": len(corpus),
        "triplets": triplets,
        "entities": entities,
        "relations": sorted({t["relation"] for t in triplets}),
        # Context text injected into the RAG prompt.
        "graphContext": build_context(triplets),
        "latencyMs": round((time.perf_counter() - started) * 1000, 2),
        "usage": {"inputTokens": input_tokens, "outputTokens": output_tokens},
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: audit[k] for k in ("chunks", "entities", "relations", "usage")}, indent=2, ensure_ascii=False))


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
