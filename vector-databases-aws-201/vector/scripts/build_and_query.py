#!/usr/bin/env python3
"""Creates a sample OpenSearch vector index, inserts chunks, and executes a filtered k-NN query.

Parent lab: vector-databases-aws-201.
AWS Services: Amazon OpenSearch (k-NN), Amazon Bedrock Runtime (Titan Embeddings v2).
Generated artifacts: vector/audit/index-query-audit.json (résumé latences, IDs matchés).
Mode: live (écrit dans l'index OpenSearch configuré via TRAINING_OPENSEARCH_INDEX).
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
    "AccessDeniedException": "Insufficient permissions: check the IAM policy of the role or current identity.",
    "ModelTimeoutException": "Model timed out: reduce prompt size or max tokens.",
    "ResourceNotFoundException": "Domaine OpenSearch introuvable : vérifiez son nom et la région.",
    "ThrottlingException": "Quota exceeded: reduce call rate or retry with exponential backoff.",
    "ValidationException": "Requête refusée : un paramètre est invalide, relisez le message ci-dessous.",
}


def explain_aws_error(exc: ClientError) -> str:
    """Translates an AWS ClientError into an actionable diagnostic message for the learner."""
    error = exc.response["Error"]
    code = error.get("Code", "Unknown")
    hint = AWS_ERROR_HINTS.get(code, "Unexpected AWS error: read details below.")
    return f"[{code}] {hint}\nAWS Details: {error.get('Message', '')}"

AWS_REGION = os.environ.get("AWS_REGION", "eu-west-1")
CHUNKS_PATH = Path("vector/data/sample-chunks.json")
INDEX_NAME = os.environ.get("TRAINING_OPENSEARCH_INDEX", "nablu-training-chunks")
OUTPUT_PATH = Path("vector/audit/index-query-audit.json")


def embed(text: str) -> list[float]:
    # Call Bedrock to retrieve the normalized embedding vector.
    client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    response = client.invoke_model(
        modelId="amazon.titan-embed-text-v2:0",
        contentType="application/json",
        accept="application/json",
        body=json.dumps(
            {"inputText": text, "dimensions": 512, "normalize": True}
        ),
    )
    return json.loads(response["body"].read())["embedding"]


def main() -> None:
    # Load demonstration corpus and insert into vector index.
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    os_client = boto3.client("opensearch", region_name=AWS_REGION)
    insert_latencies: list[float] = []
    for chunk in chunks:
        started = time.perf_counter()
        os_client.index(
            index=INDEX_NAME,
            id=chunk["id"],
            body={
                "content": chunk["content"],
                "embedding": embed(chunk["content"]),
                "metadata": chunk.get("metadata", {}),
            },
        )
        insert_latencies.append(round((time.perf_counter() - started) * 1000, 2))

    # Requête hybride : k-NN sur l'embedding + filtre méta sur la fraîcheur.
    query = "Comment proteger les appels Bedrock ?"
    started = time.perf_counter()
    response = os_client.search(
        index=INDEX_NAME,
        body={
            "size": 2,
            "query": {
                "bool": {
                    "must": [
                        {"knn": {"embedding": {"vector": embed(query), "k": 2}}},
                    ],
                    "filter": [{"term": {"metadata.freshnessDays": 7}}],
                }
            },
        },
    )
    query_latency = round((time.perf_counter() - started) * 1000, 2)
    hits = [hit["_id"] for hit in response["hits"]["hits"]]
    # Écriture de l'artefact d'audit : latences d'insertion et de recherche + IDs matchés.
    audit = {
        "indexName": INDEX_NAME,
        "region": AWS_REGION,
        "vectorsInserted": len(chunks),
        "insertLatencyMs": insert_latencies,
        "query": {
            "question": query,
            "topK": 2,
            "filter": {"metadata.freshnessDays": 7},
            "latencyMs": query_latency,
            "matchedIds": hits,
        },
    }
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
