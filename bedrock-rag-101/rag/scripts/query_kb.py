"""Demonstrates a minimal Bedrock RAG: lexical ranking, context injection, and concise response generation.

Parent lab: bedrock-rag-101.
AWS Services: Amazon Bedrock Runtime (converse).
Generated artifacts: rag/selected-docs.json + rag/rag-answer.json.
Mode: live (un appel converse avec contexte).

Environment variables:
  AWS_REGION        région Bedrock (default eu-west-1)
  TRAINING_MODEL_ID modèle appelé (default amazon.nova-lite-v1:0)
"""
import json
import os
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError, EndpointConnectionError, NoCredentialsError

RAG_DIR = Path(__file__).resolve().parents[1]
REGION = os.environ.get("AWS_REGION", "eu-west-1")
MODEL_ID = os.environ.get("TRAINING_MODEL_ID", "amazon.nova-lite-v1:0")

# Distinguishable error modes for the learner when an API call fails.
AWS_ERROR_HINTS = {
    "AccessDeniedException": (
        "Access denied. Either the IAM role/user lacks bedrock:InvokeModel, "
        "or model access is not enabled (Bedrock console > Model access)."
    ),
    "ResourceNotFoundException": (
        f"Model not found in {REGION}. Check region availability: "
        "not all models are available in every region."
    ),
    "ValidationException": (
        "Request rejected by Bedrock. Invalid model ID, or model requires an inference profile."
    ),
    "ThrottlingException": "Rate limit exceeded: retry later or reduce request frequency.",
}

DOCS = json.loads((RAG_DIR / "data/sample-docs.json").read_text(encoding="utf-8"))


def rank_docs(question: str) -> list[dict]:
    # Ranking lexical simple : score = nombre de tokens de la question présents dans le doc.
    keywords = set(question.lower().split())
    return sorted(
        DOCS,
        key=lambda doc: sum(token in doc["content"].lower() for token in keywords),
        reverse=True,
    )


def build_context(question: str) -> tuple[str, list[dict]]:
    # Top-3 documents uniquement, on conserve la liste pour audit.
    ranked = rank_docs(question)
    selected = ranked[:3]
    context = "\n\n".join(
        f"[{doc['id']}] {doc['content']}"
        for doc in selected
    )
    return context, selected


def ask_bedrock(client, context: str, question: str) -> dict:
    try:
        return client.converse(
            modelId=MODEL_ID,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "text": (
                                f"Contexte:\n{context}\n\nQuestion: {question}\n"
                                "Réponds en 2 phrases maximum, uniquement à partir du contexte."
                            )
                        }
                    ],
                }
            ],
            # Plafonner la sortie : c'est aussi un plafond de coût.
            inferenceConfig={"maxTokens": 200, "temperature": 0},
        )
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        hint = AWS_ERROR_HINTS.get(code, "Erreur Bedrock non prévue : lisez le message ci-dessous.")
        sys.exit(f"[{code}] {hint}\nDétail : {exc.response['Error'].get('Message', '')}")
    except NoCredentialsError:
        sys.exit("No AWS credentials found: run `aws configure` or export AWS_PROFILE.")
    except EndpointConnectionError:
        sys.exit(f"Impossible de joindre Bedrock dans {REGION} : vérifiez AWS_REGION et le réseau.")


def main() -> None:
    # Récupération du contexte puis appel converse avec contraintes (2 phrases max).
    question = "A quoi sert Bedrock dans un flux RAG ?"
    context, selected_docs = build_context(question)
    client = boto3.client("bedrock-runtime", region_name=REGION)
    response = ask_bedrock(client, context, question)
    answer = response["output"]["message"]["content"][0]["text"]
    usage = response.get("usage", {})
    # Sérialisation séparée : docs sélectionnés et réponse finale, pour deux panneaux UI.
    (RAG_DIR / "selected-docs.json").write_text(
        json.dumps(selected_docs, indent=2),
        encoding="utf-8",
    )
    (RAG_DIR / "rag-answer.json").write_text(
        json.dumps(
            {
                "question": question,
                "selectedDocIds": [doc["id"] for doc in selected_docs],
                "answer": answer,
                "usage": {
                    "inputTokens": usage.get("inputTokens"),
                    "outputTokens": usage.get("outputTokens"),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"selectedDocIds": [doc["id"] for doc in selected_docs], "answer": answer}, indent=2))
    # Le contexte injecté pèse dans la facture : on le rend visible à chaque appel.
    print(f"Tokens : {usage.get('inputTokens')} en entrée, {usage.get('outputTokens')} en sortie")


if __name__ == "__main__":
    main()
