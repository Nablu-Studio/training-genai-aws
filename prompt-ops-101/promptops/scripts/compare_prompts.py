#!/usr/bin/env python3
"""Compares a prompt baseline and candidate version on a JSON dataset (scores, latency, cost).

Parent lab: prompt-ops-101.
AWS Services: Amazon Bedrock Runtime (converse).
Generated artifacts: compare-report.json (résumé + détails par ligne).
Mode: live (deux invocations Bedrock par ligne du dataset).
"""
import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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

DEFAULT_MODEL_ID = "amazon.nova-lite-v1:0"
OUTPUT_FILE = "compare-report.json"
INPUT_TOKEN_PRICE_PER_1K = 0.00006
OUTPUT_TOKEN_PRICE_PER_1K = 0.00024


@dataclass
class PromptRunResult:
    label: str
    average_score: float
    average_latency_ms: float
    average_cost_usd: float
    details: list[dict[str, Any]]


def parse_args() -> argparse.Namespace:
    # argparse : baseline/candidate/dataset obligatoires, model/region/output avec défauts.
    parser = argparse.ArgumentParser(
        description="Compare two prompt versions against a JSON evaluation dataset."
    )
    parser.add_argument("--baseline", required=True, help="Path to the baseline prompt file.")
    parser.add_argument("--candidate", required=True, help="Path to the candidate prompt file.")
    parser.add_argument("--dataset", required=True, help="Path to a JSON dataset with input and expected fields.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Bedrock model ID used with converse.")
    parser.add_argument("--region", default="eu-west-1", help="AWS region for the Bedrock runtime client.")
    parser.add_argument("--output", default=OUTPUT_FILE, help="Where to write the comparison report.")
    return parser.parse_args()


def read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8").strip()


def load_dataset(path: str) -> list[dict[str, str]]:
    # Validation stricte : tableau non vide, lignes objet avec input/expected non vides.
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("Dataset must be a non-empty JSON array.")

    normalized: list[dict[str, str]] = []
    for index, row in enumerate(data, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Dataset row {index} must be an object.")
        user_input = str(row.get("input", "")).strip()
        expected = str(row.get("expected", "")).strip()
        if not user_input or not expected:
            raise ValueError(
                f"Dataset row {index} must include non-empty input and expected fields."
            )
        normalized.append({"input": user_input, "expected": expected})
    return normalized


def normalize_text(value: str) -> list[str]:
    # Normalisation : on ne garde que les alphanumériques en minuscule, séparés par espaces.
    normalized = "".join(ch.lower() if ch.isalnum() else " " for ch in value)
    return [token for token in normalized.split() if token]


def score_answer(answer: str, expected: str) -> float:
    # Recouvrement de tokens normalisés (équivalent recall côté référence).
    answer_tokens = set(normalize_text(answer))
    expected_tokens = set(normalize_text(expected))
    if not expected_tokens:
        return 0.0
    overlap = len(answer_tokens & expected_tokens)
    return round(overlap / len(expected_tokens), 4)


def estimate_cost_usd(usage: dict[str, Any]) -> float:
    # Coût = (in/1000)*prix_in + (out/1000)*prix_out, tarifs Nova Lite.
    input_tokens = float(usage.get("inputTokens", 0))
    output_tokens = float(usage.get("outputTokens", 0))
    input_cost = (input_tokens / 1000.0) * INPUT_TOKEN_PRICE_PER_1K
    output_cost = (output_tokens / 1000.0) * OUTPUT_TOKEN_PRICE_PER_1K
    return round(input_cost + output_cost, 6)


def invoke_model(
    client: Any, model_id: str, prompt: str, user_input: str
) -> tuple[str, dict[str, Any], float]:
    # Construction du prompt complet puis appel converse avec inférence déterministe.
    started_at = time.perf_counter()
    response = client.converse(
        modelId=model_id,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "text": (
                            f"{prompt}\n\nUser input:\n{user_input}\n\n"
                            "Respond in plain text."
                        )
                    }
                ],
            }
        ],
        inferenceConfig={
            "temperature": 0.2,
            "maxTokens": 300,
            "topP": 0.9,
        },
    )
    latency_ms = (time.perf_counter() - started_at) * 1000.0
    output = response.get("output", {})
    message = output.get("message", {})
    content = message.get("content", [])
    # Concaténation défensive des blocs texte.
    text = "".join(block.get("text", "") for block in content if isinstance(block, dict)).strip()
    usage = response.get("usage", {})
    return text, usage, latency_ms


def run_prompt(
    label: str,
    prompt_path: str,
    dataset: list[dict[str, str]],
    client: Any,
    model_id: str,
) -> PromptRunResult:
    prompt = read_text(prompt_path)
    details: list[dict[str, Any]] = []

    for row in dataset:
        answer, usage, latency_ms = invoke_model(client, model_id, prompt, row["input"])
        details.append(
            {
                "input": row["input"],
                "expected": row["expected"],
                "answer": answer,
                "score": score_answer(answer, row["expected"]),
                "latency_ms": round(latency_ms, 2),
                "cost_usd": estimate_cost_usd(usage),
                "usage": usage,
            }
        )

    return PromptRunResult(
        label=label,
        average_score=round(statistics.mean(item["score"] for item in details), 4),
        average_latency_ms=round(statistics.mean(item["latency_ms"] for item in details), 2),
        average_cost_usd=round(statistics.mean(item["cost_usd"] for item in details), 6),
        details=details,
    )


def build_report(
    baseline: PromptRunResult, candidate: PromptRunResult, args: argparse.Namespace
) -> dict[str, Any]:
    # Delta de score et variation de coût en pourcentage (n/a si baseline nulle).
    baseline_score = baseline.average_score
    candidate_score = candidate.average_score
    score_delta = round(candidate_score - baseline_score, 4)

    baseline_cost = baseline.average_cost_usd
    candidate_cost = candidate.average_cost_usd
    if math.isclose(baseline_cost, 0.0):
        cost_delta_pct = "n/a"
    else:
        cost_delta_pct = f"{((candidate_cost - baseline_cost) / baseline_cost) * 100:+.1f}%"

    # Recommandation = prompt avec le meilleur score moyen (égalité -> baseline).
    return {
        "model_id": args.model_id,
        "region": args.region,
        "dataset": args.dataset,
        "baseline": {
            "prompt_path": args.baseline,
            "average_score": baseline_score,
            "average_latency_ms": baseline.average_latency_ms,
            "average_cost_usd": baseline_cost,
            "details": baseline.details,
        },
        "candidate": {
            "prompt_path": args.candidate,
            "average_score": candidate_score,
            "average_latency_ms": candidate.average_latency_ms,
            "average_cost_usd": candidate_cost,
            "details": candidate.details,
        },
        "summary": {
            "score_delta": score_delta,
            "cost_delta_pct": cost_delta_pct,
            "latency_delta_ms": round(candidate.average_latency_ms - baseline.average_latency_ms, 2),
            "recommended_winner": "candidate" if candidate_score >= baseline_score else "baseline",
        },
    }


def main() -> None:
    # 1) parse args, 2) execute baseline + candidate on dataset, 3) write report.
    args = parse_args()
    dataset = load_dataset(args.dataset)
    client = boto3.client("bedrock-runtime", region_name=args.region)

    baseline = run_prompt("baseline", args.baseline, dataset, client, args.model_id)
    candidate = run_prompt("candidate", args.candidate, dataset, client, args.model_id)
    report = build_report(baseline, candidate, args)

    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"Report written to {args.output}")


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
