#!/usr/bin/env python3
"""Applies runtime cost guardrails (maxTokens limit, model fallback) and outputs a budget report.

Parent lab: cost-capacity-101.
AWS Services: Amazon Bedrock Runtime (converse, modèles primaire + fallback).
Generated artifacts: cost/budget-audit-log.json (runs, coûts, fallbackCount, totalCostUsd).
Mode: live (chaque prompt est envoyé à Bedrock).
"""
import json
import os
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError, EndpointConnectionError, NoCredentialsError

from batching import chunk_requests

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
PRIMARY_MODEL_ID = os.environ.get("BEDROCK_PRIMARY_MODEL_ID", "amazon.nova-lite-v1:0")
FALLBACK_MODEL_ID = os.environ.get("BEDROCK_FALLBACK_MODEL_ID", "amazon.nova-micro-v1:0")
POLICY_PATH = Path("cost/budget-policy.json")
PROMPTS_PATH = Path("cost/prompts.txt")
OUTPUT_PATH = Path("cost/budget-audit-log.json")

INPUT_TOKEN_PRICE_PER_1K = 0.00006
OUTPUT_TOKEN_PRICE_PER_1K = 0.00024


def load_policy() -> dict:
    return json.loads(POLICY_PATH.read_text(encoding="utf-8"))


def load_prompts() -> list[str]:
    return [line.strip() for line in PROMPTS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def estimate_cost_usd(usage: dict) -> float:
    # Coût = (in/1000)*prix_in + (out/1000)*prix_out, tarifs Nova Lite.
    in_t = float(usage.get("inputTokens", 0))
    out_t = float(usage.get("outputTokens", 0))
    return round((in_t / 1000.0) * INPUT_TOKEN_PRICE_PER_1K + (out_t / 1000.0) * OUTPUT_TOKEN_PRICE_PER_1K, 6)


def run_with_budget(client, model_id: str, prompt: str, max_tokens: int) -> dict:
    # maxTokens est imposé par prompt pour plafonner la dépense.
    response = client.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": max_tokens},
    )
    usage = response.get("usage", {})
    return {
        "modelId": model_id,
        "stopReason": response.get("stopReason"),
        "inputTokens": usage.get("inputTokens"),
        "outputTokens": usage.get("outputTokens"),
        "costUsd": estimate_cost_usd(usage),
        "requestId": response["ResponseMetadata"]["RequestId"],
    }


def main() -> None:
    # Load policy, batch prompts, and execute primary model with conditional fallback.
    policy = load_policy()
    max_tokens = int(policy.get("maxTokens", 500))
    client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    prompts = load_prompts()
    batches = chunk_requests(prompts, batch_size=3)

    runs: list[dict] = []
    for batch in batches:
        for prompt in batch:
            primary = run_with_budget(client, PRIMARY_MODEL_ID, prompt, max_tokens)
            entry = {
                "prompt": prompt,
                "primary": primary,
                "withinBudget": primary["costUsd"] <= policy["monthlyLimitUsd"] / 1000.0,
            }
            # Fallback automatique uniquement si le primaire a tronqué sa réponse.
            if primary["stopReason"] == "max_tokens":
                entry["fallback"] = run_with_budget(
                    client, FALLBACK_MODEL_ID, prompt, max_tokens
                )
            runs.append(entry)

    # Aggregation: total cost, fallback count, and monthly budget compliance.
    total_cost = round(sum(item["primary"]["costUsd"] for item in runs), 6)
    fallback_count = sum(1 for item in runs if "fallback" in item)
    audit_log = {
        "region": AWS_REGION,
        "policy": policy,
        "primaryModelId": PRIMARY_MODEL_ID,
        "fallbackModelId": FALLBACK_MODEL_ID,
        "batches": len(batches),
        "runs": runs,
        "summary": {
            "totalCostUsd": total_cost,
            "monthlyLimitUsd": policy["monthlyLimitUsd"],
            "fallbackCount": fallback_count,
        },
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
