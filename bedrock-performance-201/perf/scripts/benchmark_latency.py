"""Benchmarks Bedrock latency with and without prompt caching, computing p50/p99 percentiles.

Parent lab: bedrock-performance-201.
AWS Services: Amazon Bedrock Runtime (converse + promptCache).
Generated artifacts: perf/audit/latency-audit.json (p50/p99 sans et avec cache, delta).
Mode: live (N itérations Bedrock par variante, attention au coût).
"""
import json
import os
import statistics
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
MODEL_ID = os.environ.get("TRAINING_PERF_MODEL", "amazon.nova-lite-v1:0")
ITERATIONS = int(os.environ.get("TRAINING_PERF_ITERATIONS", "10"))
OUTPUT_PATH = Path("perf/audit/latency-audit.json")

SYSTEM_PROMPT = (
    "You are an expert AWS GenAI assistant. Answer concisely in English, "
    "citing the relevant AWS service. Always include a brief technical rationale. "
    "Detailed business context to bring the system prompt above the Bedrock cache threshold: "
    + " ".join(f"mot{i}" for i in range(1200))
)

client = boto3.client("bedrock-runtime", region_name=AWS_REGION)


def call_once(use_cache: bool, question: str) -> tuple[float, dict]:
    # Enable prompt cache only when use_cache=True to benchmark cache latency.
    inference_config = {"temperature": 0.0, "maxTokens": 128}
    if use_cache:
        inference_config["promptCache"] = {"type": "default"}
    started = time.perf_counter()
    response = client.converse(
        modelId=MODEL_ID,
        system=[{"text": SYSTEM_PROMPT}],
        messages=[{"role": "user", "content": [{"text": question}]}],
        inferenceConfig=inference_config,
    )
    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    return latency_ms, response.get("usage", {})


def percentile(values: list[float], p: float) -> float:
    # Compute percentile p using inclusive rank method.
    if not values:
        return 0.0
    return round(statistics.quantiles(values, n=100, method="inclusive")[int(p) - 1], 2)


def main() -> None:
    # Warmup call before each run series to neutralize cold-start overhead.
    call_once(False, "warmup")
    no_cache: list[float] = []
    no_cache_usage: list[dict] = []
    for _ in range(ITERATIONS):
        latency, usage = call_once(False, "What is the capital of France?")
        no_cache.append(latency)
        no_cache_usage.append(usage)

    call_once(True, "warmup")
    cached: list[float] = []
    cached_usage: list[dict] = []
    for _ in range(ITERATIONS):
        latency, usage = call_once(True, "What is the capital of France?")
        cached.append(latency)
        cached_usage.append(usage)

    # Final audit: p50/p99 by variant + p99 delta (caching benefit).
    audit = {
        "modelId": MODEL_ID,
        "region": AWS_REGION,
        "iterations": ITERATIONS,
        "noCache": {
            "p50Ms": percentile(no_cache, 50),
            "p99Ms": percentile(no_cache, 99),
            "meanMs": round(statistics.fmean(no_cache), 2),
        },
        "withCache": {
            "p50Ms": percentile(cached, 50),
            "p99Ms": percentile(cached, 99),
            "meanMs": round(statistics.fmean(cached), 2),
        },
        "deltaP99Ms": round(percentile(no_cache, 99) - percentile(cached, 99), 2),
        "sampleUsage": {"noCache": no_cache_usage[0], "withCache": cached_usage[0]},
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
