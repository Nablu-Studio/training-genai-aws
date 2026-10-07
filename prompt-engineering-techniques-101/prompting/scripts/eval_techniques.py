"""Compares prompt engineering techniques (zero-shot, few-shot, chain-of-thought) across a test dataset.

Parent lab: prompt-engineering-techniques-101.
AWS Services: Amazon Bedrock Runtime (converse).
Generated artifacts: prompting/audit/techniques-audit.json (score moyen par technique, détails).
Mode: live (un appel Bedrock par question x technique).
"""
import json
import os
import re
import sys
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
MODEL_ID = os.environ.get("TRAINING_PROMPTING_MODEL", "amazon.nova-lite-v1:0")
QUESTIONS_PATH = Path("prompting/data/questions.json")
OUTPUT_PATH = Path("prompting/audit/techniques-audit.json")

PROMPTS_DIR = Path("prompting/prompts")
SYSTEM_PROMPTS = {
    "zero_shot": (PROMPTS_DIR / "system-zero-shot.txt").read_text(encoding="utf-8"),
    "few_shot": (PROMPTS_DIR / "system-few-shot.txt").read_text(encoding="utf-8"),
    "chain_of_thought": (PROMPTS_DIR / "system-chain-of-thought.txt").read_text(encoding="utf-8"),
}

client = boto3.client("bedrock-runtime", region_name=AWS_REGION)


def call_model(system: str, question: str) -> str:
    # Température 0 = comparaison déterministe entre techniques.
    response = client.converse(
        modelId=MODEL_ID,
        system=[{"text": system}],
        messages=[{"role": "user", "content": [{"text": question}]}],
        inferenceConfig={"temperature": 0.0, "maxTokens": 256},
    )
    return "".join(
        block.get("text", "")
        for block in response["output"]["message"].get("content", [])
    ).strip()


def score(answer: str, golden: str) -> dict:
    # Simple evaluation heuristic: exact match first, normalized token overlap fallback.
    normalized = answer.lower()
    golden_l = golden.lower()
    if golden_l in normalized:
        return {"match": "contains", "score": 1.0}
    tokens_golden = set(re.findall(r"\w+", golden_l))
    tokens_answer = set(re.findall(r"\w+", normalized))
    if not tokens_golden:
        return {"match": "empty_golden", "score": 0.0}
    return {
        "match": "partial",
        "score": round(len(tokens_golden & tokens_answer) / len(tokens_golden), 3),
    }


def main() -> None:
    # Benchmark questions across techniques with score aggregation.
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    results: list[dict] = []
    aggregate: dict[str, list[float]] = {key: [] for key in SYSTEM_PROMPTS}
    for question in questions:
        for key, system in SYSTEM_PROMPTS.items():
            answer = call_model(system, question["question"])
            entry = score(answer, question["golden"])
            results.append(
                {
                    "technique": key,
                    "questionId": question["id"],
                    "answer": answer,
                    "score": entry["score"],
                    "match": entry["match"],
                }
            )
            aggregate[key].append(entry["score"])
    audit = {
        "modelId": MODEL_ID,
        "region": AWS_REGION,
        "techniques": list(SYSTEM_PROMPTS.keys()),
        "meanScore": {key: round(sum(values) / len(values), 3) for key, values in aggregate.items()},
        "results": results,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit["meanScore"], indent=2))


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
