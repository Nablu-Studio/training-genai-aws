"""Evaluates Bedrock prompt completions against a reference rubric using Jaccard token overlap.

Parent lab: evals-llm-101.
AWS Services: Amazon Bedrock Runtime (converse).
Generated artifacts: evals/eval_report_example.json (par cas : id, catégorie, score, latence).
Mode: live (un appel Bedrock par cas du dataset).
"""
import json
import sys
from pathlib import Path
from time import perf_counter

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


def load_cases(path: str) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def score_case(answer: str, expected: str) -> float:
    # Score = |tokens réponse ∩ tokens attendus| / |tokens attendus|.
    answer_tokens = set(answer.lower().split())
    expected_tokens = set(expected.lower().split())
    if not expected_tokens:
        return 0.0
    return round(len(answer_tokens & expected_tokens) / len(expected_tokens), 4)


def main() -> None:
    # Load scoring rubric, evaluate each case, and aggregate into JSON audit report.
    client = boto3.client("bedrock-runtime", region_name="eu-west-1")
    cases = load_cases("evals/datasets/rubric.json")
    report = []
    for case in cases:
        # Latence par cas pour identifier les outliers dans le dataset.
        started_at = perf_counter()
        response = client.converse(
            modelId="amazon.nova-lite-v1:0",
            messages=[{"role": "user", "content": [{"text": case["input"]}]}],
        )
        latency_ms = round((perf_counter() - started_at) * 1000, 2)
        answer = response["output"]["message"]["content"][0]["text"]
        report.append(
            {
                "id": case["id"],
                "category": case["category"],
                "score": score_case(answer, case["expected"]),
                "latency_ms": latency_ms,
            }
        )
    Path("evals/eval_report_example.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report[:2], indent=2))


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
