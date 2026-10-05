"""Builds a redacted-friendly observability trace around an Amazon Bedrock invocation.

Parent lab: llm-observability-101.
AWS Services: Amazon Bedrock Runtime (converse).
Generated artifacts: observability/bedrock-trace.json (timestamp, latence, tokens, status).
Mode: live (un appel Bedrock de démonstration).
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

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


def redact_prompt(prompt: str) -> str:
    # Politique conservative : on ne log jamais le prompt brut côté observabilité.
    return "[redacted]" if prompt else ""


def build_trace(
    prompt_version: str,
    prompt: str,
    latency_ms: float,
    input_tokens: int,
    output_tokens: int,
) -> dict:
    return {
        # Horodatage UTC ISO pour ingestion par n'importe quel backend de logs.
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prompt_version": prompt_version,
        "prompt_preview": redact_prompt(prompt),
        "latency_ms": latency_ms,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "status": "success",
    }


def main() -> None:
    # Exemple d'appel : on mesure la latence et on enrichit la trace avec les métadonnées AWS.
    prompt = "Donne moi une synthese finance"
    client = boto3.client("bedrock-runtime", region_name="eu-west-1")
    started_at = perf_counter()
    response = client.converse(
        modelId="amazon.nova-lite-v1:0",
        messages=[{"role": "user", "content": [{"text": prompt}]}],
    )
    latency_ms = round((perf_counter() - started_at) * 1000, 2)
    trace = build_trace("v1", prompt, latency_ms, 321, 118)
    trace["request_id"] = response["ResponseMetadata"]["RequestId"]
    trace["stop_reason"] = response.get("stopReason")
    # Sérialisation pour exploitation par la UI de formation.
    Path("observability/bedrock-trace.json").write_text(
        json.dumps(trace, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(trace, indent=2))


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
