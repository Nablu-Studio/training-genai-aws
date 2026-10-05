"""Captures an observability trace (latency, request_id, token preview) following a Bedrock call.

Parent lab: observability-llm-201.
AWS Services: Amazon Bedrock Runtime (converse).
Generated artifacts: observability/redacted-trace.json (résumé d'appel anonymisé).
Mode: live (un seul appel Bedrock).
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



client = boto3.client("bedrock-runtime", region_name="eu-west-1")


def main() -> None:
    # Mesure de la latence totale de bout en bout, utile pour le dashboard.
    started_at = perf_counter()
    response = client.converse(
        modelId="amazon.nova-lite-v1:0",
        messages=[{"role": "user", "content": [{"text": "Resume la politique d'acces en 2 points."}]}],
    )
    # Trace minimaliste : identifiant AWS, raison d'arrêt, aperçu tronqué du contenu.
    trace = {
        "latency_ms": round((perf_counter() - started_at) * 1000, 2),
        "request_id": response["ResponseMetadata"]["RequestId"],
        "stop_reason": response.get("stopReason"),
        "output_preview": response["output"]["message"]["content"][0]["text"][:120],
    }
    # Écriture de la trace pour exploitation par la UI de formation.
    Path("observability/redacted-trace.json").write_text(
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
