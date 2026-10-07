"""Provisions (or simulates) Bedrock Provisioned Throughput capacity sized for target traffic.

Parent lab: provisioned-throughput-and-pricing-201.
AWS Services: Amazon Bedrock control plane (create_provisioned_model_throughput).
Generated artifacts: pricing/audit/pt-provisioned-audit.json (MU, coût mensuel, ARNProvisioned).
Mode : dry-run par défaut (TRAINING_PT_DRY_RUN=1), achat réel sinon.
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
    "ConflictException": "Resource with the same name already exists: delete it or choose another name.",
    "ResourceNotFoundException": "Bedrock resource not found: verify identifier and region.",
    "ServiceQuotaExceededException": "Service quota exceeded: request a quota increase or release unused resources.",
    "ThrottlingException": "Quota exceeded: reduce call rate or retry with exponential backoff.",
    "ValidationException": "Request validation error: verify parameter format in details below.",
}


def explain_aws_error(exc: ClientError) -> str:
    """Translates an AWS ClientError into an actionable diagnostic message for the learner."""
    error = exc.response["Error"]
    code = error.get("Code", "Unknown")
    hint = AWS_ERROR_HINTS.get(code, "Unexpected AWS error: read details below.")
    return f"[{code}] {hint}\nAWS Details: {error.get('Message', '')}"


AWS_REGION = os.environ.get("AWS_REGION", "eu-west-1")
PROVISIONED_NAME = os.environ.get("TRAINING_PT_NAME", "nablu-training-pt")
TARGET_MODEL = os.environ.get("TRAINING_PT_MODEL", "amazon.nova-lite-v1:0")
TARGET_TPM = int(os.environ.get("TRAINING_PT_TARGET_TPM", "30000"))
OUTPUT_PATH = Path("pricing/audit/pt-provisioned-audit.json")
PRICING_PATH = Path("pricing/data/model-pricing.json")


def size_for_traffic(model: dict, target_tokens_per_min: int) -> int:
    # One model unit = `muCapacityTokensPerMin` tokens/min: size precisely to target throughput.
    return max(1, round(target_tokens_per_min / model["muCapacityTokensPerMin"]))


def main() -> None:
    # Sizing calculation: required model units and projected monthly cost (24h * 30d).
    pricing = json.loads(PRICING_PATH.read_text(encoding="utf-8"))
    model = next(item for item in pricing if item["modelId"] == TARGET_MODEL)
    model_units = size_for_traffic(model, TARGET_TPM)
    monthly_cost = round(model_units * model["ptPricePerMuHour"] * 24 * 30, 2)

    dry_run = bool(int(os.environ.get("TRAINING_PT_DRY_RUN", "1")))
    if dry_run:
        # Default dry-run branch: no AWS API call; records purchase intent to audit log.
        audit = {
            "dryRun": True,
            "modelId": TARGET_MODEL,
            "modelUnits": model_units,
            "monthlyCostUsd": monthly_cost,
            "note": "Export TRAINING_PT_DRY_RUN=0 et lancez avec un role autorise pour acheter reellement.",
        }
    else:
        # Live branch: provision Provisioned Throughput allocation.
        bedrock = boto3.client("bedrock", region_name=AWS_REGION)
        response = bedrock.create_provisioned_model_throughput(
            modelUnits=model_units,
            provisionedModelName=PROVISIONED_NAME,
            modelId=TARGET_MODEL,
        )
        audit = {
            "dryRun": False,
            "modelId": TARGET_MODEL,
            "modelUnits": model_units,
            "monthlyCostUsd": monthly_cost,
            "provisionedModelName": response.get("provisionedModelName"),
            "provisionedModelArn": response.get("provisionedModelArn"),
            "status": response.get("status"),
            "requestId": response["ResponseMetadata"]["RequestId"],
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
