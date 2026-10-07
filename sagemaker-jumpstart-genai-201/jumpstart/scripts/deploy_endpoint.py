"""Prepares and provisions deployment of a SageMaker JumpStart foundation model.

Parent lab: sagemaker-jumpstart-genai-201.
AWS Services: Amazon SageMaker (JumpStart, endpoints).
Generated artifacts: jumpstart/audit/deploy-audit.json (plan, statut, teardown).
Mode : dry-run par défaut (TRAINING_JUMPSTART_DEPLOY=0).

Un endpoint JumpStart se facture à l'heure tant qu'il tourne : le script
n'engage aucune dépense sans un TRAINING_JUMPSTART_DEPLOY=1 explicite, et
rappelle systématiquement les trois commandes de teardown.

Environment variables:
  AWS_REGION                    région du déploiement (default eu-west-1)
  TRAINING_JUMPSTART_MODEL_ID   modèle du catalogue (default : le premier)
  TRAINING_JUMPSTART_INSTANCE   type d'instance (default : celui du catalogue)
  TRAINING_JUMPSTART_DEPLOY     1 pour déployer réellement (default 0)
  TRAINING_JUMPSTART_ACCEPT_EULA 1 si le modèle exige d'accepter une licence
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
    "AccessDeniedException": "Droits insuffisants : le rôle doit pouvoir créer un modèle, une endpoint-config et un endpoint SageMaker.",
    "ResourceLimitExceeded": "Quota d'instances atteint pour ce type : demandez une augmentation ou choisissez une instance plus petite.",
    "ResourceNotFound": "Modèle ou endpoint introuvable : vérifiez l'identifiant et la région.",
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
CATALOG_PATH = Path("jumpstart/data/model-catalog.json")
OUTPUT_PATH = Path("jumpstart/audit/deploy-audit.json")
POLL_TIMEOUT_S = int(os.environ.get("TRAINING_JUMPSTART_TIMEOUT", "1800"))


def select_model(catalog: list[dict]) -> dict:
    wanted = os.environ.get("TRAINING_JUMPSTART_MODEL_ID")
    if not wanted:
        return catalog[0]
    for entry in catalog:
        if entry["modelId"] == wanted:
            return entry
    sys.exit(
        f"Modèle « {wanted} » absent de {CATALOG_PATH} : "
        f"choisissez parmi {', '.join(e['modelId'] for e in catalog)}."
    )


def wait_in_service(sagemaker, endpoint_name: str) -> dict:
    """Status polling: endpoints take several minutes to transition to InService."""
    deadline = time.time() + POLL_TIMEOUT_S
    while time.time() < deadline:
        described = sagemaker.describe_endpoint(EndpointName=endpoint_name)
        status = described["EndpointStatus"]
        if status in ("InService", "Failed"):
            return {"status": status, "failureReason": described.get("FailureReason")}
        print(f"  statut {status}, nouvelle vérification dans 30 s...", file=sys.stderr)
        time.sleep(30)
    return {"status": "Timeout", "failureReason": f"toujours en cours après {POLL_TIMEOUT_S} s"}


def deploy(model: dict, instance: str, endpoint_name: str) -> dict:
    # Lazy import: SageMaker SDK is only required when live deployment is requested.
    from sagemaker.jumpstart.model import JumpStartModel

    started = time.perf_counter()
    jumpstart_model = JumpStartModel(model_id=model["modelId"], region=AWS_REGION)
    jumpstart_model.deploy(
        initial_instance_count=1,
        instance_type=instance,
        endpoint_name=endpoint_name,
        accept_eula=bool(int(os.environ.get("TRAINING_JUMPSTART_ACCEPT_EULA", "0"))),
    )
    result = wait_in_service(boto3.client("sagemaker", region_name=AWS_REGION), endpoint_name)
    return {**result, "durationMs": round((time.perf_counter() - started) * 1000, 2)}


def main() -> None:
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    model = select_model(catalog)
    instance = os.environ.get("TRAINING_JUMPSTART_INSTANCE", model["instance"])
    endpoint_name = os.environ.get("TRAINING_JUMPSTART_ENDPOINT", f"training-{model['modelId'][:40]}")
    will_deploy = bool(int(os.environ.get("TRAINING_JUMPSTART_DEPLOY", "0")))

    audit = {
        "modelId": model["modelId"],
        "instance": instance,
        "region": AWS_REGION,
        "license": model["license"],
        "endpointName": endpoint_name,
        "dryRun": not will_deploy,
        # Always logged: an orphaned SageMaker endpoint incurs hourly charges.
        "teardownSteps": [
            f"aws sagemaker delete-endpoint --endpoint-name {endpoint_name} --region {AWS_REGION}",
            f"aws sagemaker delete-endpoint-config --endpoint-config-name {endpoint_name} --region {AWS_REGION}",
            f"aws sagemaker delete-model --model-name {endpoint_name} --region {AWS_REGION}",
        ],
    }

    if will_deploy:
        audit["deployment"] = deploy(model, instance, endpoint_name)
    else:
        audit["note"] = (
            "Dry-run : aucune ressource créée. Relancez avec TRAINING_JUMPSTART_DEPLOY=1 "
            "pour déployer, en sachant que l'instance est facturée tant que l'endpoint existe."
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))


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
    except ImportError:
        sys.exit("Le déploiement réel exige la SDK SageMaker : pip install sagemaker")
