"""Invokes the SageMaker JumpStart endpoint deployed in step 1, then tears it down.

Parent lab: sagemaker-jumpstart-genai-201.
AWS Services: Amazon SageMaker (runtime, endpoints).
Generated artifacts: jumpstart/audit/invoke-audit.json (réponse, latence, teardown).
Mode: live — l'invocation et la suppression touchent le compte AWS.

Le teardown est actif par default: un endpoint oublié se facture à l'heure,
24/7. C'est la suppression qu'il faut choisir de désactiver, pas l'inverse.

Environment variables:
  AWS_REGION                   région de l'endpoint (default eu-west-1)
  TRAINING_JUMPSTART_PROMPT    prompt envoyé au modèle
  TRAINING_JUMPSTART_TEARDOWN  0 pour garder l'endpoint (default 1 : supprimer)
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
    "AccessDeniedException": "Droits insuffisants : le rôle doit pouvoir invoquer puis supprimer l'endpoint SageMaker.",
    "ModelError": "Le modèle a rejeté le payload : comparez-le au payloadFormat du catalogue.",
    "ResourceNotFound": "Endpoint introuvable : il a peut-être déjà été supprimé, ou vous n'êtes pas dans la bonne région.",
    "ServiceUnavailable": "Endpoint pas encore prêt : un cold start JumpStart prend 30 à 90 secondes.",
    "ValidationException": "Requête refusée : un paramètre est invalide, relisez le message ci-dessous.",
}


def explain_aws_error(exc: ClientError) -> str:
    """Translates an AWS ClientError into an actionable diagnostic message for the learner."""
    error = exc.response["Error"]
    code = error.get("Code", "Unknown")
    hint = AWS_ERROR_HINTS.get(code, "Unexpected AWS error: read details below.")
    return f"[{code}] {hint}\nAWS Details: {error.get('Message', '')}"


AWS_REGION = os.environ.get("AWS_REGION", "eu-west-1")
CATALOG_PATH = Path("jumpstart/data/model-catalog.json")
DEPLOY_AUDIT_PATH = Path("jumpstart/audit/deploy-audit.json")
OUTPUT_PATH = Path("jumpstart/audit/invoke-audit.json")
PROMPT = os.environ.get(
    "TRAINING_JUMPSTART_PROMPT",
    "Explique en trois phrases ce qu'est le retrieval augmented generation.",
)


def build_payload(model: dict) -> dict:
    """Expected payload differs by task: text generation vs vector embeddings."""
    template = model["payloadFormat"]
    if isinstance(template.get("inputs"), list):
        return {"inputs": [PROMPT]}
    return {"inputs": PROMPT, "parameters": dict(template.get("parameters", {}))}


def invoke(endpoint_name: str, payload: dict) -> dict:
    runtime = boto3.client("sagemaker-runtime", region_name=AWS_REGION)
    started = time.perf_counter()
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=json.dumps(payload),
    )
    body = json.loads(response["Body"].read())
    return {
        "latencyMs": round((time.perf_counter() - started) * 1000, 2),
        # La forme de la réponse dépend du modèle : on garde un extrait lisible.
        "response": json.dumps(body, ensure_ascii=False)[:800],
    }


def teardown(endpoint_name: str) -> list[dict]:
    """Order of resource cleanup: endpoint, endpoint config, model."""
    sagemaker = boto3.client("sagemaker", region_name=AWS_REGION)
    results = []
    for label, call in (
        ("endpoint", lambda: sagemaker.delete_endpoint(EndpointName=endpoint_name)),
        ("endpoint-config", lambda: sagemaker.delete_endpoint_config(EndpointConfigName=endpoint_name)),
        ("model", lambda: sagemaker.delete_model(ModelName=endpoint_name)),
    ):
        try:
            call()
            results.append({"resource": label, "deleted": True})
        except ClientError as exc:
            # Une ressource déjà absente n'est pas un échec : on continue la
            # suppression des deux autres, sinon un oubli reste facturé.
            results.append(
                {"resource": label, "deleted": False, "error": exc.response["Error"].get("Code", "Unknown")}
            )
    return results


def main() -> None:
    deploy_audit = json.loads(DEPLOY_AUDIT_PATH.read_text(encoding="utf-8"))
    endpoint_name = deploy_audit["endpointName"]

    if deploy_audit.get("dryRun"):
        sys.exit(
            f"{DEPLOY_AUDIT_PATH} est un dry-run : aucun endpoint n'existe. "
            "Relancez deploy_endpoint.py avec TRAINING_JUMPSTART_DEPLOY=1 avant cette étape."
        )

    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    model = next(entry for entry in catalog if entry["modelId"] == deploy_audit["modelId"])
    will_teardown = bool(int(os.environ.get("TRAINING_JUMPSTART_TEARDOWN", "1")))

    audit = {
        "endpointName": endpoint_name,
        "modelId": model["modelId"],
        "instanceType": deploy_audit["instance"],
        "region": AWS_REGION,
        "prompt": PROMPT,
        "invocation": invoke(endpoint_name, build_payload(model)),
        "teardownSteps": [
            f"aws sagemaker delete-endpoint --endpoint-name {endpoint_name} --region {AWS_REGION}",
            f"aws sagemaker delete-endpoint-config --endpoint-config-name {endpoint_name} --region {AWS_REGION}",
            f"aws sagemaker delete-model --model-name {endpoint_name} --region {AWS_REGION}",
        ],
    }
    audit["teardown"] = (
        teardown(endpoint_name)
        if will_teardown
        else [{"resource": "all", "deleted": False, "error": "TRAINING_JUMPSTART_TEARDOWN=0"}]
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    if not will_teardown:
        print(
            "\nEndpoint conservé : il est facturé tant qu'il existe. "
            "Exécutez les trois commandes teardownSteps quand vous avez fini.",
            file=sys.stderr,
        )


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
