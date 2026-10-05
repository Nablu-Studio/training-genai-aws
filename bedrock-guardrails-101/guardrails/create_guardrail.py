"""Provisions an Amazon Bedrock Guardrail from a JSON policy and publishes an immutable version.

Parent lab: bedrock-guardrails-101.
AWS Services: Amazon Bedrock control plane (create_guardrail, create_guardrail_version).
Generated artifacts: guardrails/guardrail-runtime-config.json (ID, version, contrôle actif).
Mode: live (création effective du guardrail facturable).
"""
import json
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError, EndpointConnectionError, NoCredentialsError

# Distinguishable failure modes when the call fails.
AWS_ERROR_HINTS = {
    "AccessDeniedException": "Insufficient permissions: check the IAM policy of the role or current identity.",
    "ConflictException": "Une ressource du même nom existe déjà : supprimez-la ou changez de nom.",
    "ResourceNotFoundException": "Ressource Bedrock introuvable : vérifiez l'identifiant et la région.",
    "ServiceQuotaExceededException": "Quota de service atteint : demandez une augmentation ou libérez une ressource.",
    "ThrottlingException": "Quota exceeded: reduce call rate or retry with exponential backoff.",
    "ValidationException": "Requête refusée : un paramètre est invalide, relisez le message ci-dessous.",
}


def explain_aws_error(exc: ClientError) -> str:
    """Translates an AWS ClientError into an actionable diagnostic message for the learner."""
    error = exc.response["Error"]
    code = error.get("Code", "Unknown")
    hint = AWS_ERROR_HINTS.get(code, "Unexpected AWS error: read details below.")
    return f"[{code}] {hint}\nAWS Details: {error.get('Message', '')}"


AWS_REGION = "eu-west-1"
GUARDRAIL_NAME = "training-guardrails-101"
MODEL_ID = "amazon.nova-lite-v1:0"
POLICY_PATH = Path("guardrails/config/guardrail-policy.json")
CONTROL_MATRIX_PATH = Path("guardrails/guardrails-control-matrix.json")
OUTPUT_PATH = Path("guardrails/guardrail-runtime-config.json")


def load_policy() -> dict:
    return json.loads(POLICY_PATH.read_text(encoding="utf-8"))


def load_control_matrix() -> list[dict]:
    return json.loads(CONTROL_MATRIX_PATH.read_text(encoding="utf-8"))


def select_runtime_control(control_matrix: list[dict]) -> dict:
    # Premier contrôle "runtime" trouvé, sinon fallback sur le premier de la matrice.
    return next(
        (item for item in control_matrix if item["control"] == "bedrock_guardrail_runtime"),
        control_matrix[0],
    )


def build_create_payload(policy: dict) -> dict:
    # Mapping policy JSON -> schéma create_guardrail Bedrock (filtres + topics + PII + mots).
    return {
        "name": policy.get("name", GUARDRAIL_NAME),
        "description": policy.get(
            "description",
            "Hands-on Bedrock Guardrails for certification training",
        ),
        "blockedInputMessaging": policy.get(
            "blockedInputMessaging",
            "Votre demande a ete bloquee par le guardrail.",
        ),
        "blockedOutputsMessaging": policy.get(
            "blockedOutputsMessaging",
            "La reponse a ete bloquee par le guardrail.",
        ),
        "contentPolicyConfig": {
            "filtersConfig": [
                {
                    "type": item["type"],
                    "inputStrength": item["inputStrength"],
                    "outputStrength": item["outputStrength"],
                }
                for item in policy.get("contentFilters", [])
            ]
        },
        "topicPolicyConfig": {
            "topicsConfig": [
                {
                    "name": topic["name"],
                    "definition": topic["definition"],
                    "examples": topic.get("examples", []),
                    "type": topic.get("action", "DENY"),
                }
                for topic in policy.get("deniedTopics", [])
            ]
        },
        "sensitiveInformationPolicyConfig": {
            "piiEntitiesConfig": [
                {
                    "type": entity["type"],
                    "action": entity["action"],
                }
                for entity in policy.get("sensitiveInformationPolicy", {}).get(
                    "piiEntities",
                    [],
                )
            ]
        },
        "wordPolicyConfig": {
            "wordsConfig": [
                {"text": word}
                for word in policy.get("wordPolicy", {}).get("customWords", [])
            ]
        },
    }


def main() -> None:
    # 1) Load policy + matrix, 2) create guardrail, 3) publish version.
    policy = load_policy()
    control_matrix = load_control_matrix()
    active_control = select_runtime_control(control_matrix)
    client = boto3.client("bedrock", region_name=AWS_REGION)

    create_response = client.create_guardrail(**build_create_payload(policy))
    version_response = client.create_guardrail_version(
        guardrailIdentifier=create_response["guardrailId"],
        description="Published version for the Guardrails 101 hands-on",
    )

    # L'output runtime_config sert de fichier d'entrée à `check_request.py`.
    runtime_config = {
        "region": AWS_REGION,
        "modelId": MODEL_ID,
        "guardrailIdentifier": create_response["guardrailId"],
        "guardrailVersion": version_response["version"],
        "policyPath": str(POLICY_PATH),
        "controlMatrixPath": str(CONTROL_MATRIX_PATH),
        "activeControl": active_control,
        "testPrompt": active_control["testPrompt"],
    }
    OUTPUT_PATH.write_text(json.dumps(runtime_config, indent=2), encoding="utf-8")
    print(json.dumps(runtime_config, indent=2))


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
