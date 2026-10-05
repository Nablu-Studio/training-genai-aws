#!/usr/bin/env python3
"""Demonstrates IAM security boundaries: executes allowed and denied actions and logs the evaluation verdict.

Parent lab: iam-security-genai-201.
AWS Services: Bedrock (list), Secrets Manager (get), IAM (create_user) - via boto3.
Generated artifacts: iam/deny-test-audit-log.json (cas, résumé, résultat global ok/mismatch).
Mode: live (chaque appel peut échouer volontairement, c'est le but du test).
"""
import json
import os
import sys
import traceback
from pathlib import Path

import boto3
from botocore.exceptions import ClientError, EndpointConnectionError, NoCredentialsError

# Distinguishable failure modes when the call fails.
AWS_ERROR_HINTS = {
    "AccessDenied": "Droits insuffisants sur IAM : il faut iam:GetRole ou équivalent.",
    "AccessDeniedException": "Insufficient permissions: check the IAM policy of the role or current identity.",
    "ConflictException": "Une ressource du même nom existe déjà : supprimez-la ou changez de nom.",
    "DecryptionFailure": "Déchiffrement impossible : vérifiez les droits sur la clé KMS du secret.",
    "NoSuchEntity": "Rôle ou policy introuvable : vérifiez le nom exact.",
    "ResourceNotFoundException": "Secret introuvable : vérifiez son nom et la région.",
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

AWS_REGION = os.environ.get("AWS_REGION", "eu-west-1")
OUTPUT_PATH = Path("iam/deny-test-audit-log.json")
EXPECTED_DENIES = {
    "secretsmanager:GetSecretValue": "AccessDenied",
    "iam:CreateUser": "AccessDenied",
}
EXPECTED_ALLOWS = {
    "bedrock:ListFoundationModels": "ok",
}


def try_action(client, service: str, action_label: str, fn) -> dict:
    # Encapsule l'appel pour normaliser allowed / denied / error dans le même format.
    try:
        result = fn(client)
        return {
            "action": action_label,
            "service": service,
            "result": "allowed",
            "evidence": result,
        }
    except client.exceptions.AccessDeniedException:  # type: ignore[attr-defined]
        return {
            "action": action_label,
            "service": service,
            "result": "denied",
            "errorCode": "AccessDenied",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "action": action_label,
            "service": service,
            "result": "error",
            "errorCode": type(exc).__name__,
            "errorMessage": str(exc),
            "trace": traceback.format_exc(limit=1).strip().splitlines()[-1],
        }


def main() -> None:
    # Les 3 clients sont créés à partir des credentials actifs de l'environnement.
    bedrock = boto3.client("bedrock", region_name=AWS_REGION)
    secretsmanager = boto3.client("secretsmanager", region_name=AWS_REGION)
    iam = boto3.client("iam", region_name=AWS_REGION)

    cases: list[dict] = [
        # Positive case: expect `allowed`.
        try_action(
            bedrock,
            "bedrock",
            "bedrock:ListFoundationModels",
            lambda client: [
                model["modelId"]
                for model in client.list_foundation_models().get("modelSummaries", [])
            ][:3],
        ),
        # Negative cases: expect `denied` (IAM boundary enforcement).
        try_action(
            secretsmanager,
            "secretsmanager",
            "secretsmanager:GetSecretValue",
            lambda client: client.get_secret_value(SecretId="nablu/training/never-exists"),
        ),
        try_action(
            iam,
            "iam",
            "iam:CreateUser",
            lambda client: client.create_user(UserName="nablu-training-never-created"),
        ),
    ]

    summary = {
        "denyTests": [
            case for case in cases if case["action"] in EXPECTED_DENIES
        ],
        "allowTests": [
            case for case in cases if case["action"] in EXPECTED_ALLOWS
        ],
    }
    # Verdict global : tous les denies constatés et tous les allows constatés.
    passed = all(
        case["result"] == "denied" and case.get("errorCode") == EXPECTED_DENIES[case["action"]]
        for case in summary["denyTests"]
    ) and all(
        case["result"] == "allowed" for case in summary["allowTests"]
    )
    audit_log = {
        "region": AWS_REGION,
        "cases": cases,
        "summary": summary,
        "result": "ok" if passed else "mismatch",
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
