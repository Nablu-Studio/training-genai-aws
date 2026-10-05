"""Orchestrates a Bedrock fine-tuning workflow: create job, poll status, provision throughput, and smoke test.

Parent lab: fine-tuning-201.
AWS Services: Amazon Bedrock (create/get_model_customization_job, create_provisioned_model_throughput, converse).
Generated artifacts: custom/audit/customization-audit.json (jobArn, statut, modèle, PT, inférence).
Mode: live (coûteux : lancement d'un job d'entraînement Bedrock).
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
    "AccessDeniedException": "Access denied: either the IAM policy lacks bedrock:InvokeModel, or model access is not enabled (Bedrock console > Model access).",
    "ConflictException": "Une ressource du même nom existe déjà : supprimez-la ou changez de nom.",
    "ModelTimeoutException": "Model timed out: reduce prompt size or max tokens.",
    "ResourceNotFoundException": "Ressource introuvable dans cette région : tous les modèles ne sont pas disponibles partout.",
    "ServiceQuotaExceededException": "Quota de service atteint : demandez une augmentation ou libérez une ressource.",
    "ThrottlingException": "Quota exceeded: reduce call rate or retry with exponential backoff.",
    "ValidationException": "Request validation error: invalid model ID or model requires an inference profile./us./global.) plutôt que l'ID direct.",
}


def explain_aws_error(exc: ClientError) -> str:
    """Translates an AWS ClientError into an actionable diagnostic message for the learner."""
    error = exc.response["Error"]
    code = error.get("Code", "Unknown")
    hint = AWS_ERROR_HINTS.get(code, "Unexpected AWS error: read details below.")
    return f"[{code}] {hint}\nAWS Details: {error.get('Message', '')}"


AWS_REGION = os.environ.get("AWS_REGION", "eu-west-1")
JOB_NAME = os.environ.get("TRAINING_CUSTOMIZATION_JOB", "nablu-training-fine-tune")
BASE_MODEL = os.environ.get("TRAINING_BASE_MODEL", "amazon.nova-lite-v1:0")
DATASET_URI = os.environ.get(
    "TRAINING_DATASET_URI",
    "s3://nablu-training-data/fine-tune/training-samples.jsonl",
)
ROLE_ARN = os.environ.get(
    "TRAINING_BEDROCK_ROLE_ARN",
    "arn:aws:iam::123456789012:role/nablu-training-bedrock-execution",
)
OUTPUT_PATH = Path("custom/audit/customization-audit.json")


def launch_job() -> str:
    # Step 1: create_model_customization_job avec hyperparams minimales (3 epochs, lr 1e-4).
    bedrock = boto3.client("bedrock", region_name=AWS_REGION)
    response = bedrock.create_model_customization_job(
        jobName=JOB_NAME,
        customModelName=f"{JOB_NAME}-model",
        roleArn=ROLE_ARN,
        baseModelIdentifier=BASE_MODEL,
        trainingDataConfig={"s3Uri": DATASET_URI},
        outputDataConfig={"s3Uri": f"s3://nablu-training-data/fine-tune/{JOB_NAME}/"},
        hyperParameters={"epochCount": "3", "batchSize": "1", "learningRate": "0.0001"},
    )
    return response["jobArn"]


def wait_for_job(job_arn: str) -> dict:
    # Step 2: polling get_model_customization_job toutes les 30s, plafond 30 minutes.
    bedrock = boto3.client("bedrock", region_name=AWS_REGION)
    response: dict = {}
    for _ in range(60):
        response = bedrock.get_model_customization_job(jobIdentifier=job_arn)
        if response["status"] in {"Completed", "Failed", "Stopped"}:
            return response
        time.sleep(30)
    return response


def create_provisioned_throughput(model_arn: str) -> dict:
    # Étape 3 (optionnelle) : achat d'un PT pour utiliser le modèle custom en production.
    bedrock = boto3.client("bedrock", region_name=AWS_REGION)
    return bedrock.create_provisioned_model_throughput(
        modelUnits=1,
        provisionedModelName=f"{JOB_NAME}-pt",
        modelId=model_arn,
    )


def invoke_inference(model_arn: str) -> dict:
    # Step 4: smoke test du modèle customisé via converse.
    runtime = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    response = runtime.converse(
        modelId=model_arn,
        messages=[{"role": "user", "content": [{"text": "Resumer la personnalite GenAI attendue."}]}],
    )
    answer = "".join(
        block.get("text", "")
        for block in response["output"]["message"].get("content", [])
        if isinstance(block, dict)
    ).strip()
    return {
        "answer": answer,
        "stopReason": response.get("stopReason"),
        "requestId": response["ResponseMetadata"]["RequestId"],
    }


def main() -> None:
    # Orchestration séquentielle : job -> PT -> inférence conditionnelle.
    job_arn = launch_job()
    job = wait_for_job(job_arn)
    model_arn = job.get("outputModelArn", "")
    # Étapes 3-4 conditionnelles : on ne tente le PT et l'inférence que si l'entraînement a réussi.
    pt = create_provisioned_throughput(model_arn) if model_arn else {}
    inference = invoke_inference(model_arn) if model_arn else {}
    audit = {
        "region": AWS_REGION,
        "jobArn": job_arn,
        "jobStatus": job.get("status"),
        "baseModel": BASE_MODEL,
        "outputModelArn": model_arn,
        "provisionedThroughput": {
            "name": pt.get("provisionedModelName"),
            "modelUnits": pt.get("modelUnits"),
            "status": pt.get("status"),
        },
        "inference": inference,
    }
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
