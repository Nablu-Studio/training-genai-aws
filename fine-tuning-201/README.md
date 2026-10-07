# Fine-tuning 201 — Runbook

## 1. Pré-requis

- Compte AWS avec `bedrock:CreateModelCustomizationJob` et accès S3.
- Dataset JSON Lines conforme dans un bucket S3 (10 exemples par défaut dans `custom/datasets/training-samples.jsonl`).
- Role IAM de service Bedrock (fourniture via `TRAINING_BEDROCK_ROLE_ARN`).

## 2. Installation

```bash
cd training/aws/examples/fine-tuning-201
python3 -m venv .venv && source .venv/bin/activate
pip install boto3
```

## 3. Etape 1 — Valider le dataset

```bash
python3 custom/scripts/validate_dataset.py
cat custom/datasets/dataset-quality-report.json
```

Sortie attendue :

```json
{ "lineCount": 10, "issueCount": 0, "issues": [], "passed": true }
```

## 4. Etape 2 — Lancer la personnalisation

```bash
export AWS_REGION=eu-west-1
export TRAINING_DATASET_URI=s3://<votre-bucket>/fine-tune/training-samples.jsonl
export TRAINING_BEDROCK_ROLE_ARN=arn:aws:iam::<account>:role/<role-bedrock>
python3 custom/scripts/launch_customization.py
cat custom/audit/customization-audit.json
```

Le script orchestre : `create_model_customization_job` → poll → `create_provisioned_model_throughput` → `converse` inference.

## 5. Anti-patterns

- Personnaliser sans golden answers.
- Oublier la Provisioned Throughput après l'entraînement.
- Confondre fine-tuning et continued pre-training dans la console.
