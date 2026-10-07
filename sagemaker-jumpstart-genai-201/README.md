# SageMaker JumpStart GenAI 201 — Runbook

## 1. Pré-requis

- Compte AWS avec SageMaker Studio ou un role autorise.
- `boto3` + `sagemaker` (haut niveau).

## 2. Installation

```bash
cd training/aws/examples/sagemaker-jumpstart-genai-201
python3 -m venv .venv && source .venv/bin/activate
pip install boto3 sagemaker
```

## 3. Etape 1 — Lire le catalogue

```bash
cat jumpstart/data/model-catalog.json
aws sagemaker list-endpoints --region eu-west-1
```

## 4. Etape 2 — Deployer, invoquer, teardown

```bash
export AWS_REGION=eu-west-1
export TRAINING_JUMPSTART_MODEL_ID=meta-textgeneration-llama-2-7b-f
export TRAINING_JUMPSTART_INSTANCE=ml.g5.2xlarge
python3 jumpstart/scripts/invoke_and_teardown.py
cat jumpstart/audit/invoke-audit.json
```

Le script documente la config ciblee et les 3 appels de teardown obligatoire. Pour deployer reellement, utilisez `sagemaker.jumpstart.model.JumpStartModel`.

## 5. Anti-patterns

- Oublier le teardown : facture a l heure qui peut atteindre des centaines de dollars par jour.
- Deployer un modele trop gros (70B+) sur une instance trop petite : OOM et crash.
- Invoquer avec un payload non adapte au modele : erreur 400 opaque.
