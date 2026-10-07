# Provisioned Throughput & Pricing 201 — Runbook

> **Prix d'exercice.** Les valeurs de `pricing/data/model-pricing.json` sont illustratives et les
> modèles cités sont datés. Relevez les tarifs actuels de votre région sur
> https://aws.amazon.com/bedrock/pricing/ avant de tirer une conclusion réelle du TCO.

## 1. Pré-requis

- Compte AWS avec `bedrock:CreateProvisionedModelThroughput`.
- `boto3`.

## 2. Installation

```bash
cd training/aws/examples/provisioned-throughput-and-pricing-201
python3 -m venv .venv && source .venv/bin/activate
pip install boto3
```

## 3. Etape 1 — Dimensionner la PT (dry run)

```bash
cat pricing/data/model-pricing.json
export TRAINING_PT_TARGET_TPM=30000
python3 pricing/scripts/buy_provisioned.py
cat pricing/audit/pt-provisioned-audit.json
```

Par defaut le script tourne en dry run. Pour acheter reellement :

```bash
export TRAINING_PT_DRY_RUN=0
python3 pricing/scripts/buy_provisioned.py
```

## 4. Etape 2 — Comparer les 3 strategies de TCO

```bash
python3 pricing/scripts/tco_compare.py
cat pricing/audit/tco-comparison.json
```

## 5. Anti-patterns

- Acheter de la PT sans calcul de break-even.
- Ignorer le ratio input/output dans le calcul on-demand.
- Reserver un modele qu on n a pas encore teste en production.
