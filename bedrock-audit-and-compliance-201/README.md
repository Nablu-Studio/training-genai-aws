# Bedrock Audit & Compliance 201 — Runbook

## 1. Pré-requis

- Compte AWS avec CloudTrail, Config, et Bedrock actifs.
- `boto3`.

## 2. Installation

```bash
cd training/aws/examples/bedrock-audit-and-compliance-201
python3 -m venv .venv && source .venv/bin/activate
pip install boto3
```

## 3. Etape 1 — Inspecter CloudTrail et Config

```bash
cat audit/data/trail-config.json
export AWS_REGION=eu-west-1
python3 audit/scripts/inspect_trail.py
cat audit/audit-trail.json
```

## 4. Etape 2 — Rapport d audit de gouvernance

```bash
python3 audit/scripts/audit_governance.py
cat audit/audit-governance.json
```

## 5. Anti-patterns

- Activer CloudTrail data events sur tout AWS : cout exorbitant. Limiter a Bedrock, KMS, S3 sensibles.
- Oublier KMS sur le bucket CloudTrail : logs en clair, non conformes RGPD/SOC2.
- Auditer sans invariant clair : le rapport ne produit rien d exploitable.
