# Amazon Q 101 — Runbook

## 1. Pré-requis

- Compte AWS avec IAM Identity Center configure.
- `boto3`.

## 2. Installation

```bash
cd training/aws/examples/amazon-q-101
python3 -m venv .venv && source .venv/bin/activate
pip install boto3
```

## 3. Etape 1 — Lire la matrice produit

```bash
cat q/data/product-matrix.json
```

## 4. Etape 2 — Lancer un chat Q Business

```bash
export AWS_REGION=eu-west-1
export TRAINING_QBIZ_APPLICATION_ID="<votre-application-id>"
export TRAINING_QBIZ_USER_ID="<votre-user-id>"
python3 q/scripts/query_qbusiness.py
cat q/audit/qbusiness-query-audit.json
```

Sans `APPLICATION_ID` le script ecrit un audit `skipped: true` documentant la configuration requise.

## 5. Anti-patterns

- Utiliser Q Business pour de l assistance code : pas adapte.
- Indexer des donnees personnelles sans cadre RGPD.
- Oublier IAM Identity Center : Q Business refuse de creer l application.
