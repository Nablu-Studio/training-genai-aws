# RAG Advanced 201 — Runbook

## 1. Pré-requis

- Compte AWS avec Bedrock et `cohere.rerank-v3-5:0` + `amazon.titan-embed-image-v1` + `amazon.nova-lite-v1:0` actives.
- `boto3`.

## 2. Installation

```bash
cd training/aws/examples/rag-advanced-201
python3 -m venv .venv && source .venv/bin/activate
pip install boto3
```

## 3. Etape 1 — Re-ranking

```bash
cat rag/data/tech-corpus.json
python3 rag/scripts/rerank.py
cat rag/audit/rerank-audit.json
```

## 4. Etape 2 — RAG multi-modal

```bash
export AWS_REGION=eu-west-1
export TRAINING_RAG_IMAGE_PATH=rag/data/sample-diagram.png
python3 rag/scripts/multimodal_rag.py
cat rag/audit/multimodal-rag-audit.json
```

## 5. Anti-patterns

- Re-ranker trop aggressif (top 1 au lieu de top 3) : detruit la diversite des sources.
- Stocker les images en base64 dans la base vectorielle : coute 33 % de plus qu en binaire S3.
- Re-utiliser un embedder text-only pour des images : similarite sans sens.
