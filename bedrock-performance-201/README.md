# Bedrock Performance 201 — Runbook

## 1. Pré-requis

- Compte AWS avec Bedrock et `amazon.nova-lite-v1:0` activé.
- `boto3`.

## 2. Installation

```bash
cd training/aws/examples/bedrock-performance-201
python3 -m venv .venv && source .venv/bin/activate
pip install boto3
```

## 3. Etape 1 — Choisir le mode de throughput

```bash
cat perf/data/throughput-decisions.json
aws bedrock list-provisioned-model-throughput --region eu-west-1
```

## 4. Etape 2 — Bench de latence et caching

```bash
export AWS_REGION=eu-west-1
python3 perf/scripts/benchmark_latency.py
cat perf/audit/latency-audit.json
```

Le script fait 10 itérations sans cache, puis 10 itérations avec cache, et calcule p50/p99/mean + delta.

## 5. Anti-patterns

- Provisioned Throughput sans trafic consomme l engagement : demarrer petit, monitorer.
- Mesurer la latence avec 1 seul appel : la p99 nécessite au moins 50 iterations.
- Activer le cache sur un system prompt < 1024 tokens (pas eligible).
