# Bedrock Agents 201 — Runbook

## 1. Pré-requis

- Compte AWS avec `bedrock-agent:CreateAgent` et IAM pour invoquer l agent.
- `boto3`.

## 2. Installation

```bash
cd training/aws/examples/bedrock-agents-201
python3 -m venv .venv && source .venv/bin/activate
pip install boto3
```

## 3. Etape 1 — Lire la spec

```bash
cat agent/agent-design.json
```

## 4. Etape 2 — Invoquer l agent

```bash
export AWS_REGION=eu-west-1
export TRAINING_AGENT_ID="<votre-agent-id>"
export TRAINING_AGENT_ALIAS_ID="<votre-alias-id>"
python3 agent/scripts/invoke_agent.py
cat agent/audit/invoke-audit.json
```

Le script lance 2 tours consecutifs avec un meme `sessionId` et capture la trace (`enableTrace: true`).

Sans `AGENT_ID` le script ecrit un audit `skipped: true` documentant la configuration requise.

## 5. Anti-patterns

- Oublier `enableTrace: true` : la trace est vide et le debug est impossible.
- Re-creer un `sessionId` a chaque tour detruit la continuite.
- Action group non idempotent : l agent boucle.
