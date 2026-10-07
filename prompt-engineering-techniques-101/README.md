# Prompt Engineering Techniques 101 — Runbook

## 1. Pré-requis

- Compte AWS avec accès Bedrock (model `amazon.nova-lite-v1:0` activé).
- `boto3`.

## 2. Installation

```bash
cd training/aws/examples/prompt-engineering-techniques-101
python3 -m venv .venv && source .venv/bin/activate
pip install boto3
```

## 3. Etape 1 — Comparer les techniques

```bash
export AWS_REGION=eu-west-1
python3 prompting/scripts/eval_techniques.py
cat prompting/audit/techniques-audit.json | head -30
```

Le script joue 8 questions sous 3 system prompts (zero-shot, few-shot, CoT) et écrit `meanScore` par technique.

## 4. Etape 2 — Function calling

```bash
python3 prompting/scripts/function_calling.py
cat prompting/audit/function-calling-audit.json
```

Le script declare un outil `lookup_genai_service`, appelle `converse` avec `toolConfig`, et valide la conformite du payload.

## 5. Anti-patterns

- Few-shot avec exemples specifiques au point de mirer le dataset (surapprentissage de format).
- CoT sur des taches deterministes (extraction, classification binaire).
- Schema JSON sans description : le modele ignore l outil.
