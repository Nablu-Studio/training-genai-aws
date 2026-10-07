# Workspace Guardrails 101

Ce workspace sert de runbook d'execution pour un vrai hands-on Bedrock Guardrails. Vous partez d'une matrice de controle, vous traduisez ce raisonnement en policy Bedrock, vous creez puis publiez le guardrail avec `boto3`, puis vous prouvez l'execution runtime avec un audit exploitable.

## Objectif

Relier explicitement :

- un risque metier ;
- un controle applique ;
- un proprietaire de ce controle ;
- une preuve runtime defendable.

## Prerequis

- acces AWS a Bedrock dans la region cible ;
- permissions IAM pour `bedrock:CreateGuardrail`, `bedrock:CreateGuardrailVersion` et `bedrock-runtime:Converse` ;
- Python 3.11+ et `boto3` installes.

## Ordre d'execution

1. Lire `guardrails/guardrails-control-matrix.json`
2. Lire `guardrails/config/guardrail-policy.json`
3. Executer `python guardrails/create_guardrail.py`
4. Verifier `guardrails/guardrail-runtime-config.json`
5. Executer `python guardrails/check_request.py`
6. Inspecter `guardrails/guardrail-audit-log.json`

## Role des artefacts

### `guardrails/guardrails-control-matrix.json`

- formalise `risk -> control -> owner -> proof` ;
- choisit le controle runtime actif pour le hands-on ;
- fournit le `testPrompt` associe au scenario retenu.

### `guardrails/config/guardrail-policy.json`

- mappe les primitives Bedrock reelles ;
- expose `contentFilters`, `deniedTopics`, `sensitiveInformationPolicy` et `wordPolicy` ;
- sert de source pour `create_guardrail.py`.

### `guardrails/create_guardrail.py`

- charge la policy et la matrice ;
- appelle `boto3.client("bedrock").create_guardrail(...)` ;
- publie une version avec `create_guardrail_version(...)` ;
- ecrit `guardrails/guardrail-runtime-config.json`.

### `guardrails/check_request.py`

- recharge `guardrails/guardrail-runtime-config.json` ;
- applique `guardrailConfig` sur un appel `converse(...)` ;
- ecrit `guardrails/guardrail-audit-log.json` avec `risk`, `controlOwner`, `stopReason` et `requestId`.

## Ce que vous devez verifier

- le `guardrailIdentifier` vient bien de la creation Bedrock ;
- la `guardrailVersion` est une version publiee ;
- `activeControl` apparait dans `guardrails/guardrail-runtime-config.json` ;
- l'audit relie le risque, le controle, le proprietaire et la preuve attendue.

## Troubleshooting

- si la creation echoue, verifier la region Bedrock et les permissions IAM ;
- si l'execution runtime echoue, verifier que la version du guardrail est bien publiee ;
- si l'audit est incomplet, verifier que `activeControl` et `testPrompt` sont bien presents dans la config runtime.

## Questions de certification a garder en tete

- pourquoi Guardrails ne remplace-t-il pas IAM ni la validation applicative ?
- pourquoi faut-il publier une version stable au lieu de rester en `DRAFT` ?
- quelle preuve minimale faut-il conserver pour justifier le comportement du systeme sans recopier des donnees sensibles ?
