# Workspace Guardrails 101

Ce dossier est le coeur executable du lab. Le flux attendu est :

1. relire la matrice de controle ;
2. traduire ce raisonnement en policy Bedrock ;
3. creer et publier le guardrail avec `boto3` ;
4. rejouer une requete protegee avec `guardrailConfig` ;
5. conserver une preuve d'audit minimale.

## Prerequis
- un compte AWS avec acces a Bedrock dans la bonne region
- des permissions IAM pour creer un guardrail et appeler `bedrock-runtime:Converse`
- Python 3.11+ et `boto3`

## Ordre de travail
1. lire `guardrails/guardrails-control-matrix.json`
2. lire `guardrails/config/guardrail-policy.json`
3. `python guardrails/create_guardrail.py`
4. verifier `guardrails/guardrail-runtime-config.json`
5. `python guardrails/check_request.py`
6. inspecter `guardrails/guardrail-audit-log.json`

## Role des fichiers

### `guardrails-control-matrix.json`
- relie chaque `risk` a un `control`, un `owner` et une `proof`
- alimente `create_guardrail.py`
- choisit le `testPrompt` du scenario runtime retenu

### `config/guardrail-policy.json`
- document de travail qui relie les risques metier aux primitives Bedrock
- mappe `contentFilters`, `deniedTopics`, `sensitiveInformationPolicy` et `wordPolicy`

### `create_guardrail.py`
- charge la policy et la matrice de controle
- appelle `boto3.client("bedrock").create_guardrail(...)`
- publie une version avec `create_guardrail_version(...)`
- ecrit `guardrail-runtime-config.json` avec `activeControl`

### `check_request.py`
- charge `guardrail-runtime-config.json`
- appelle `boto3.client("bedrock-runtime").converse(...)`
- applique `guardrailConfig`
- ecrit `guardrail-audit-log.json` avec le risque, le proprietaire et la preuve attendue

## Ce que vous devez verifier
- le `guardrailIdentifier` vient bien de la creation Bedrock, pas d'une valeur inventee
- la `guardrailVersion` est une version publiee et non un placeholder permanent
- `activeControl` apparait dans `guardrail-runtime-config.json`
- `stopReason` et `requestId` sont presents dans la preuve d'audit
- vous pouvez expliquer ce que Guardrails couvre et ce qui doit rester pris en charge par IAM ou l'application

## Si la creation ou l'execution echoue
- verifiez la region AWS utilisee dans `create_guardrail.py`
- verifiez vos permissions IAM Bedrock
- confirmez que le guardrail a bien ete publie avant d'executer `check_request.py`
- verifiez que `guardrails-control-matrix.json` contient bien un controle `bedrock_guardrail_runtime`
- gardez `DRAFT` pour l'iteration rapide seulement, pas comme sortie finale du hands-on
