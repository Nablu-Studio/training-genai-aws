# Embeddings and Chunking 101

## Objectif

Choisir un modele d embedding adapte, appliquer une strategie de chunking defendable, et mesurer la qualite du resultat (coherence intra-chunk, couverture du corpus).

## Prerequis

- Compte AWS avec acces Bedrock active dans `eu-west-1`
- `aws configure` deja execute
- Python 3.11+ avec `boto3` et `numpy`

## Ordre d execution

1. Activer `amazon.titan-embed-text-v2:0` dans Bedrock (ou utiliser un autre modele d embedding via `BEDROCK_EMBED_MODEL_ID`).
2. Creer l environnement : `python3 -m venv .venv && source .venv/bin/activate && pip install boto3 numpy`.
3. Lancer l embedding de reference : `python3 embeddings/scripts/embed_corpus.py`. Verifier `embeddings/reference-vector.json`.
4. Lancer le chunking : `python3 embeddings/scripts/chunk_corpus.py`. Verifier `embeddings/chunk-audit.json` (summary.chunkCount, summary.meanCoherence, summary.lowCoherenceChunks).

## Ce que vous devez obtenir

| Fichier | Contenu attendu |
|---|---|
| `embeddings/reference-vector.json` | `docId`, `modelId`, `rawDimension`, `l2Norm` proche de 1, `head` non vide |
| `embeddings/chunk-audit.json` | `strategy` (size, overlap), `chunks` (avec `coherence` par chunk), `summary` (`chunkCount`, `meanCoherence`, `lowCoherenceChunks`) |

## Strategies couvertes

- Fixed-size avec overlap (defaut : 200 tokens, overlap 40)
- Semantique (decoupage par paragraphe, voir bloc dedie)
- Hierarchical (decoupage multi-niveau, mention theorique)

## Troubleshooting

| Symptome | Cause probable | Action |
|---|---|---|
| `AccessDeniedException` sur `InvokeModel` | Modele non active ou IAM insuffisant | Activer le modele dans Bedrock, verifier `bedrock:InvokeModel` |
| `ValidationException: dimensions` | Parametre `dimensions` non supporte par le modele | Choisir une dimension supportee (256, 512, 1024) ou omettre |
| `meanCoherence` tres bas (< 0.3) | Corpus heterogene ou phrases trop courtes | Basculer en semantique, regrouper les chunks courts |
| Beaucoup de `lowCoherenceChunks` | Corpus bruite ou chunks mal calibres | Revoir taille de chunk et overlap |
