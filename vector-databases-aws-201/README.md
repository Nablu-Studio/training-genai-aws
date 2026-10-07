# Vector Databases AWS 201

## Objectif

Choisir entre les quatre bases vectorielles natives AWS, creer un index, inserer des vecteurs, et executer une requete k-NN avec metadata filter. Produire un audit log defensable.

## Prerequis

- Compte AWS avec acces Bedrock + OpenSearch Service
- `TRAINING_OPENSEARCH_INDEX` (defaut : `nablu-training-chunks`)
- Python 3.11+ avec `boto3`

## Ordre d execution

1. Creer l environnement : `python3 -m venv .venv && source .venv/bin/activate && pip install boto3`.
2. Verifier la matrice : `cat vector/decisions/decision-matrix.json` et `cat vector/decisions/shortlist.json`.
3. Lancer le script : `python3 vector/scripts/build_and_query.py`. Verifier `vector/audit/index-query-audit.json`.
4. Comparer la latence p99 d insertion et la latence de la requete k-NN.

## Ce que vous devez obtenir

| Fichier | Contenu attendu |
|---|---|
| `vector/decisions/decision-matrix.json` | 4 options, 5 criteres, scores par critere |
| `vector/decisions/shortlist.json` | 1-2 options retenues avec justification explicite |
| `vector/audit/index-query-audit.json` | `indexName`, `vectorsInserted`, `query.latencyMs`, `query.matchedIds` |

## Troubleshooting

| Symptome | Cause probable | Action |
|---|---|---|
| `ResourceNotFoundException` | Index non cree ou nom invalide | Verifier la region et le nom d index |
| Latence k-NN > 500 ms | Index sous-dimensionne ou filtre metadata non indexe | Revoir la topologie, pre-filter sur metadata |
| `AccessDenied` sur `opensearch:ESHttpPOST` | IAM insuffisant | Ajouter `es:ESHttpPost` et `es:ESHttpPut` sur l index |
| Pas de hits | Le filtre metadata exclut tous les documents | Relacher le filtre, inspecter les metadonnees |
