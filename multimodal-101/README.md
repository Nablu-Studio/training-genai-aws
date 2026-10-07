# Multimodal 101 — Runbook

## 1. Pré-requis

- Compte AWS avec Bedrock et `amazon.nova-lite-v1:0` + `amazon.titan-image-generator-v1` activés.
- `boto3`.

## 2. Installation

```bash
cd training/aws/examples/multimodal-101
python3 -m venv .venv && source .venv/bin/activate
pip install boto3
```

## 3. Etape 1 — Lire la matrice des modeles

```bash
cat multimodal/data/model-matrix.json
aws bedrock list-foundation-models --region eu-west-1 --query "modelSummaries[?contains(modelId, 'image') || contains(modelId, 'nova')].modelId" --output text
```

## 4. Etape 2 — Invoquer vision et image gen

```bash
export AWS_REGION=eu-west-1
export TRAINING_MULTIMODAL_IMAGE_PATH=multimodal/data/sample-architecture.png
python3 multimodal/scripts/describe_image.py
python3 multimodal/scripts/generate_image.py
ls -la multimodal/audit/ multimodal/output/
```

Le script `describe_image.py` envoie l'image à Nova Lite via `converse` et écrit `vision-audit.json`.
Le script `generate_image.py` invoque Titan Image Generator v1 et sauvegarde l'image dans `multimodal/output/`.

## 5. Anti-patterns

- Utiliser Stable Diffusion pour classifier une image (il genere, il ne comprend pas).
- Image gen > 1 MP sans pré-traitement : pré-redimensionner avec Pillow.
- Confondre `image_generation` (sortie image) et `vision` (entree image) dans la matrice.
