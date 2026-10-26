# Scoring crédit — mise en production et monitoring

Projet 8 du parcours OpenClassrooms : *Confirmez vos compétences en MLOps (partie 2/2)*.

« Prêt à Dépenser » dispose d'un modèle de scoring crédit, entraîné et versionné dans le
[projet 6](https://github.com/matthieudevivie/projet-6---Initiez-vous-au-ML-Ops---partie-1).
Ce dépôt le met en production : API, conteneur Docker, pipeline CI/CD, puis stockage des
données de production et suivi de la dérive des données.

## Le modèle

| | |
|---|---|
| Modèle | `scoring_credit_lgbm`, version 10 du registre MLflow (alias `champion`) |
| Algorithme | LightGBM (`LGBMClassifier`) |
| Entrées | 800 variables issues des tables Home Credit (184 obligatoires, les autres peuvent être manquantes) |
| Sortie | Probabilité de défaut de paiement |
| Décision | Crédit refusé si la probabilité est ≥ **0,09** (seuil minimisant le coût métier) |

Les fichiers du modèle sont dans [`model/`](model/), avec leurs métadonnées dans
[`model/metadata.json`](model/metadata.json). Ils sont copiés à l'identique depuis le registre
MLflow du projet 6 : l'empreinte SHA-256 de `model.pkl` y est consignée.

## Environnement

Le modèle a été sérialisé sous Python 3.13 avec des versions précises de scikit-learn, LightGBM
et pandas. Le projet reprend **exactement ces versions** (voir `pyproject.toml`) : charger un
modèle avec d'autres versions peut produire des erreurs ou des prédictions différentes.

```bash
uv sync
```

## Avancement

- [x] Étape 1 — Dépôt Git et structure du projet
- [ ] Étape 2 — API, tests, Docker et pipeline CI/CD
- [ ] Étape 3 — Stockage des données de production et analyse du drift
- [ ] Étape 4 — Analyse et optimisation des performances
