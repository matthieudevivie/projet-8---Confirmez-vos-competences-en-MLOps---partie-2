# Scoring crédit — mise en production et monitoring

Projet 8 du parcours OpenClassrooms : *Confirmez vos compétences en MLOps (partie 2/2)*.

« Prêt à Dépenser » dispose d'un modèle de scoring crédit, entraîné et versionné dans le
[projet 6](https://github.com/matthieudevivie/projet-6---Initiez-vous-au-ML-Ops---partie-1).
Ce dépôt le met en production : API, conteneur Docker, pipeline CI/CD, puis stockage des
données de production et suivi de la dérive des données.

**En production** : https://scoring-credit.onrender.com (documentation interactive sur `/docs`).
L'offre gratuite de Render met le service en veille : le premier appel peut prendre une minute.

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

## Lancer l'API

**En local**, avec la journalisation des appels dans une base SQLite :

```bash
mkdir -p logs
DATABASE_URL=sqlite:///logs/appels.db uv run uvicorn api.main:app --port 8000
```

Sans `DATABASE_URL`, l'API fonctionne aussi, mais n'écrit les appels que dans ses logs (au format JSON).

**Avec Docker**, comme en production :

```bash
docker build -t scoring-credit .
docker run --rm -p 7860:7860 scoring-credit
```

| Route | Rôle |
|---|---|
| `GET /docs` | Documentation interactive (la racine `/` y renvoie) |
| `GET /health` | État du service, modèle chargé et commit déployé |
| `POST /predict` | Probabilité de défaut et décision pour un client (paramètre `id_client` facultatif, pour le suivi) |
| `GET /drift` | Dernier rapport de data drift (Evidently) |

Exemple d'appel, avec le premier client d'exemple enregistré par MLflow :

```bash
uv run python -c "
import json, math
e = json.load(open('model/input_example.json'))
client = dict(zip(e['columns'], e['data'][0]))
print(json.dumps({k: None if isinstance(v, float) and math.isnan(v) else v for k, v in client.items()}))
" > client.json
curl -X POST "localhost:8000/predict?id_client=100001" -H "Content-Type: application/json" -d @client.json
# {"probabilite_defaut":0.0487,"decision":"accorde","seuil":0.09,"version_modele":10}
```

## Tests et CI/CD

```bash
uv run pytest
```

Le pipeline [`.github/workflows/ci.yml`](.github/workflows/ci.yml) tourne sur chaque pull request
et sur chaque push dans `main` :

1. **tests** : la suite pytest ;
2. **image** : construction de l'image Docker, démarrage du conteneur, vérification de `/health` et de `/drift` ;
3. sur `main` seulement : publication de l'image sur `ghcr.io`, puis **deploy** sur Render, et
   attente que `/health` renvoie le commit qui vient d'être déployé.

## Stockage des données de production

Chaque appel à `/predict`, réussi ou refusé, est enregistré par un middleware de l'API
([`api/journal.py`](api/journal.py)) : une ligne JSON dans les logs, et une ligne dans la table
`appels` si une base est configurée.

| Colonne | Contenu |
|---|---|
| `date_appel`, `id_client` | Quand, et pour quel client |
| `statut_http` | 200 réussi, 422 refusé par la validation, 500 panne |
| `duree_ms`, `duree_inference_ms` | Latence totale de l'API, et temps du seul modèle |
| `probabilite_defaut`, `decision`, `seuil`, `version_modele` | La réponse renvoyée |
| `entrees` | Les 12 variables surveillées (JSON) : les 10 plus importantes du modèle, plus 2 variables 0/1 |
| `erreur` | La raison d'un refus ou d'une panne (JSON) |

Captures de la base : [structure](captures/stockage_1_structure.png),
[données](captures/stockage_2_donnees.png), [requête de suivi](captures/stockage_3_requete.png).

**Choix et limites.** La base est un fichier SQLite local, alimenté par une simulation de
production : 5 000 clients de `application_test` envoyés à l'API locale
([`scripts/simuler_production.py`](scripts/simuler_production.py)). Sur Render, le disque est
effacé à chaque redémarrage : l'API n'y écrit que ses logs JSON. En production réelle, la même
table irait dans une base hébergée (PostgreSQL managé, par exemple) : il suffit de changer
`DATABASE_URL`, le code ne change pas.

**Volume et coût.** Environ 500 octets par appel (2,5 Mo pour 5 000 appels), dont les trois quarts
pour `entrees`. Un million d'appels représente donc environ 500 Mo. C'est pour cela qu'on ne
journalise que les 12 variables surveillées, et pas les 800 : le volume serait environ 60 fois plus
grand. Au-delà de la durée de conservation, les lignes seraient supprimées ou agrégées
(statistiques par jour).

**RGPD.** Ces données concernent des personnes : `id_client` est un identifiant pseudonyme, qui
reste une donnée personnelle, et `entrees` contient l'âge (`DAYS_BIRTH`) et le sexe
(`CODE_GENDER_F`). Les principes appliqués :
- **minimisation** : 12 variables sur 800, et jamais la valeur refusée d'une requête invalide ;
- **finalité** : uniquement la surveillance du modèle (dérive, erreurs, latence) ;
- **conservation limitée** : durée à fixer avec le DPO. L'analyse n'a besoin que d'une fenêtre
  récente et d'une référence, et les statistiques agrégées peuvent être gardées sans les lignes ;
- **accès restreint** à l'équipe data. Les fichiers `logs/` et `data/` ne sont pas versionnés.

## Monitoring : lire l'analyse du drift

L'analyse se trouve dans [`notebooks/analyse_drift.ipynb`](notebooks/analyse_drift.ipynb). Elle
compare les appels journalisés aux données d'entraînement (la référence) sur les 12 variables
surveillées : tests statistiques (KS pour les variables numériques, χ² pour les variables 0/1),
PSI, visualisations, sortie du modèle, erreurs et latence, puis suivi par fenêtres de 500 appels.
Elle produit le rapport Evidently [`rapports/rapport_drift.html`](rapports/rapport_drift.html),
servi par l'API sur [`/drift`](https://scoring-credit.onrender.com/drift).

Pour la relancer (les données brutes Kaggle doivent être dans `../projet6/data`) :

```bash
uv run python scripts/preparer_donnees.py                    # data/reference.parquet et data/production.parquet
# l'API locale lancée avec DATABASE_URL (voir plus haut), puis :
uv run python scripts/simuler_production.py --clients 5000   # remplit logs/appels.db
```

puis exécuter le notebook.

**Lire le rapport `/drift`.** Le haut de page donne le nombre de variables en dérive. Evidently ne
conclut à une dérive globale que si la moitié des variables dérivent. Ce seuil est trop tolérant
ici : les 12 variables ont été choisies pour leur importance, et une seule suffit à justifier une
vérification. Le tableau détaille ensuite chaque variable : test utilisé, score, et distribution
de la référence comparée à celle de la production.

**Règle d'alerte retenue** (calculée sur une fenêtre des 500 derniers appels) :

| Niveau | Déclencheur |
|---|---|
| **Alerte** | une variable avec PSI > 0,25 · un écart de valeurs manquantes > 10 points · PSI des probabilités prédites > 0,1 ou taux de refus qui varie de plus de 5 points |
| **À surveiller** | PSI entre 0,1 et 0,25 · distance de Wasserstein normalisée > 0,1 |

**Résultat sur la simulation.** Deux variables sur 12 ont dérivé : `PAYMENT_RATE` (PSI 1,47, des
crédits plus courts) et `EXT_SOURCE_1` (13,6 points de valeurs manquantes en moins). Les décisions
n'ont pas changé : le PSI des scores est de 0,007, et le taux de refus passe de 27,2 % à 25,9 %.
L'API fonctionne normalement : 1,8 % d'erreurs, toutes des 422 attendues, et une latence p99 de
6,9 ms. Le modèle peut donc rester en production, sous surveillance. Le plan d'action est détaillé
à la fin du notebook.

## Avancement

- [x] Étape 1 — Dépôt Git et structure du projet
- [x] Étape 2 — API, tests, Docker et pipeline CI/CD
- [x] Étape 3 — Stockage des données de production et analyse du drift
- [ ] Étape 4 — Analyse et optimisation des performances
