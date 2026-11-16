# PGA Tour — score par tour (projet MLOps)

[![CI](https://github.com/lbouchand-eng/2026-2027_mlops_nl/actions/workflows/ci.yml/badge.svg)](https://github.com/lbouchand-eng/2026-2027_mlops_nl/actions/workflows/ci.yml)

Prévoir le score **vs par** de chaque joueur du PGA Tour à chaque tour, à partir de :
la météo **prévue** sur sa fenêtre de départ, sa forme récente relative au field, son
historique sur le parcours et la difficulté du parcours (éditions passées uniquement).

| Période    | Rôle                                         |
|------------|----------------------------------------------|
| 2016-2017  | historique pour construire les variables     |
| 2018-2022  | entraînement / référence (validation : 2022) |
| 2023-2024  | période de drift                             |

Architecture, répartition du travail et contrat API ↔ monitoring : [`docs/architecture.md`](docs/architecture.md).

## Métrique

**MAE sur `to_par` (en coups)**, comparée à une baseline naïve (moyenne historique du
parcours) via un score de skill `1 − MAE/MAE_baseline`. Justification, métriques
secondaires et protocole : [`docs/metric.md`](docs/metric.md).

## Démarrage rapide (clone neuf)

Prérequis : git, [uv](https://docs.astral.sh/uv/), [gh](https://cli.github.com/)
(authentifié avec un compte ayant accès au dépôt) et Docker. Aucune clé d'API, aucun secret.

```bash
git clone https://github.com/lbouchand-eng/2026-2027_mlops_nl.git && cd 2026-2027_mlops_nl
gh release download data-v1 -p pga-dvcstore.tar.gz -D ..    # données versionnées (store DVC)
tar xzf ../pga-dvcstore.tar.gz -C ..                        # à côté du dépôt : remote DVC par défaut
uv sync && uv run dvc pull                                  # data/ aux hash exacts de dvc.lock
docker compose up --build                                   # registry (champion + previous), API, MLflow
```

Le premier lancement entraîne les deux versions du modèle (≈ 1 min). Ensuite :

| Service | Adresse |
|---|---|
| API (Swagger) | http://127.0.0.1:8000/docs |
| Interface MLflow | http://127.0.0.1:5001 |

```bash
# Une prédiction, puis un rollback sur le service en cours d'exécution
curl -s -X POST localhost:8000/predict -H 'content-type: application/json' \
  -d '{"rows":[{"event_id":"401580344","player_id":"9478","round":1,"form_ewm":-0.8,"fc_wind_mean":14.2}]}'
docker compose exec api python -m pga.rollback --reason "essai" --api-url http://127.0.0.1:8000
curl -s localhost:8000/health

# Démo complète de l'incident : requêtes -> alerte -> rollback -> retour à la normale
docker compose run --rm --build demo                        # pause à chaque étape (Entrée)
docker compose run --rm --build -e DEMO_ARGS=--no-pause demo   # d'une traite

docker compose down -v                                      # arrêt et remise à zéro du registry
```

Sans Docker, après le `dvc pull` : `uv run pytest` (114 tests, aucune donnée requise),
`bash scripts/demo_setup.sh` puis `uv run uvicorn pga.api:app`. Pour tout reconstruire
depuis les API publiques plutôt que depuis la Release : `uv run dvc repro` (≈ 40 min).

Chaque étape du pipeline est idempotente : les réponses brutes des API sont gardées sous
`data/raw/` et réutilisées, donc un `dvc repro` interrompu se relance sans perte.
`uv run python scripts/registry_status.py` affiche les versions du modèle et leurs alias.

CI (`.github/workflows/ci.yml`) : à chaque PR et push sur `main`, `uv sync --locked`,
`ruff check` puis `pytest` (sans données DVC ni MLflow local), et build de l'image Docker
avec vérification que le service démarre.

## Pipeline de données (`dvc.yaml`)

| Étape          | Script                    | Sortie                                         |
|----------------|---------------------------|------------------------------------------------|
| `calendar`     | `src/pga/calendar.py`     | `data/interim/events.parquet` (1 appel/saison) |
| `leaderboards` | `src/pga/leaderboards.py` | `data/raw/espn/leaderboard/{event_id}.json`    |
| `rounds`       | `src/pga/rounds.py`       | `data/processed/rounds.parquet`                |
| `courses`      | `src/pga/geocode.py`      | `data/interim/courses.parquet`                 |
| `weather`      | `src/pga/weather.py`      | `data/processed/weather_rounds.parquet`        |
| `features`     | `src/pga/features.py`     | `data/processed/features.parquet`              |
| `train`        | `src/pga/train.py`        | run MLflow + version du Registry, `reports/metrics.json` |

Sources (gratuites, sans clé) :
- ESPN `site.api.espn.com/.../golf/pga/scoreboard?dates=` (calendrier) et
  `site.web.api.espn.com/.../golf/leaderboard?event=` (scores par tour, teeTime, parcours) ;
- Nominatim / OpenStreetMap (géocodage ville → lat/lon, corrections dans
  `data/courses_override.csv`) ;
- Open-Meteo `historical-forecast-api` (prévision archivée, entrée du modèle) et
  `archive-api` ERA5 (observé, pour mesurer l'écart observé − prévu).

Choix de nettoyage (`rounds.py`) : stroke-play individuel uniquement (match play et
team events exclus), tours de playoff exclus, dernier tour d'un joueur WD/DQ retiré,
`to_par` pris dans `displayValue` d'ESPN (correct même sur les tournois multi-parcours,
signalés par `multi_course`). Paramètres dans `params.yaml`.

## Features (`features.py`)

Une ligne par joueur-tour ; schéma unique `FEATURES` / `TARGET` importable
(`from pga.features import FEATURES, TARGET`) par l'entraînement, l'API et le monitoring.

| Groupe | Features |
|---|---|
| Forme relative au field | `form_rel_short`, `form_rel_long`, `form_rel_xlong` (150 tours, niveau de fond), `form_ewm`, `form_ewm_slow`, `n_rounds_hist`, `days_since_last`, `is_new_player` |
| Historique / difficulté parcours | `course_rel_mean`, `course_n_rounds`, `course_diff_mean`, `course_diff_std`, `is_new_course` |
| Météo **prévue** | `fc_temp_mean`, `fc_humidity_mean`, `fc_precip_sum`, `fc_wind_mean`, `fc_gust_max` |
| Contexte | `round`, `major`, `multi_course`, `yards`, `host_par` |

Anti-fuite : `rel = to_par − moyenne du field (tournoi, tour)` ; la forme n'utilise que les
tours antérieurs du joueur, l'historique et la difficulté du parcours que les éditions
précédentes ; la météo observée (`obs_*`, `diff_*`) est réservée au monitoring de drift.
Vérifié par `tests/test_features.py`.

## Entraînement et MLflow (`train.py`)

```bash
uv run dvc repro train                       # ou : uv run python -m pga.train [--refit-years 2021 2023]
uv run dvc metrics show                      # MAE / skill de validation
uv run mlflow ui --backend-store-uri "sqlite:///mlflow.db"   # http://127.0.0.1:5000
```

- Modèle : `HistGradientBoostingRegressor(loss="absolute_error")`, qui optimise directement la MAE
  et gère les valeurs manquantes. Hyperparamètres dans `params.yaml → train`.
- Protocole : fit 2018-2021, validation 2022 (décision), refit 2018-2022 (modèle enregistré),
  évaluation informative sur 2023 et 2024 (drift).
- Chaque run logue : hyperparamètres et fenêtres d'années (params) ; MAE, RMSE, biais,
  skill, MAE par tour (metrics) ; `git_commit`, `git_dirty`, `data_hash` (md5 de
  `features.parquet`, identique à `dvc.lock`) (tags) ; le modèle avec sa signature.
- Registry : modèle `pga-score-model`. Alias **`champion`** = modèle servi, **`previous`** =
  champion précédent (cible du rollback). Promotion seulement si skill > 0 et MAE de
  validation meilleure que celle du champion (`docs/metric.md`).
- Stockage local : `mlflow.db` (runs + registry) et `mlartifacts/` (modèles), ignorés par git ;
  URI surchargeable via `MLFLOW_TRACKING_URI` dans `.env` (voir `.env.example`).
  Chargement : `mlflow.pyfunc.load_model("models:/pga-score-model@champion")`.

## API (`api.py`)

```bash
uv run uvicorn pga.api:app --reload          # http://127.0.0.1:8000/docs (Swagger)
curl -s localhost:8000/health
curl -s -X POST localhost:8000/predict -H 'content-type: application/json' \
  -d '{"rows":[{"event_id":"401580344","player_id":"9478","round":1,"form_ewm":-0.8,"course_diff_mean":0.4,"fc_wind_mean":14.2}]}'
curl -s -X POST localhost:8000/reload        # après un rollback ou une promotion
```

- Sert `models:/pga-score-model@champion`, chargé **par numéro de version** : la version
  renvoyée par `/health` et `/predict` est celle qui a prédit.
- Entrées validées par Pydantic à partir de `pga.features.FEATURES` : feature absente ou
  `null` → NaN ; colonne inconnue, type invalide, tour hors 1-4 → 422 ; pas de modèle → 503.
- `/reload` relit l'alias sans redémarrer ; en cas d'échec, l'ancien modèle reste servi.
- Chaque prédiction est ajoutée à `data/monitoring/predictions.parquet` (variable
  `PREDICTIONS_LOG`), relu par le monitoring. Contrat complet : `docs/architecture.md`.
- `/reload` exige l'en-tête `X-API-Key` si `PGA_API_KEY` est défini dans `.env` ; le rollback et
  le déclencheur l'envoient. Sans cette variable, pas de contrôle (ports limités à la machine locale).
- `GET /metrics` (format Prometheus) : requêtes par route et statut, latence de l'inférence,
  lignes prédites par version, rechargements, version servie.
- Tests d'intégration (`tests/test_api.py`) : vrai modèle entraîné dans un MLflow
  temporaire, vérification de la version servie, du journal, des 422/503 et du
  rechargement après déplacement de l'alias.

## Monitoring et rollback

```bash
uv run python -m pga.monitor replay --years 2023 2024                 # drift naturel -> reports/monitoring/timeline_replay.png
uv run python -m pga.monitor replay --years 2023 2024 --inject fc_wind_mean+15 --since 2024-03-01   # incident simulé
uv run python -m pga.monitor check                                    # production : journal de l'API
uv run python -m pga.rollback --reason "..." --api-url http://127.0.0.1:8000   # champion <-> previous
```

- Signaux sur les 8 derniers tournois : PSI par feature (seuils calibrés sur 2018-2022),
  z-score du biais par tournoi et ratio de MAE (seuils lus dans les tags du champion) ;
  plus un CUSUM du biais par tournoi, cumulé sans fenêtre, contre les petites dérives persistantes.
  Statut `ok` / `warning` / `alert`, avec le code de sortie 0 / 1 / 2.
- Drift naturel 2023-2024 : 79 ok, 9 warning (biais de +0,65 coup début 2024), 1 alerte.
  Incident « vent +15 km/h » : alerte dès la 1re semaine touchée grâce au CUSUM (4 tournois sans). Le PSI est calculé par rapport aux
  données d'entraînement du modèle servi (`reference_profile.json`).
- Méthode, seuils et résultats : [`docs/monitoring.md`](docs/monitoring.md) ; quoi faire en
  cas d'alerte : [`docs/runbook.md`](docs/runbook.md).

## Déclencheur de réentraînement (`trigger.py`)

```bash
uv run python -m pga.trigger --dry-run                 # rien (0) / réentraînement (10) / incident de données (20) / rollback (30)
uv run python -m pga.trigger --as-of 2024-02-05        # simulation : réentraînement après la dérive de janvier 2024
```

Champion et challenger sont comparés sur les 8 mêmes tournois récents, lus dans le journal de
l'API quand il les couvre (entrées réellement servies, qu'un backfill ne peut pas réécrire). Une dégradation ou
une alerte PSI peu après une mise en service déclenche un rollback, une alerte PSI ne
déclenche jamais de réentraînement. Un challenger dont les données s'écartent de celles du
champion n'est pas promu. Détails et résultats : [`docs/runbook.md`](docs/runbook.md).

Planification :
- `docker compose --profile ops up -d` ajoute le service `scheduler`, qui lance le déclencheur
  chaque semaine et **exécute** ses décisions sur le registry de la stack ;
- le workflow `trigger.yml` tourne chaque lundi à 6 h UTC et **décide sans agir** (registry
  éphémère du runner) : décision dans le résumé du job, `latest.json` en artefact.

Avec `PGA_ALERT_WEBHOOK` (`.env`, ou secret GitHub pour le workflow), les alertes de
`monitor check` et toute action du déclencheur autre que « rien » sont envoyées sur ce webhook.

## Incident simulé et démo

```bash
uv run python scripts/incident.py all       # scénario complet -> reports/incident/ (registry isolé mlflow_incident.db)
uv run python scripts/incident.py all --guards   # même panne après correctifs -> reports/incident_guarded/
bash scripts/demo.sh --prepare               # avant l'oral : champion = version corrompue
bash scripts/demo.sh                         # démo en direct, pause entre chaque étape
```

Scénario : le 1er mars 2024, un réentraînement de routine se fait sur un backfill météo
téléchargé en m/s au lieu de km/h (vent ÷ 3,6). La fenêtre de décision étant corrompue
elle aussi, le challenger gagne la comparaison et devient champion. En production, sur des
données saines, le PSI ne bouge pas, mais le biais monte à +0,88 coup en 4 tournois. Le
déclencheur choisit le **rollback** le 25 mars (dégradation juste après une mise en
service), et le monitoring revient à ok (19 contrôles sur 19).

La démo (`scripts/demo.sh`, ≈ 10 s sans les pauses) enchaîne : l'API sert la version
corrompue, 3 442 prédictions de mars-avril 2024, `monitor check` → alerte (biais +1,79,
z = 5,9), `trigger` → rollback et rechargement de l'API, mêmes requêtes → ok. Rapport :
[`docs/incident.md`](docs/incident.md).

Après correctifs, le même backfill est refusé à la promotion. S'il est forcé, le PSI alerte dès
le premier tournoi et le rollback a lieu le 11 mars au lieu du 25 (19 alertes PSI sur 19
contrôles, contre 0 auparavant). Comparé sur le journal de l'API, le challenger corrompu perd
aussi à lui seul (MAE 2,652 contre 2,268). La dernière étape de la démo montre le refus.

## Rapport du projet

Présentation complète du projet en anglais (problème, système, monitoring, incident, choix et
limites) : [`report/project.qmd`](report/project.qmd), rendu en une page HTML autonome.

```bash
uv pip install quarto-cli                         # Quarto dans .venv (ou Quarto installé à part)
uv run python report/make_figures.py              # figures tirées de reports/ et params.yaml
uv run quarto render report/project.qmd           # -> report/project.html
```

## Remote DVC (stockage local)

Le remote par défaut `localstore` est un dossier **à côté du dépôt** :
`../pga-dvcstore` (chemin relatif, identique pour chaque membre).
Google Drive a été écarté : l'application OAuth de DVC est bloquée par Google et les
comptes étudiants ne peuvent pas créer leur propre client OAuth.

Deux façons d'obtenir les données sur une nouvelle machine :

1. **Reconstruire depuis les API** (aucune clé, ≈ 30-40 min) : `uv run dvc repro`.
   Les sources sont gratuites et les étapes idempotentes. Si une API a modifié ses
   données entre-temps, `uv run dvc status` le signale (hash différent du `dvc.lock`).
2. **Récupérer le stockage** : l'archive `pga-dvcstore.tar.gz` de la Release `data-v1` du
   dépôt (privée comme lui), à extraire à côté du dépôt, puis `uv run dvc pull`.

Pour utiliser un autre emplacement sans toucher au dépôt :
`uv run dvc remote modify --local localstore url /chemin/vers/pga-dvcstore`.

`.dvc/config.local` et `.env` sont ignorés par git : aucun secret dans le dépôt.
