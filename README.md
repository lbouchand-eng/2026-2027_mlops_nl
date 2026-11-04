# PGA Tour — score par tour (projet MLOps)

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

Prérequis : [uv](https://docs.astral.sh/uv/) et git. Aucune clé d'API, aucun secret.

```bash
git clone https://github.com/lbouchand-eng/2026-2027_mlops_nl.git && cd 2026-2027_mlops_nl
uv sync                                  # Python 3.12 et dépendances (verrouillées par uv.lock)
uv run pytest                            # 60+ tests, aucune donnée requise
uv run dvc repro                         # données depuis les API publiques + features + entraînement (≈ 40 min)
bash scripts/demo_setup.sh               # 2e version du modèle : champion + previous (pour le rollback)
uv run uvicorn pga.api:app               # API sur http://127.0.0.1:8000/docs
```

**Plus rapide**, si on vous a transmis le dossier `pga-dvcstore` (≈ 130 Mo) : placez-le à
côté du dépôt et remplacez `dvc repro` par `uv run dvc pull && uv run python -m pga.train`
(après un `pull`, DVC considère l'étape `train` à jour : il faut lancer l'entraînement
directement pour créer le modèle dans le MLflow local, qui n'est pas versionné).

Chaque étape du pipeline est idempotente : les réponses brutes des API sont gardées sous
`data/raw/` et réutilisées, donc un `dvc repro` interrompu se relance sans perte.
`uv run python scripts/registry_status.py` affiche les versions du modèle et leurs alias.

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
  z-score du biais par tournoi et ratio de MAE (seuils lus dans les tags du champion).
  Statut `ok` / `warning` / `alert`, avec le code de sortie 0 / 1 / 2.
- Drift naturel 2023-2024 : 79 ok, 9 warning (biais de +0,65 coup début 2024), 1 alerte.
  Incident « vent +15 km/h » : alerte en 4 tournois.
- Méthode, seuils et résultats : [`docs/monitoring.md`](docs/monitoring.md) ; quoi faire en
  cas d'alerte : [`docs/runbook.md`](docs/runbook.md).

## Déclencheur de réentraînement (`trigger.py`)

```bash
uv run python -m pga.trigger --dry-run                 # rien (0) / réentraînement (10) / incident de données (20) / rollback (30)
uv run python -m pga.trigger --as-of 2024-02-02        # simulation : réentraînement après la dérive de janvier 2024
```

Champion et challenger sont comparés sur les 8 mêmes tournois récents. Une dégradation
peu après une mise en service déclenche un rollback, une alerte PSI ne déclenche jamais
de réentraînement. Détails et résultats : [`docs/runbook.md`](docs/runbook.md).

## Remote DVC (stockage local)

Le remote par défaut `localstore` est un dossier **à côté du dépôt** :
`../pga-dvcstore` (chemin relatif, identique pour chaque membre).
Google Drive a été écarté : l'application OAuth de DVC est bloquée par Google et les
comptes étudiants ne peuvent pas créer leur propre client OAuth.

Deux façons d'obtenir les données sur une nouvelle machine :

1. **Reconstruire depuis les API** (aucune clé, ≈ 30-40 min) : `uv run dvc repro`.
   Les sources sont gratuites et les étapes idempotentes. Si une API a modifié ses
   données entre-temps, `uv run dvc status` le signale (hash différent du `dvc.lock`).
2. **Copier le stockage** : récupérer le dossier `pga-dvcstore` (≈ 130 Mo, zip partagé
   entre membres), le placer à côté du dépôt, puis `uv run dvc pull`.

Pour utiliser un autre emplacement sans toucher au dépôt :
`uv run dvc remote modify --local localstore url /chemin/vers/pga-dvcstore`.

`.dvc/config.local` et `.env` sont ignorés par git : aucun secret dans le dépôt.
