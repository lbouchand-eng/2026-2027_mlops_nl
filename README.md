# PGA Tour — score par tour (projet MLOps)

Prévoir le score **vs par** de chaque joueur du PGA Tour à chaque tour, à partir de :
la météo **prévue** sur sa fenêtre de départ, sa forme récente relative au field, son
historique sur le parcours et la difficulté du parcours (éditions passées uniquement).

| Période    | Rôle                                         |
|------------|----------------------------------------------|
| 2016-2020  | historique pour construire les variables     |
| 2021-2022  | entraînement / référence                     |
| 2023-2024  | période de drift                             |

## Métrique

**MAE sur `to_par` (en coups)**, comparée à une baseline naïve (moyenne historique du
parcours) via un score de skill `1 − MAE/MAE_baseline`. Justification, métriques
secondaires et protocole : [`docs/metric.md`](docs/metric.md).

## Démarrage

```bash
uv sync                       # Python 3.12, dépendances + dvc
uv run dvc pull               # données versionnées depuis ../pga-dvcstore (voir « Remote DVC »)
uv run pytest                 # tests des parseurs
```

Pour (re)construire les données depuis les API (≈ 30-40 min, surtout du throttle) :

```bash
uv run dvc repro              # sans stockage : reconstruit tout ; calendar → leaderboards → rounds → courses → weather → features
uv run dvc push
```

Chaque étape est idempotente : les réponses brutes sont stockées sous `data/raw/` et
réutilisées si elles existent déjà. Un script interrompu se relance sans perte.

## Pipeline de données (`dvc.yaml`)

| Étape          | Script                    | Sortie                                         |
|----------------|---------------------------|------------------------------------------------|
| `calendar`     | `src/pga/calendar.py`     | `data/interim/events.parquet` (1 appel/saison) |
| `leaderboards` | `src/pga/leaderboards.py` | `data/raw/espn/leaderboard/{event_id}.json`    |
| `rounds`       | `src/pga/rounds.py`       | `data/processed/rounds.parquet`                |
| `courses`      | `src/pga/geocode.py`      | `data/interim/courses.parquet`                 |
| `weather`      | `src/pga/weather.py`      | `data/processed/weather_rounds.parquet`        |
| `features`     | `src/pga/features.py`     | `data/processed/features.parquet`              |

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
| Forme relative au field | `form_rel_short`, `form_rel_long`, `form_ewm`, `n_rounds_hist`, `days_since_last`, `is_new_player` |
| Historique / difficulté parcours | `course_rel_mean`, `course_n_rounds`, `course_diff_mean`, `course_diff_std`, `is_new_course` |
| Météo **prévue** | `fc_temp_mean`, `fc_humidity_mean`, `fc_precip_sum`, `fc_wind_mean`, `fc_gust_max` |
| Contexte | `round`, `major`, `multi_course`, `yards`, `host_par` |

Anti-fuite : `rel = to_par − moyenne du field (tournoi, tour)` ; la forme n'utilise que les
tours antérieurs du joueur, l'historique et la difficulté du parcours que les éditions
précédentes ; la météo observée (`obs_*`, `diff_*`) est réservée au monitoring de drift.
Vérifié par `tests/test_features.py`.

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
