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
uv sync                       # Python 3.12, dépendances + dvc[gdrive]
uv run dvc pull               # récupère les données versionnées (voir « Remote DVC »)
uv run pytest                 # tests des parseurs
```

Pour (re)construire les données depuis les API (≈ 30-40 min, surtout du throttle) :

```bash
uv run dvc repro              # calendar → leaderboards → rounds → courses → weather
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

## Remote DVC (Google Drive)

Une fois pour le projet (Léo) :
1. Créer un dossier partagé sur Google Drive et le partager avec l'équipe ; son
   identifiant est la fin de l'URL `drive.google.com/drive/folders/<FOLDER_ID>`.
2. `uv run dvc remote add -d gdrive gdrive://<FOLDER_ID>` puis committer `.dvc/config`.

Pour chaque membre : `uv run dvc pull` ouvre une authentification Google dans le
navigateur. Si Google bloque le client OAuth par défaut de DVC, créer un client OAuth
« Desktop » dans Google Cloud Console et le déclarer **en local uniquement** :

```bash
uv run dvc remote modify --local gdrive gdrive_client_id '<id>'
uv run dvc remote modify --local gdrive gdrive_client_secret '<secret>'
```

`.dvc/config.local` et `.env` sont ignorés par git : aucun secret dans le dépôt.
