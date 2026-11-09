# Architecture, répartition et contrat API ↔ monitoring

## Vue d'ensemble

```
 ESPN / Open-Meteo / OSM
          │  dvc repro (calendar → leaderboards → rounds → courses → weather)
          ▼
 data/processed/features.parquet ──────────────┐   (DVC, Release data-v1 du dépôt)
          │  pga.train                          │
          ▼                                     │ référence 2018-2022
 MLflow (runs + Model Registry)                 │
   pga-score-model @champion / @previous        │
          │  mlflow.pyfunc.load_model           │
          ▼                                     ▼
 API FastAPI (pga.api) ──► data/monitoring/predictions.parquet ──► pga.monitor
   /health /predict /reload                                        (PSI + MAE/biais)
          ▲                                                              │ alerte
          │ /reload                                                      ▼
          └──────────── pga.rollback (champion ← previous)  ou  réentraînement (pga.train)
```

## Répartition

| Bloc | Responsable | Fichiers |
|---|---|---|
| API FastAPI, tests d'intégration, journal des prédictions | **Nicolas** | `src/pga/api.py`, `tests/test_api.py` |
| Monitoring du drift | **Léo** | `src/pga/monitor.py`, `tests/test_monitor.py` |
| Rollback et runbook | **Léo** | `src/pga/rollback.py`, `docs/runbook.md` |
| Déclencheur de réentraînement, incident simulé, test « clone neuf », slides | **ensemble** | `src/pga/trigger.py`, `docs/incident.md` |

## Contrat API

**Modèle servi** : `mlflow.pyfunc.load_model("models:/pga-score-model@champion")`.
Tracking URI : variable `MLFLOW_TRACKING_URI` (`.env`), sinon `pga.config.MLFLOW_URI`.
Les colonnes d'entrée sont exactement `pga.features.FEATURES` (schéma unique, ne pas
recopier la liste).

### `GET /health`

```json
{"status": "ok", "model_name": "pga-score-model", "model_alias": "champion", "model_version": "2"}
```

### `POST /predict`

Entrée : une liste de joueur-tours. `event_id`, `player_id`, `round` sont obligatoires ;
chaque feature de `FEATURES` est un nombre ou `null` (→ NaN, géré par le modèle). Colonne
inconnue ou type invalide → **422**.

```json
{"rows": [{"event_id": "401580344", "player_id": "9478", "round": 1,
           "form_ewm": -0.8, "course_diff_mean": 0.4, "fc_wind_mean": 14.2, "...": null}]}
```

Sortie :

```json
{"model_version": "2",
 "predictions": [{"event_id": "401580344", "player_id": "9478", "round": 1, "prediction": -0.63}]}
```

### `POST /reload`

Relit l'alias `champion` dans le Registry et renvoie la même réponse que `/health`.
Appelé après un rollback ou une promotion, sans redémarrer le service.
Si `PGA_API_KEY` est défini, l'en-tête `X-API-Key` est obligatoire (401 sinon).

### `GET /metrics`

Format Prometheus : `pga_http_requests_total{path,status}`, `pga_predict_latency_seconds`,
`pga_predicted_rows_total{model_version}`, `pga_reloads_total{result}`, `pga_served_model_version`.

## Journal des prédictions (entrée du monitoring)

Chaque appel à `/predict` ajoute ses lignes à `data/monitoring/predictions.parquet`
(chemin `pga.config.PREDICTIONS_LOG`, ignoré par git) :

| Colonne | Type |
|---|---|
| `ts` | horodatage UTC de la prédiction |
| `event_id`, `player_id`, `round` | clés joueur-tour |
| `model_version` | version du Registry qui a prédit |
| `prediction` | `to_par` prédit |
| colonnes de `FEATURES` | valeurs reçues |

`pga.monitor` relit ce fichier, le joint aux scores réels une fois les tours joués
(`rounds.parquet`) et calcule le drift des données (PSI par feature) et de performance
(MAE, biais, skill).

## Règles de travail

- Une branche et une PR par bloc, relue et approuvée par l'autre avant le merge.
- `uv run pytest` vert avant toute PR.
- Secrets uniquement dans `.env` (jamais dans git) ; données uniquement sous DVC.
- Chaque nouvelle commande est documentée dans le README.
