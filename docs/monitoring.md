# Monitoring du drift

Code : `src/pga/monitor.py`. Paramètres : `params.yaml → monitor`. Seuils PSI par feature :
`reports/monitoring/calibration.json`. Procédure en cas d'alerte : [`runbook.md`](runbook.md).

## Principe

Après chaque tournoi, on contrôle une **fenêtre glissante des 8 derniers tournois**
(≈ 2 mois de saison, ≈ 1 000 joueur-tours) avec deux familles de signaux.

| Signal | Disponible | Formule | Warning | Alert |
|---|---|---|---|---|
| Drift des données | dès la prédiction | PSI de chaque feature surveillée vs données d'entraînement du champion | seuil calibré par feature | seuil calibré par feature |
| Biais | quand les scores sont connus | `z = biais moyen par tournoi / (σ_tournoi / √n_tournois)` | \|z\| > 2 | \|z\| > 3 |
| MAE | quand les scores sont connus | MAE de la fenêtre / MAE de validation du champion | | > 1,10 |

Le statut est le pire des signaux. Code de sortie : 0 `ok`, 1 `warning`, 2 `alert`.
En dessous de 500 lignes (journal de l'API encore court, scores pas encore publiés), les
signaux ne sont pas évalués plutôt que de lever de fausses alertes.

La référence du PSI est le **profil enregistré avec le modèle servi** (`reference_profile.json`) :
elle suit le champion après un rollback. Les versions sans profil utilisent `reference_years`.
Pour un modèle standard, les deux sont identiques (2018-2022). Voir `docs/incident.md`, § 9.

## Pourquoi un z-score du biais, et pas un seuil fixe

Tout le field d'un tournoi partage la météo et le réglage du parcours : les erreurs sont
corrélées au sein d'un tournoi. Le biais par tournoi du champion a un écart-type de
**0,85 coup** sur la validation 2022 (`σ_tournoi`, tag `valid_event_bias_sd` de la version
dans le Registry). Un seuil fixe « |biais| > 0,2 » sonnerait sur la moitié des tournois.
Le z-score ramène le biais de la fenêtre à son bruit attendu ; il est lu dans les tags du
**champion** : après un rollback, les seuils suivent automatiquement le modèle servi.

## Pourquoi des seuils PSI calibrés par feature

Sur 8 tournois, la météo prévue dépend de la saison et de la région (Hawaï en janvier,
Écosse en juillet). Avec les seuils usuels 0,1 / 0,25, le replay de 2023-2024 levait des
alertes sur 53 contrôles sur 89. On mesure donc le **PSI naturel** sur 204 fenêtres de la
période de référence 2018-2022 (chaque fenêtre comparée à la référence privée de ses
propres années), puis : warning = quantile 99 %, alert = 1,5 × maximum observé, avec des
planchers à 0,10 / 0,25.

| Feature | warning | alert |
|---|---|---|
| `form_rel_short`, `form_ewm` | 0,10 | 0,25 |
| `form_rel_long` | 0,14 | 0,25 |
| `form_rel_xlong` | 0,21 | 0,42 |
| `form_ewm_slow` | 0,18 | 0,30 |
| `days_since_last` | 1,01 | 1,67 |
| `course_rel_mean` | 0,37 | 0,67 |
| `fc_temp_mean` | 2,51 | 4,02 |
| `fc_humidity_mean` | 1,54 | 2,80 |
| `fc_precip_sum` | 0,72 | 1,10 |
| `fc_wind_mean` | 0,97 | 1,57 |
| `fc_gust_max` | 1,46 | 3,26 |

Features non surveillées par PSI : `n_rounds_hist` et `course_n_rounds` (compteurs qui
croissent mécaniquement avec le temps) et les features constantes par tournoi (parcours,
contexte), dont la distribution sur 8 tournois n'a pas de sens.

## Résultats

### Drift naturel 2023-2024 (`timeline_replay.png`)

89 contrôles : **79 ok, 9 warning, 1 alerte**.
- **Biais** : warnings du 18 janvier au 1er février 2024, biais de +0,61 à +0,65 coup
  (z = 2,0 à 2,2). Le field joue de plus en plus bas, et le modèle, entraîné sur 2018-2022,
  surestime les scores. La MAE reste stable (ratio < 1,05) : sans le suivi du biais, cette
  dérive passerait inaperçue.
- **Seule alerte** : le Q-School de décembre 2023, un field de joueurs sans historique sur
  le PGA Tour, absent de la période de référence. C'est une vraie nouvelle population.
- Le reste : des warnings isolés sur la météo ou la forme, sans effet sur la performance.

**Lecture** : dérive lente, détectée au niveau warning ; elle justifie un réentraînement
planifié, pas une intervention d'urgence.

### Incident simulé : vent prévu +15 km/h à partir du 1er mars 2024 (`timeline_incident.png`)

```bash
uv run python -m pga.monitor replay --years 2023 2024 --inject fc_wind_mean+15 --since 2024-03-01
```

| Tournoi | Statut | PSI vent | Biais | z |
|---|---|---|---|---|
| 7 mars (1er tournoi touché) | ok | 0,42 | +0,51 | 1,7 |
| 14 mars, THE PLAYERS | **warning** | 0,74 | +0,77 | 2,6 |
| 28 mars, Houston Open | **alert** | 1,62 | +1,03 | 3,4 |
| 11 avril, Masters | alert | 3,47 | +1,21 | 4,0 |

Détection en **4 tournois** : la fenêtre de 8 tournois dilue les premiers tournois touchés.
C'est le compromis assumé : une fenêtre plus courte détecterait plus vite mais lèverait
plus de fausses alertes (voir ci-dessus).

## Sorties

| Fichier | Contenu |
|---|---|
| `reports/monitoring/latest.json` | dernier contrôle : statut, raisons, signaux, modèle (lu par le déclencheur) |
| `reports/monitoring/summary_<scénario>.json` | même contenu, par scénario de replay |
| `reports/monitoring/timeline_<scénario>.csv` / `.png` | un contrôle par tournoi |
| `reports/monitoring/calibration.json` | distribution du PSI naturel et seuils par feature |
| MLflow, expérience `pga-monitoring` | un run par commande, métriques par tournoi (`step`) |

## Commandes

```bash
uv run python -m pga.monitor calibrate        # recalcule les seuils PSI (après changement de référence)
uv run python -m pga.monitor replay --years 2023 2024
uv run python -m pga.monitor replay --years 2023 2024 --inject fc_wind_mean*3.6 --since 2024-03-01 --name incident
uv run python -m pga.monitor check            # production : journal de l'API + scores réels
uv run python -m pga.monitor backfill-tags    # tag valid_event_bias_sd sur les versions anciennes
```
