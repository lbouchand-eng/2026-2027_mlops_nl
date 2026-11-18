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
| CUSUM du biais | quand les scores sont connus | `S± = max(0, S± ± z_t − k)`, `z_t = biais du tournoi / σ_tournoi`, cumulé sans fenêtre | S > 3 | S > 4 |

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

## Pourquoi un CUSUM en plus du z-score

Le z-score est une moyenne sur 8 tournois : un biais qui apparaît est dilué par les tournois
sains de la fenêtre, d'où 4 tournois pour détecter l'incident ci-dessous. Le CUSUM (Page,
1954) ne moyenne pas, il **accumule** : chaque tournoi ajoute `z_t − k` à la somme, qui ne
descend jamais sous 0. Le bruit centré est absorbé par `k`, un petit biais qui se répète
s'additionne. Deux sommes, une par sens (prédictions trop hautes / trop basses).

- `k = 0,5` : réglage standard pour détecter un décalage de 1 σ_tournoi (≈ 0,85 coup).
- Seuils vérifiés **hors échantillon** : modèle entraîné jusqu'à l'année N−1, évalué sur N,
  pour N = 2019 à 2022 (167 tournois). Toutes les alertes (S > 4) tombent en **2020**
  (saison Covid, biais moyen −0,44 coup : une vraie rupture) ; aucune en 2019, 2021, 2022.
  Avec S > 5, une seule alerte, aussi en 2020 : 4 est le compromis retenu.
- Le CUSUM est lu dans le même tag `valid_event_bias_sd` du champion que le z-score.

Limite : le déclencheur (`trigger.py`) décide toujours sur le z-score et la MAE ; il ne voit
que les derniers tournois (« as-of »), ce qui tronquerait le cumul. Le CUSUM accélère l'alerte
et le rollback manuel du runbook, pas encore la décision automatique.

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
- **CUSUM** : aucun statut modifié. Il dépasse le seuil de warning une seule fois (3,3 le
  18 janvier 2024, contrôle déjà en warning par le z-score) et n'atteint jamais l'alerte.

**Lecture** : dérive lente, détectée au niveau warning ; elle justifie un réentraînement
planifié, pas une intervention d'urgence.

### Incident simulé : vent prévu +15 km/h à partir du 1er mars 2024 (`timeline_incident.png`)

```bash
uv run python -m pga.monitor replay --years 2023 2024 --inject fc_wind_mean+15 --since 2024-03-01
```

| Tournoi | Statut sans CUSUM | Statut avec CUSUM | PSI vent | Biais | z | CUSUM |
|---|---|---|---|---|---|---|
| 29 fév., Cognizant Classic (avant l'injection) | ok | ok | 0,35 | +0,31 | 1,0 | 1,9 |
| 7 mars, Puerto Rico Open (1re semaine touchée) | ok | **warning** | 0,42 | +0,51 | 1,7 | 3,6 |
| 7 mars, Arnold Palmer Invitational | ok | **alert** | 0,42 | +0,60 | 2,0 | 4,9 |
| 14 mars, THE PLAYERS | warning | alert | 0,74 | +0,77 | 2,6 | 5,3 |
| 28 mars, Houston Open | alert | alert | 1,62 | +1,03 | 3,4 | 7,7 |
| 11 avril, Masters | alert | alert | 3,47 | +1,21 | 4,0 | 8,6 |

Sans CUSUM, détection en **4 tournois** : la fenêtre de 8 tournois dilue les premiers
tournois touchés. Une fenêtre plus courte détecterait plus vite mais lèverait plus de
fausses alertes. Avec le CUSUM, **alerte dès la 1re semaine touchée** (3 semaines plus tôt),
sans alerte supplémentaire sur le drift naturel. À noter : le CUSUM entre en mars à 1,9,
déjà chargé par la dérive naturelle de janvier ; c'est voulu (un biais qui revient dans le
même sens est plus suspect), mais il faut le dire en lisant ce gain.

## Sorties

| Fichier | Contenu |
|---|---|
| `reports/monitoring/latest.json` | dernier contrôle : statut, raisons, signaux, modèle (lu par le déclencheur) |
| `reports/monitoring/summary_<scénario>.json` | même contenu, par scénario de replay |
| `reports/monitoring/timeline_<scénario>.csv` / `.png` | un contrôle par tournoi |
| `reports/monitoring/calibration.json` | distribution du PSI naturel et seuils par feature |
| MLflow, expérience `pga-monitoring` | un run par commande, métriques par tournoi (`step`) |
| Pushgateway (`check`, si `PUSHGATEWAY_URL`) | dernier contrôle, job `pga_monitor`, affiché dans Grafana (ci-dessous) |

## Tableau de bord (Prometheus + Grafana)

`docker compose --profile observability up -d --build` ajoute Prometheus (:9090), une
Pushgateway (:9091) et Grafana (:3000, accès anonyme, ports limités à la machine locale).
Configuration versionnée dans `observability/` : `prometheus.yml`, source de données et
dashboard « PGA MLOps » provisionnés en fichiers (aucun réglage à la main).

- **API** : Prometheus scrape `api:8000/metrics` toutes les 5 s (requêtes par route et
  statut, latence de l'inférence, lignes prédites par version, rechargements, version servie).
- **Monitoring** : `monitor check` est un processus court, que Prometheus ne peut pas scraper.
  Si `PUSHGATEWAY_URL` est défini (`http://pushgateway:9091` dans la stack), il pousse son
  dernier contrôle :

| Métrique | Contenu |
|---|---|
| `pga_drift_status` | 0 ok, 1 warning, 2 alert |
| `pga_monitor_model_version` | version du champion contrôlé |
| `pga_bias`, `pga_bias_z` | biais de la fenêtre (coups) et son z-score |
| `pga_mae_ratio` | MAE de la fenêtre / MAE de validation |
| `pga_psi_max{feature}` | PSI de la feature la plus proche de son seuil d'alerte |
| `pga_cusum{direction="up"\|"down"}` | CUSUM du biais par tournoi |

Chaque push remplace tout le groupe `job=pga_monitor` : un signal non évalué (journal trop
court, scores pas encore connus) disparaît au lieu de garder une valeur périmée. Une
Pushgateway injoignable est signalée sur stderr sans interrompre le contrôle ; `latest.json`
reste la source lue par le déclencheur. Sans `PUSHGATEWAY_URL`, rien ne change.

Le dashboard reprend les seuils de ce document (z = 2 / 3, ratio de MAE 1,10, CUSUM 3 / 4).
`scripts/demo_live.sh` joue l'incident sur cette stack : alerte, rollback, retour à ok
(README, « Démo sur la stack visible »).

## Commandes

```bash
uv run python -m pga.monitor calibrate        # recalcule les seuils PSI (après changement de référence)
uv run python -m pga.monitor replay --years 2023 2024
uv run python -m pga.monitor replay --years 2023 2024 --inject fc_wind_mean*3.6 --since 2024-03-01 --name incident
uv run python -m pga.monitor check            # production : journal de l'API + scores réels
PUSHGATEWAY_URL=http://127.0.0.1:9091 uv run python -m pga.monitor check   # + push vers la Pushgateway de la stack
uv run python -m pga.monitor backfill-tags    # tag valid_event_bias_sd sur les versions anciennes
```
