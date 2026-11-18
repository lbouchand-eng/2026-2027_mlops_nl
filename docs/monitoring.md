# Drift monitoring

Code: `src/pga/monitor.py`. Parameters: `params.yaml → monitor`. PSI thresholds per feature:
`reports/monitoring/calibration.json`. Procedure on an alert: [`runbook.md`](runbook.md).

## Principle

After each tournament, we check a **rolling window of the last 8 tournaments**
(≈ 2 months of season, ≈ 1,000 player-rounds) with two families of signals.

| Signal | Available | Formula | Warning | Alert |
|---|---|---|---|---|
| Data drift | at prediction time | PSI of each monitored feature vs the champion's training data | threshold calibrated per feature | threshold calibrated per feature |
| Bias | once scores are known | `z = mean per-tournament bias / (σ_tournament / √n_tournaments)` | \|z\| > 2 | \|z\| > 3 |
| MAE | once scores are known | window MAE / champion's validation MAE | | > 1.10 |
| Bias CUSUM | once scores are known | `S± = max(0, S± ± z_t − k)`, `z_t = tournament bias / σ_tournament`, accumulated without a window | S > 3 | S > 4 |

The status is the worst of the signals. Exit code: 0 `ok`, 1 `warning`, 2 `alert`.
Below 500 rows (API journal still short, scores not yet published), the
signals are not evaluated rather than raising false alerts.

The PSI reference is the **profile registered with the served model** (`reference_profile.json`):
it follows the champion after a rollback. Versions without a profile use `reference_years`.
For a standard model, both are identical (2018-2022). See `docs/incident.md`, § 9.

## Why a bias z-score, and not a fixed threshold

The whole field of a tournament shares the weather and the course setup: errors are
correlated within a tournament. The champion's per-tournament bias has a standard deviation of
**0.85 strokes** on the 2022 validation (`σ_tournament`, tag `valid_event_bias_sd` of the version
in the Registry). A fixed threshold "|bias| > 0.2" would fire on half of the tournaments.
The z-score scales the window's bias by its expected noise; it is read from the
**champion's** tags: after a rollback, the thresholds automatically follow the served model.

## Why a CUSUM on top of the z-score

The z-score is an average over 8 tournaments: an emerging bias is diluted by the healthy
tournaments in the window, hence 4 tournaments to detect the incident below. The CUSUM (Page,
1954) does not average, it **accumulates**: each tournament adds `z_t − k` to the sum, which never
goes below 0. Centered noise is absorbed by `k`; a small bias that repeats
adds up. Two sums, one per direction (predictions too high / too low).

- `k = 0.5`: standard setting to detect a shift of 1 σ_tournament (≈ 0.85 strokes).
- Thresholds checked **out of sample**: model trained up to year N−1, evaluated on N,
  for N = 2019 to 2022 (167 tournaments). All alerts (S > 4) fall in **2020**
  (Covid season, mean bias −0.44 strokes: a real break); none in 2019, 2021, 2022.
  With S > 5, a single alert, also in 2020: 4 is the chosen compromise.
- The CUSUM reads the same `valid_event_bias_sd` tag of the champion as the z-score.

Limitation: the trigger (`trigger.py`) still decides on the z-score and the MAE; it only sees
the latest tournaments ("as-of"), which would truncate the accumulation. The CUSUM speeds up the alert
and the runbook's manual rollback, not yet the automatic decision.

## Why PSI thresholds calibrated per feature

Over 8 tournaments, the weather forecast depends on the season and the region (Hawaii in January,
Scotland in July). With the usual 0.1 / 0.25 thresholds, the 2023-2024 replay raised
alerts on 53 checks out of 89. We therefore measure the **natural PSI** over 204 windows of the
2018-2022 reference period (each window compared with the reference excluding its
own years), then: warning = 99% quantile, alert = 1.5 × observed maximum, with
floors at 0.10 / 0.25.

| Feature | warning | alert |
|---|---|---|
| `form_rel_short`, `form_ewm` | 0.10 | 0.25 |
| `form_rel_long` | 0.14 | 0.25 |
| `form_rel_xlong` | 0.21 | 0.42 |
| `form_ewm_slow` | 0.18 | 0.30 |
| `days_since_last` | 1.01 | 1.67 |
| `course_rel_mean` | 0.37 | 0.67 |
| `fc_temp_mean` | 2.51 | 4.02 |
| `fc_humidity_mean` | 1.54 | 2.80 |
| `fc_precip_sum` | 0.72 | 1.10 |
| `fc_wind_mean` | 0.97 | 1.57 |
| `fc_gust_max` | 1.46 | 3.26 |

Features not monitored by PSI: `n_rounds_hist` and `course_n_rounds` (counters that
grow mechanically over time) and the features constant per tournament (course,
context), whose distribution over 8 tournaments is meaningless.

## Results

### Natural drift 2023-2024 (`timeline_replay.png`)

89 checks: **79 ok, 9 warning, 1 alert**.
- **Bias**: warnings from January 18 to February 1, 2024, bias of +0.61 to +0.65 strokes
  (z = 2.0 to 2.2). The field scores lower and lower, and the model, trained on 2018-2022,
  overestimates the scores. The MAE stays stable (ratio < 1.05): without bias tracking, this
  drift would go unnoticed.
- **Only alert**: the December 2023 Q-School, a field of players with no PGA Tour
  history, absent from the reference period. It is a genuinely new population.
- The rest: isolated warnings on weather or form, with no effect on performance.
- **CUSUM**: no status changed. It crosses the warning threshold only once (3.3 on
  January 18, 2024, a check already in warning from the z-score) and never reaches the alert.

**Reading**: slow drift, detected at warning level; it justifies a scheduled
retraining, not an emergency intervention.

### Simulated incident: forecast wind +15 km/h from March 1, 2024 (`timeline_incident.png`)

```bash
uv run python -m pga.monitor replay --years 2023 2024 --inject fc_wind_mean+15 --since 2024-03-01
```

| Tournament | Status without CUSUM | Status with CUSUM | Wind PSI | Bias | z | CUSUM |
|---|---|---|---|---|---|---|
| Feb. 29, Cognizant Classic (before injection) | ok | ok | 0.35 | +0.31 | 1.0 | 1.9 |
| March 7, Puerto Rico Open (1st affected week) | ok | **warning** | 0.42 | +0.51 | 1.7 | 3.6 |
| March 7, Arnold Palmer Invitational | ok | **alert** | 0.42 | +0.60 | 2.0 | 4.9 |
| March 14, THE PLAYERS | warning | alert | 0.74 | +0.77 | 2.6 | 5.3 |
| March 28, Houston Open | alert | alert | 1.62 | +1.03 | 3.4 | 7.7 |
| April 11, Masters | alert | alert | 3.47 | +1.21 | 4.0 | 8.6 |

Without CUSUM, detection in **4 tournaments**: the 8-tournament window dilutes the first
affected tournaments. A shorter window would detect faster but raise more
false alerts. With the CUSUM, **alert from the 1st affected week** (3 weeks earlier),
with no additional alert on the natural drift. Note: the CUSUM enters March at 1.9,
already loaded by the natural January drift; this is intended (a bias that comes back in the
same direction is more suspicious), but it should be said when reading this gain.

## Outputs

| File | Content |
|---|---|
| `reports/monitoring/latest.json` | latest check: status, reasons, signals, model (read by the trigger) |
| `reports/monitoring/summary_<scenario>.json` | same content, per replay scenario |
| `reports/monitoring/timeline_<scenario>.csv` / `.png` | one check per tournament |
| `reports/monitoring/calibration.json` | natural PSI distribution and thresholds per feature |
| MLflow, experiment `pga-monitoring` | one run per command, metrics per tournament (`step`) |
| Pushgateway (`check`, if `PUSHGATEWAY_URL`) | latest check, job `pga_monitor`, shown in Grafana (below) |

## Dashboard (Prometheus + Grafana)

`docker compose --profile observability up -d --build` adds Prometheus (:9090), a
Pushgateway (:9091) and Grafana (:3000, anonymous access, ports bound to the local machine).
Versioned configuration in `observability/`: `prometheus.yml`, data source and
"PGA MLOps" dashboard provisioned as files (no manual setup).

- **API**: Prometheus scrapes `api:8000/metrics` every 5 s (requests per route and
  status, inference latency, predicted rows per version, reloads, served version).
- **Monitoring**: `monitor check` is a short-lived process that Prometheus cannot scrape.
  If `PUSHGATEWAY_URL` is set (`http://pushgateway:9091` in the stack), it pushes its
  latest check:

| Metric | Content |
|---|---|
| `pga_drift_status` | 0 ok, 1 warning, 2 alert |
| `pga_monitor_model_version` | version of the checked champion |
| `pga_bias`, `pga_bias_z` | window bias (strokes) and its z-score |
| `pga_mae_ratio` | window MAE / validation MAE |
| `pga_psi_max{feature}` | PSI of the feature closest to its alert threshold |
| `pga_cusum{direction="up"\|"down"}` | CUSUM of the per-tournament bias |

Each push replaces the whole `job=pga_monitor` group: a signal that is not evaluated (journal too
short, scores not yet known) disappears instead of keeping a stale value. An unreachable
Pushgateway is reported on stderr without interrupting the check; `latest.json`
remains the source read by the trigger. Without `PUSHGATEWAY_URL`, nothing changes.

The dashboard uses the thresholds of this document (z = 2 / 3, MAE ratio 1.10, CUSUM 3 / 4).
`scripts/demo_live.sh` plays the incident on this stack: alert, rollback, back to ok
(README, "Demo on the visible stack").

## Commands

```bash
uv run python -m pga.monitor calibrate        # recompute the PSI thresholds (after a reference change)
uv run python -m pga.monitor replay --years 2023 2024
uv run python -m pga.monitor replay --years 2023 2024 --inject fc_wind_mean*3.6 --since 2024-03-01 --name incident
uv run python -m pga.monitor check            # production: API journal + actual scores
PUSHGATEWAY_URL=http://127.0.0.1:9091 uv run python -m pga.monitor check   # + push to the stack's Pushgateway
uv run python -m pga.monitor backfill-tags    # valid_event_bias_sd tag on older versions
```
