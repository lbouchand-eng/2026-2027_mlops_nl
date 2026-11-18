# Runbook: monitoring alert, rollback, retraining

## 1. Read the alert

```bash
uv run python -m pga.monitor check           # or: cat reports/monitoring/latest.json
```

`latest.json` gives the status, the reasons (which signal, which value, which threshold), the
served version and the last tournament checked. Per-tournament details are in MLflow,
experiment `pga-monitoring`.

## 2. Choose the response

| Situation | Typical signal | Response |
|---|---|---|
| Performance degraded **right after a promotion** | bias or MAE in alert, the previous version was fine | **Rollback** (§ 3) |
| PSI alert **right after a promotion** | the served model's training data differs from production | **Rollback** (§ 3), then check the backfill |
| Corrupted input data (feed bug, unit change) | PSI in alert on one or two features, suddenly, old champion | Fix the source. **No retraining**: the model would learn the bug. Rollback only if the previous version is less sensitive to it. |
| Slow, confirmed real-world drift | repeated bias warnings, moderate PSI | **Retraining** on a more recent window (§ 4) |
| New population (e.g. Q-School) | isolated PSI alert on form | Check, document, no action if performance holds |

## 3. Rollback

```bash
uv run python -m pga.rollback --reason "bias alert of 2024-03-28" --api-url http://127.0.0.1:8000
```

- `champion` ← `previous`, and `previous` ← former champion. `--to VERSION` picks another
  version.
- The API reloads the model (`POST /reload`), then the command checks that `/health` does
  serve the new version. If the API does not follow, the exit code is 2 and the manual
  reload command is printed.
- Trace: `rolled_back_from`, `rolled_back_at`, `rollback_reason` tags on the version
  put back into service, and a run in the MLflow experiment `pga-operations`.
- **Undo a rollback**: rerun the same command (the swap is symmetric).
- Explicit refusal (code 1) if there is no `previous`, if the target version does not exist or if
  it is already the champion.

Checks after a rollback:
1. `curl -s localhost:8000/health` → expected `model_version`.
2. `uv run python -m pga.monitor check` → the performance thresholds are those of the
   new version (champion's tags).
3. Compatibility: the API still sends the `pga.features.FEATURES` columns. MLflow
   enforces the served version's signature: extra columns are ignored (e.g. v1,
   trained with 21 features, accepts the current 23), a missing column makes the
   prediction fail. Before removing a feature, check which versions may still
   be put back into service.

Tested locally (November 4, 2026): v2 → v1 → v2, the API followed every time.

## 4. Retraining: the trigger

```bash
uv run python -m pga.trigger --dry-run          # decision only
uv run python -m pga.trigger                    # decides and acts
uv run python -m pga.trigger --as-of 2024-02-05 # simulation at a past date
```

The trigger recomputes monitoring "as-of" over the last 3 checks, then:

| Finding | Action | Code |
|---|---|---|
| nothing abnormal | nothing | 0 |
| bias/MAE degraded (alert, or 3 consecutive checks out of threshold), champion in service for more than 60 days | **retraining** | 10 |
| PSI alert on an input, champion in service for more than 60 days | **data incident**: no retraining, check the source | 20 |
| bias/MAE degraded or PSI alert, champion in service for less than 60 days | **rollback** (the latest deployment is suspect) | 30 |

Fair comparison: the decision window is made of the last 8 tournaments **finished**
at the date (last tee time + 6 h): an ongoing tournament would bring scores of rounds not
yet played.
The challenger is trained on everything before it, and **champion and challenger are
evaluated on this same window**. Promotion if the MAE is better and the |bias| no worse.
The window is read from the API journal if it contains at least one scored row for each of the
8 tournaments and `monitor.min_rows` rows in total. Otherwise the comparison uses the feature table,
and the `decision_source=backfill` tag flags it.
Trace: run `retrain as-of …` in `pga-score`, decision in `pga-operations`,
`reports/trigger/latest.json` and an archive per executed decision.

Actual result (`reports/trigger/2024-02-05_retrain.json`): on Monday, February 5, 2024, after
three consecutive bias warnings (z = 2.0 to 2.2, the last on the AT&T Pebble Beach
Pro-Am finished on February 3), the challenger (data prior to November 9, 2023, start of
the decision window) beats the champion on the same 8 tournaments: MAE 2.326 versus 2.362,
bias +0.48 versus +0.63. It is promoted and the former champion becomes `previous`. Afterwards,
no action until the end of 2024: the mean 2024 bias goes from +0.22 to +0.16 strokes.
On February 2, Pebble Beach is still in progress: the trigger only sees two warnings and
does nothing (`test_event_in_progress_is_not_used`).

Promotion guardrail: if the wind median leaves its range or if the PSI between the
challenger's data and the champion's profile exceeds the alert threshold, the challenger stays
`candidate` (tag `blocked_reason`), even if it wins the comparison.

Known limitation: a corruption of the **target** does not change the features, so this guardrail does not
see it. The comparison on the journal refuses it
(`tests/test_trigger.py::test_journal_catches_what_the_backfill_hides`), but without a journal the
challenger can win (`test_corrupted_backfill_fools_the_comparison`). Only production monitoring
detects it then; the rule "degradation shortly after a deployment → rollback"
takes over.

## 5. Automation and alerts

| Mechanism | Frequency | Effect |
|---|---|---|
| Compose service `scheduler` (`--profile ops`) | every week (`TRIGGER_INTERVAL_SECONDS`) | decides **and acts**, reloads the API |
| GitHub workflow `trigger.yml` | Monday 6:00 UTC, or on demand | decides without acting |

`PGA_ALERT_WEBHOOK` receives the `monitor check` alerts, the trigger's actions and its failures.
An unreachable webhook never interrupts a decision.

Limitation: the pipeline stops at the 2024 season (`data-v1`). In production, a scheduled `dvc repro`
should run before the trigger to feed it the recent tournaments.
