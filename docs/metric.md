# Choice of metric (set before any training)

**Target**: `to_par`, a player's score on a round relative to par (in strokes).

## Main metric: MAE (mean absolute error, in strokes)

$$\text{MAE} = \frac{1}{n}\sum_i |\hat{y}_i - y_i|$$

Why the MAE:

1. **Readable by a non-technical audience**: for example "the model is off by 2.4 strokes per
   round on average". The unit is the target's.
2. **Robust to disaster rounds.** The distribution of `to_par` has a heavy right
   tail (standard deviation 3.3, maximum +22: injury, undeclared withdrawal, extreme weather).
   These rounds are unpredictable by nature; an RMSE would square them and make
   the metric, and therefore the model choice and the alerts, depend on a few rows.
3. **Stable over time**, hence usable as an **alert threshold in production**:
   a rise in the rolling MAE over the last tournaments signals a real degradation
   rather than noise.
4. It estimates the conditional median, which suits an asymmetric target.

## Reference: naive baseline and skill score

An MAE alone means nothing without a point of comparison. The **baseline** predicts, for each
round, the historical mean of `to_par` on the course (previous editions only,
0 if the course is new). It knows neither the player nor the weather.

| Year | Baseline MAE | "Predict 0" MAE | Baseline RMSE |
|------|--------------|-----------------|---------------|
| 2021 | 2.527        | 2.626           | 3.224         |
| 2022 | 2.564        | 2.666           | 3.247         |
| 2023 | 2.518        | 2.642           | 3.204         |
| 2024 | 2.508        | 2.717           | 3.182         |

We report the **skill**: `skill = 1 − MAE_model / MAE_baseline`.

**Promotion rule**: a model is promoted in the MLflow Model Registry only if its
validation skill is strictly positive **and** better than that of the model in
production.

## Secondary metrics (logged, not used for decisions)

- **RMSE**: to see the effect of large errors.
- **Mean bias** (`mean(ŷ − y)`): level-drift signal. The field scores lower and
  lower (mean −0.40 in 2022, −0.77 in 2024); a growing positive bias means the
  model is lagging behind this shift.
- **MAE per round (R1 to R4)**: after the cut, the field is more homogeneous.

## Evaluation protocol (temporal split, never random)

| Period    | Role |
|-----------|------|
| 2016-2017 | history to compute the features (form, course); no training |
| 2018-2021 | training |
| 2022      | validation (hyperparameter choice, promotion decision) |
| 2018-2022 | reference period: the registered model is retrained on it |
| 2023-2024 | drift period: per-tournament evaluation, feeds monitoring |

### Revision of 2026-11-03

The initial protocol trained on 2021 only (2016-2020 serving as history). After
experimenting, the training window is extended to **2018-2021**: it is the only
significant lever found (validation MAE 2.411 → 2.380, skill +6.0% → +7.2%), along with two
baseline-level player features. 2018 is the first year where the weather forecast is 100% available.

**The metric, the baseline, the validation year and the promotion rule are
unchanged.** The new model went through the same rule as the previous one (see the
Registry: v1 is kept under the `previous` alias).

Leads tested without gain (validation 2022, MAE): weather specific to the tee-time wave
(2.415), field strength and size (2.416), previous round of the same tournament (2.415),
hyperparameters (2.410 to 2.424), squared loss (2.424). Even the **observed** weather
(information unavailable in production) gains only 0.007 strokes: a round's score is
dominated by irreducible noise.
