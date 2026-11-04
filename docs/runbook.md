# Runbook : alerte de monitoring, rollback, réentraînement

## 1. Lire l'alerte

```bash
uv run python -m pga.monitor check           # ou : cat reports/monitoring/latest.json
```

`latest.json` donne le statut, les raisons (quel signal, quelle valeur, quel seuil), la
version servie et le dernier tournoi contrôlé. Le détail par tournoi est dans MLflow,
expérience `pga-monitoring`.

## 2. Choisir la réaction

| Situation | Signal typique | Réaction |
|---|---|---|
| Performance dégradée **juste après une promotion** | biais ou MAE en alerte, la version précédente allait bien | **Rollback** (§ 3) |
| Données d'entrée corrompues (bug de flux, changement d'unité) | PSI en alerte sur une ou deux features, d'un coup | Corriger la source. **Pas de réentraînement** : on apprendrait le bug. Rollback seulement si la version précédente y est moins sensible. |
| Dérive lente et confirmée du monde réel | warnings de biais répétés, PSI modéré | **Réentraînement** sur une fenêtre plus récente (§ 4) |
| Nouvelle population (ex. Q-School) | alerte PSI sur la forme, isolée | Vérifier, documenter, pas d'action si la performance tient |

## 3. Rollback

```bash
uv run python -m pga.rollback --reason "alerte biais du 2024-03-28" --api-url http://127.0.0.1:8000
```

- `champion` ← `previous`, et `previous` ← ancien champion. `--to VERSION` choisit une
  autre version.
- L'API recharge le modèle (`POST /reload`), puis la commande vérifie que `/health` sert
  bien la nouvelle version. Si l'API ne suit pas, le code de sortie vaut 2 et la commande
  de rechargement manuel est affichée.
- Trace : tags `rolled_back_from`, `rolled_back_at`, `rollback_reason` sur la version
  remise en service, et un run dans l'expérience MLflow `pga-operations`.
- **Annuler un rollback** : relancer la même commande (l'échange est symétrique).
- Refus explicite (code 1) si pas de `previous`, si la version cible n'existe pas ou si
  elle est déjà le champion.

Vérifications après un rollback :
1. `curl -s localhost:8000/health` → `model_version` attendu.
2. `uv run python -m pga.monitor check` → les seuils de performance sont ceux de la
   nouvelle version (tags du champion).
3. Compatibilité : l'API envoie toujours les colonnes de `pga.features.FEATURES`. MLflow
   applique la signature de la version servie : les colonnes en trop sont ignorées (ex. v1,
   entraînée avec 21 features, accepte les 23 actuelles), une colonne manquante fait
   échouer la prédiction. Avant de supprimer une feature, vérifier les versions encore
   susceptibles d'être remises en service.

Testé en local (4 novembre 2026) : v2 → v1 → v2, l'API a suivi à chaque fois.

## 4. Réentraînement

On décale toutes les fenêtres d'un an, pour que la validation porte sur la période qui a
dérivé :

```bash
uv run python -m pga.train --train-years 2018 2022 --valid-years 2023 2023 --refit-years 2018 2023 \
  --run-name "retrain après dérive 2023"
```

Attention : seul `--refit-years` ne suffit pas. La validation resterait 2022, avec la même
MAE, et le nouveau modèle ne serait jamais promu. Autre limite : la règle de promotion
actuelle compare la MAE de validation du challenger (2023) à celle du champion (2022),
donc sur deux années différentes. Le déclencheur de réentraînement (étape suivante)
évaluera les deux modèles **sur la même fenêtre récente** avant de décider. Après une
promotion : `POST /reload`.
