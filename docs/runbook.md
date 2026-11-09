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
| Alerte PSI **juste après une promotion** | les données d'entraînement du modèle servi diffèrent de la production | **Rollback** (§ 3), puis vérifier le backfill |
| Données d'entrée corrompues (bug de flux, changement d'unité) | PSI en alerte sur une ou deux features, d'un coup, champion ancien | Corriger la source. **Pas de réentraînement** : on apprendrait le bug. Rollback seulement si la version précédente y est moins sensible. |
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

## 4. Réentraînement : le déclencheur

```bash
uv run python -m pga.trigger --dry-run          # décision seule
uv run python -m pga.trigger                    # décide et agit
uv run python -m pga.trigger --as-of 2024-02-05 # simulation à une date passée
```

Le déclencheur recalcule le monitoring « as-of » sur les 3 derniers contrôles, puis :

| Constat | Action | Code |
|---|---|---|
| rien d'anormal | rien | 0 |
| biais/MAE dégradés (alerte, ou 3 contrôles consécutifs hors seuil), champion en service depuis plus de 60 j | **réentraînement** | 10 |
| alerte PSI sur une entrée, champion en service depuis plus de 60 j | **incident de données** : pas de réentraînement, vérifier la source | 20 |
| biais/MAE dégradés ou alerte PSI, champion en service depuis moins de 60 j | **rollback** (le dernier déploiement est suspect) | 30 |

Comparaison juste : la fenêtre de décision est formée des 8 derniers tournois **terminés**
à la date (dernier départ + 6 h) : un tournoi en cours apporterait les scores de tours pas
encore joués.
Le challenger est entraîné sur tout ce qui précède, et **champion et challenger sont
évalués sur cette même fenêtre**. Promotion si la MAE est meilleure et le |biais| pas pire.
La fenêtre est lue dans le journal de l'API s'il contient au moins une ligne notée pour chacun des
8 tournois et `monitor.min_rows` lignes au total. Sinon la comparaison utilise la table de features,
et le tag `decision_source=backfill` le signale.
Trace : run `retrain as-of …` dans `pga-score`, décision dans `pga-operations`,
`reports/trigger/latest.json` et une archive par décision exécutée.

Résultat réel (`reports/trigger/2024-02-05_retrain.json`) : le lundi 5 février 2024, après
trois warnings de biais consécutifs (z = 2,0 à 2,2, le dernier sur l'AT&T Pebble Beach
Pro-Am terminé le 3 février), le challenger (données antérieures au 9 novembre 2023, début de
la fenêtre de décision) bat le champion sur les mêmes 8 tournois : MAE 2,326 contre 2,362,
biais +0,48 contre +0,63. Il est promu et l'ancien champion passe en `previous`. Ensuite,
aucune action jusqu'à fin 2024 : le biais moyen 2024 passe de +0,22 à +0,16 coup.
Au 2 février, Pebble Beach est encore en cours : le déclencheur ne voit que deux warnings et
ne fait rien (`test_event_in_progress_is_not_used`).

Garde-fou de promotion : si la médiane du vent sort de sa plage ou si le PSI entre les données
du challenger et le profil du champion dépasse le seuil d'alerte, le challenger reste
`candidate` (tag `blocked_reason`), même s'il gagne la comparaison.

Limite connue : une corruption de la **cible** ne change pas les features, donc ce garde-fou ne
la voit pas. La comparaison sur le journal la refuse
(`tests/test_trigger.py::test_journal_catches_what_the_backfill_hides`), mais sans journal le
challenger peut gagner (`test_corrupted_backfill_fools_the_comparison`). Seul le monitoring en
production le détecte alors ; la règle « dégradation peu après une mise en service → rollback »
prend le relais.

## 5. Automatisation et alertes

| Mécanisme | Fréquence | Effet |
|---|---|---|
| Service Compose `scheduler` (`--profile ops`) | chaque semaine (`TRIGGER_INTERVAL_SECONDS`) | décide **et agit**, recharge l'API |
| Workflow GitHub `trigger.yml` | lundi 6 h UTC, ou à la demande | décide sans agir |

`PGA_ALERT_WEBHOOK` reçoit les alertes de `monitor check`, les actions du déclencheur et ses échecs.
Un webhook injoignable n'interrompt jamais une décision.

Limite : le pipeline s'arrête à la saison 2024 (`data-v1`). En production, un `dvc repro` planifié
devrait précéder le déclencheur pour lui fournir les tournois récents.
