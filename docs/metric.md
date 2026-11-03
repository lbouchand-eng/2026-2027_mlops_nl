# Choix de la métrique (fixé avant tout entraînement)

**Cible** : `to_par`, score d'un joueur sur un tour par rapport au par (en coups).

## Métrique principale : MAE (erreur absolue moyenne, en coups)

$$\text{MAE} = \frac{1}{n}\sum_i |\hat{y}_i - y_i|$$

Pourquoi la MAE :

1. **Lisible par un métier non technique** : par exemple « le modèle se trompe en moyenne de 2,4 coups
   par tour ». L'unité est celle de la cible.
2. **Robuste aux tours catastrophes.** La distribution de `to_par` a une queue lourde à
   droite (écart-type 3,3, maximum +22 : blessure, abandon non déclaré, météo extrême).
   Ces tours sont imprévisibles par nature ; une RMSE les élèverait au carré et ferait
   dépendre la métrique, donc le choix du modèle et les alertes, de quelques lignes.
3. **Stable dans le temps**, donc utilisable comme **seuil d'alerte en production** :
   une hausse de la MAE glissante sur les derniers tournois signale une dégradation réelle
   plutôt que du bruit.
4. Elle estime la médiane conditionnelle, ce qui convient à une cible asymétrique.

## Référence : baseline naïve et score de skill

Une MAE seule n'a pas de sens sans point de comparaison. La **baseline** prédit, pour chaque
tour, la moyenne historique de `to_par` sur le parcours (éditions précédentes uniquement,
0 si le parcours est nouveau). Elle ne connaît ni le joueur ni la météo.

| Année | MAE baseline | MAE « prédire 0 » | RMSE baseline |
|-------|--------------|-------------------|---------------|
| 2021  | 2,527        | 2,626             | 3,224         |
| 2022  | 2,564        | 2,666             | 3,247         |
| 2023  | 2,518        | 2,642             | 3,204         |
| 2024  | 2,508        | 2,717             | 3,182         |

On rapporte le **skill** : `skill = 1 − MAE_modèle / MAE_baseline`.

**Règle de promotion** : un modèle n'est promu dans le MLflow Model Registry que si son
skill sur la validation est strictement positif **et** meilleur que celui du modèle en
production.

## Métriques secondaires (loggées, non décisionnelles)

- **RMSE** : pour voir l'effet des grosses erreurs.
- **Biais moyen** (`mean(ŷ − y)`) : signal de drift de niveau. Le field joue de plus en plus
  bas (moyenne −0,40 en 2022, −0,77 en 2024) ; un biais positif croissant indique que le
  modèle est en retard sur ce décalage.
- **MAE par tour (R1 à R4)** : après le cut, le field est plus homogène.

## Protocole d'évaluation (split temporel, jamais aléatoire)

| Période   | Rôle |
|-----------|------|
| 2016-2020 | historique pour calculer les features (forme, parcours) ; pas d'entraînement |
| 2021      | entraînement |
| 2022      | validation (choix des hyperparamètres, décision de promotion) |
| 2023-2024 | période de drift : évaluation par tournoi, alimente le monitoring |
