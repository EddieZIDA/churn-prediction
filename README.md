# Customer Churn Prediction

**Prédiction du risque de départ des clients bancaires par Machine Learning**

[![Application en ligne](https://img.shields.io/badge/Streamlit-Application%20en%20ligne-ff4b4b?logo=streamlit)](https://churn-prediction7.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![LightGBM](https://img.shields.io/badge/LightGBM-AUC--ROC%200.87-success)
![Tests](https://img.shields.io/badge/tests-123%20%C2%B7%20couverture%2089%25-success)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> Dataset : 10 000 clients · Meilleur modèle : LightGBM · AUC-ROC : 0.87

---

## Résultats

```
LightGBM (modèle retenu)

Pouvoir discriminant (indépendant du seuil)
├─ PR-AUC    : 0.719    (hasard : 0.204 = taux de churn)
├─ AUC-ROC   : 0.8707   (hasard : 0.500)
└─ Brier     : 0.0993   probabilités calibrées

Au seuil de décision retenu (0.10, calibré sur le coût métier)
├─ Recall    : 89.9%    (détecte 9 churners sur 10)
├─ Precision : 34.8%
└─ Coût      : 54 750 € contre 108 500 € au seuil naïf de 0.50
```

Le **PR-AUC** est la métrique de référence ici plutôt que l'AUC-ROC : avec
20,4 % de positifs, l'AUC-ROC intègre un taux de vrais négatifs gonflé par la
classe majoritaire et flatte le modèle. Sa référence aléatoire est le taux de
churn lui-même, pas 0.5.

### Comparaison des 3 finalistes (après tuning)

| Modèle        | PR-AUC    | AUC-ROC   | Brier      | Recall | Precision |
|---------------|-----------|-----------|------------|--------|-----------|
| **LightGBM**  | **0.7190**| **0.8707**| **0.0993** | 47.9 % | 79.6 %    |
| Random Forest | 0.7120    | 0.8645    | 0.1009     | 44.0 % | 80.3 %    |
| SVM           | 0.6935    | 0.8321    | 0.1061     | 37.8 % | 85.1 %    |

> Recall et Precision sont ici mesurés au seuil par défaut de 0.5. Le modèle
> déployé applique le seuil de 0.10 calibré sur le coût métier, qui porte le
> recall à 89,9 %.

**Décision finale : LightGBM** — perdre un client coûte environ dix fois plus
cher que de le contacter inutilement (500 € contre 50 €), ce qui justifie de
privilégier le recall. LightGBM devance ses deux concurrents sur le PR-AUC et
produit les probabilités les mieux calibrées.

> Les écarts entre les trois finalistes (0.719 / 0.712 / 0.694 de PR-AUC)
> sont du même ordre que l'incertitude d'échantillonnage : sur 2 000 clients
> de test, l'intervalle de confiance à 95 % du PR-AUC est large d'environ
> ±0.04. Ce classement indique une tendance, pas une supériorité démontrée.

---

## Interprétabilité SHAP

Les 3 features les plus influentes identifiées par SHAP sur le modèle LightGBM final :

| Feature | Direction | Interprétation |
|---------|-----------|----------------|
| `age` | 🔺 Risque croissant jusqu'à 50-60 ans | Pic à 58,6 % de churn sur la tranche 50-60 ans, puis repli au-delà. Les moins de 30 ans sont les plus stables (6,3 %) |
| `products_2` | 🔻 Risque fortement réduit | Détenir exactement 2 produits est la situation la plus stable : 8,4 % de churn, contre 30,5 % pour tous les autres cas |
| `active_member` | 🔻 Risque réduit | Un client actif churne à 13,7 %, un inactif à 27,5 % |

> Ces valeurs proviennent de la sortie du notebook `03_modeling.ipynb`, et non
> d'une lecture visuelle du graphique SHAP.

---

## Stack Technique

- **ML** : scikit-learn · XGBoost · LightGBM · SVM
- **Interprétabilité** : SHAP (TreeExplainer)
- **Tracking** : MLflow
- **App** : Streamlit
- **Data** : pandas · numpy · matplotlib · seaborn
- **Méthode** : Feature engineering · RandomizedSearchCV (scoring `average_precision`) · seuil calibré sur le coût métier
- **Environnement** : Python 3.12 · Jupyter Notebook

---

## Méthodologie

### 1. Preprocessing & Feature Engineering

- Dataset : 10 000 clients, 12 variables brutes, **aucune valeur manquante**
- Encodage des variables catégorielles : `country` (OneHotEncoder), `gender` (binaire)
- Nouvelles features créées :
  - `has_balance` : client avec solde > 0
  - `active_products` : croisement activité × nombre de produits
  - `age_group` : segmentation en 5 tranches (young / middle / senior / old / very_old)
  - `tenure_group` : segmentation de l'ancienneté (new / medium / loyal)
  - `products_2` / `products_3_plus` : modalités du nombre de produits,
    dont la relation avec le churn est en U

### 2. Split & Normalisation

- Split stratifié : **80% train / 20% test**
- StandardScaler appliqué aux features numériques continues (modèles linéaires et SVM uniquement)

### 3. Benchmark — 8 modèles comparés

Classés par PR-AUC, avec `class_weight='balanced'` pour tous afin de comparer
à armes égales :

| Modèle              | PR-AUC | AUC-ROC | Recall |
|---------------------|--------|---------|--------|
| LightGBM            | 0.7015 | 0.8589  | 70.0 % |
| Logistic Regression | 0.6794 | 0.8479  | 73.7 % |
| Random Forest       | 0.6792 | 0.8503  | 44.0 % |
| SVM                 | 0.6760 | 0.8515  | 75.4 % |
| SGD                 | 0.6741 | 0.8467  | 72.5 % |
| XGBoost             | 0.6548 | 0.8341  | 61.4 % |
| Naive Bayes         | 0.6000 | 0.8224  | 44.5 % |
| Decision Tree       | 0.3383 | 0.6740  | 47.7 % |

> La régression logistique gagne 5 points d'AUC-ROC par rapport à une version
> antérieure de ce benchmark. La raison n'est pas un changement de modèle mais
> d'encodage : `products_number` lui est fourni sous forme de modalités
> (`products_2`, `products_3_plus`). Sa relation avec le churn étant en U,
> un modèle linéaire ne pouvait pas l'exprimer avec un coefficient unique.

### 4. Sélection & Tuning

- **3 finalistes retenus** : LightGBM, Random Forest, SVM
- Optimisation via **RandomizedSearchCV + StratifiedKFold** (5 folds, scoring = `average_precision`)
- Seuil de décision choisi **en validation croisée sur le train**, puis appliqué au test —
  le choisir sur le test donnerait un coût optimiste et non reproductible
- Sauvegarde du meilleur modèle : `best_model.pkl` + `model_info.pkl`, métriques mesurées à l'exécution

---

## Lancer le projet

### Installation

```bash
git clone https://github.com/EddieZIDA/churn-prediction.git
cd churn-prediction
python -m venv venv
venv/Scripts/activate        # Linux/macOS : source venv/bin/activate
pip install -r requirements.txt
```

Le dataset n'est pas versionné : voir la [section Données du rapport](REPORT.md#5-données) pour le récupérer.

Toutes les commandes ci-dessous se lancent **depuis la racine du projet**.
Sans activer l'environnement, préfixez-les par `venv/Scripts/python.exe -m`
(`venv/bin/python -m` sous Linux/macOS) : le chemin explicite garantit que
c'est bien l'environnement du projet qui s'exécute.

### Lancer l'app Streamlit

```bash
streamlit run streamlit_app.py
```

L'application est servie sur `http://localhost:8501`.

### Lancer les notebooks

```bash
jupyter lab notebooks/
```

Sélectionnez le noyau correspondant à `venv/`. Sous VS Code, ouvrez
directement le `.ipynb` et choisissez le même interpréteur — `ipykernel` est
déjà installé, rien d'autre n'est nécessaire.

Pour une exécution sans interface, par exemple pour régénérer les sorties :

```bash
cd notebooks
jupyter nbconvert --to notebook --execute --inplace 01_eda.ipynb
```

> `03_modeling.ipynb` réentraîne les modèles et **écrase
> `models/best_model.pkl`**. Comptez une dizaine de minutes, le tuning du SVM
> étant le plus lent. Les notebooks 01 et 02 sont sans effet de bord : 02
> réécrit le dataset traité à l'identique.

### Tracking MLflow

Les runs sont enregistrés dans `notebooks/mlruns/`, il faut donc lancer l'UI
depuis ce dossier :

```bash
cd notebooks && mlflow ui
```

Ouvre `http://localhost:5000` pour visualiser les runs, paramètres et métriques.

### Tests

```bash
pytest tests/ -v
```

---

## Qualité & industrialisation

- **Code modulaire** : 9 modules `src/`, importés aussi bien par les notebooks que par l'application
- **Décision séparée de la prédiction** : le ciblage par valeur attendue réduit le coût de 4,9 % sans réentraînement
- **123 tests, 89 % de couverture**, 0 violation PEP8
- **Pas de train/serve skew** : un seul feature engineering, verrouillé par des tests de parité
- **Robustesse** : logging centralisé, validation Pydantic, exceptions typées
- **CI GitHub Actions** : lint, tests, seuil de couverture 80 %, analyse de sécurité
- **Déploiement** : Streamlit Cloud automatique depuis `main`, image Docker disponible

```bash
pytest tests/ -v --cov=src                      # tests + couverture
black --check src/ tests/ streamlit_app.py      # format
flake8 src/ tests/ streamlit_app.py             # lint
pre-commit install                              # contrôles avant chaque commit
```

---

## Documentation

| Document | Contenu |
|----------|---------|
| [REPORT.md](REPORT.md) | Rapport technique : audit, corrections, architecture, données, qualité, contribution |
| [DEPLOYMENT.md](DEPLOYMENT.md) | Mise en production : Streamlit Cloud, Docker, dépannage |

---

## Suite du projet

Détection de drift à partir de `logs/predictions.csv`, pipeline de
réentraînement orchestré, API de scoring par lot.
Détail dans [REPORT.md](REPORT.md#8-ce-qui-reste).

---

## Auteur

**ZIDA Wend Kouni Eddie Eliel**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?logo=linkedin)](https://www.linkedin.com/in/wend-kouni-eddie-eliel-zida-501815260/?skipRedirect=true)

*Projet de mise en pratique des bonnes pratiques Data Science en conditions professionnelles · 2026*
