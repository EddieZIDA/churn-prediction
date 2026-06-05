# Customer Churn Prediction

**Prédiction du risque de départ des clients bancaires par Machine Learning**

(https://churn-prediction7.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![LightGBM](https://img.shields.io/badge/LightGBM-AUC--ROC%200.87-success)
![License](https://img.shields.io/badge/License-MIT-green)

> Dataset : 10 000 clients · Meilleur modèle : LightGBM · AUC-ROC : 0.87

---

## Résultats

```

LightGBM (modèle retenu)
├─ AUC-ROC   : 0.8715  
├─ Recall    : 76.9%   (détecte 8 churns sur 10)
├─ Precision : 51.9%
├─ F1-Score  : 0.620
└─ Accuracy  : 80.8%
```

### Comparaison des 3 finalistes (après tuning)

| Modèle        | AUC-ROC    | Recall    | Precision | F1-Score | Accuracy |
|---------------|------------|-----------|-----------|----------|----------|
| **LightGBM**  | **0.8715** | **76.9%** | 51.9%     | **0.620**| 80.8%    |
| Random Forest | 0.8676     | 69.3%     | 58.6%     | 0.635    | 83.8%    |
| SVM           | 0.8597     | 78.4%     | 47.3%     | 0.590    | 77.9%    |

**Décision finale : LightGBM** — Perdre un client coûte plus cher que de le contacter inutilement.
Dans ce contexte, maximiser le Recall prime sur la Precision. LightGBM rate 31 churners de moins que Random Forest.

> Random Forest reste acceptable si les campagnes de rétention sont particulièrement coûteuses
> et nécessitent un ciblage très précis.

---

## Interprétabilité SHAP

Les 3 features les plus influentes identifiées par SHAP sur le modèle LightGBM final :

| Feature | Direction | Interprétation |
|---------|-----------|----------------|
| `has_balance` | 🔺 Risque élevé si = 0 | Un client sans solde actif est moins engagé et plus susceptible de partir |
| `age_group_very_old` | 🔺 Risque élevé | Le segment très âgé présente des comportements de churn spécifiques |
| `tenure_group_loyal` | 🔻 Risque réduit | Plus un client est ancien, moins il est enclin à churner |

> Seuil de décision optimal ajusté selon le coût métier (FN = 500€ vs FP = 50€)
> pour maximiser la valeur business plutôt que le score mathématique brut.

---

## Stack Technique

- **ML** : scikit-learn · XGBoost · LightGBM · SVM
- **Interprétabilité** : SHAP (TreeExplainer)
- **Tracking** : MLflow
- **App** : Streamlit
- **Data** : pandas · numpy · matplotlib · seaborn
- **Méthode** : Feature engineering · RandomizedSearchCV · `class_weight='balanced'`
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

### 2. Split & Normalisation

- Split stratifié : **80% train / 20% test**
- StandardScaler appliqué aux features numériques continues (modèles linéaires et SVM uniquement)

### 3. Benchmark — 8 modèles comparés

| Modèle              | AUC-ROC | Recall |
|---------------------|---------|--------|
| LightGBM            | 0.8589  | 71.5%  |
| Random Forest       | 0.8537  | 43.0%  |
| SVM                 | 0.8531  | 73.7%  |
| XGBoost             | 0.8263  | 63.9%  |
| SGD                 | 0.7956  | 73.5%  |
| Logistic Regression | 0.7946  | 71.3%  |
| Naive Bayes         | 0.7918  | 50.6%  |
| Decision Tree       | 0.6600  | 46.4%  |

### 4. Sélection & Tuning

- **3 finalistes retenus** : LightGBM, Random Forest, SVM
- Optimisation via **RandomizedSearchCV + StratifiedKFold** (5 folds, scoring = `roc_auc`)
- Sauvegarde du meilleur modèle : `best_model.pkl` + `model_info.pkl`

---

## Lancer le projet

```bash
git clone https://github.com/EddieZIDA/churn-prediction.git
cd churn-prediction
pip install -r requirements.txt
jupyter notebook notebooks/03_modeling.ipynb
```

### Lancer l'app Streamlit

```bash
streamlit run app.py
```

### Tracking MLflow

```bash
mlflow ui
```

Ouvre `http://localhost:5000` pour visualiser les runs, paramètres et métriques.

### Tests

```bash
pytest tests/ -v
```

---

## Auteur

**ZIDA Wend Kouni Eddie Eliel**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?logo=linkedin)](https://www.linkedin.com/in/wend-kouni-eddie-eliel-zida-501815260/?skipRedirect=true)

---

*Projet de mise en pratique des bonnes pratiques Data Science en conditions professionnelles · 2026*
