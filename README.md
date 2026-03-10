# 🎯 Customer Churn Prediction
**Prédiction du risque de départ des clients bancaires par Machine Learning**

> Dataset : 10 000 clients · Meilleur modèle : LightGBM · AUC-ROC : 0.87

---

## 📊 Résultats

```
🏆 LightGBM (modèle retenu)
├─ AUC-ROC   : 0.8715  ⭐
├─ Recall    : 76.9%   (détecte ~8 churns sur 10)
├─ Precision : 51.9%
├─ F1-Score  : 0.620
└─ Accuracy  : 80.8%
```

### Comparaison des 3 finalistes (après tuning)

| Modèle        | AUC-ROC | Recall | Precision | F1-Score | Accuracy |
|---------------|---------|--------|-----------|----------|----------|
| **LightGBM** 🏆 | **0.8715** | **76.9%** | 51.9% | **0.620** | 80.8% |
| Random Forest | 0.8676  | 69.3%  | 58.6%     | 0.635    | 83.8%    |
| SVM           | 0.8597  | 78.4%  | 47.3%     | 0.590    | 77.9%    |

**Décision finale : LightGBM** — Perdre un client coûte structurellement plus cher que de le contacter inutilement. Dans ce contexte, maximiser le Recall prime sur la Precision. LightGBM rate 31 churners de moins que Random Forest, ce qui justifie pleinement les fausses alertes supplémentaires générées.

> Random Forest reste défendable si les campagnes de rétention sont particulièrement coûteuses et nécessitent un ciblage chirurgical.

---

## 🔧 Stack Technique

- **ML** : scikit-learn · XGBoost · LightGBM · SVM
- **Data** : pandas · numpy · matplotlib · seaborn
- **Méthode** : Feature engineering · Hyperparameter tuning (RandomizedSearchCV) · Gestion déséquilibre (`class_weight='balanced'`)
- **Environnement** : Python 3.12 · Jupyter Notebook

---

## 📈 Méthodologie

### 1. Preprocessing & Feature Engineering
- Dataset : 10 000 clients, 12 variables brutes, **aucune valeur manquante**
- Encodage des variables catégorielles : `country` (one-hot), `gender` (binaire)
- Nouvelles features créées :
  - `has_balance` — client avec solde > 0
  - `active_products` — croisement activité × nombre de produits
  - `age_group` — segmentation en 5 tranches (young / middle / senior / old / very_old)
  - `tenure_group` — segmentation de l'ancienneté (new / medium / loyal)

### 2. Split & Scaling
- Split stratifié : **80% train / 20% test** (distribution churn préservée : ~20/80)
- StandardScaler appliqué aux features numériques continues pour les modèles linéaires et SVM

### 3. Benchmark — 8 modèles comparés

| Modèle | AUC-ROC | Recall | Notes |
|--------|---------|--------|-------|
| LightGBM | 0.8589 | 71.5% | Meilleur compromis |
| Random Forest | 0.8537 | 43.0% | Trop conservateur |
| SVM | 0.8531 | 73.7% | Précision faible |
| XGBoost | 0.8263 | 63.9% | |
| SGD | 0.7956 | 73.5% | |
| Logistic Regression | 0.7946 | 71.3% | |
| Naive Bayes | 0.7918 | 50.6% | |
| Decision Tree | 0.6600 | 46.4% | |

### 4. Sélection & Tuning
- **3 finalistes retenus** : LightGBM, Random Forest, SVM
- Optimisation via **RandomizedSearchCV + StratifiedKFold** (5 folds, scoring = `roc_auc`)
- Sauvegarde du meilleur modèle : `best_model.pkl` + `model_info.pkl`

---

## 📂 Structure du Projet

```
customer-churn-prediction/
├── data/
│   ├── raw/                    # Données brutes
│   └── processed/
│       └── churn_dataset_clean.csv
├── models/
│   ├── best_model.pkl          # LightGBM entraîné
│   └── model_info.pkl          # Métriques & métadonnées
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb       # ← Ce notebook
└── README.md
```

---

## 🚀 Quick Start

```bash
git clone https://github.com/[username]/customer-churn-prediction.git
cd customer-churn-prediction
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn jupyter
jupyter notebook notebooks/03_modeling.ipynb
```

---

## 🚀 Perspectives d'Amélioration

### 📊 Métriques
- **PR-AUC** (Precision-Recall AUC) : métrique plus fiable que ROC-AUC sur dataset déséquilibré (20/80), car le ROC-AUC peut être optimiste en comptant les vrais négatifs très nombreux

### 🤖 Modélisation
- **CatBoost** : optimisé pour variables catégorielles
- **Stacking / Blending** : combiner LightGBM + Random Forest
- **Calibration des probabilités** : améliorer la fiabilité des scores de risque
- **SHAP values** : explications individuelles par client (au-delà de la feature importance globale)

### 🔬 Feature Engineering
- Features temporelles : récence des transactions, tendances
- Interactions : `age × balance`, `tenure × products`
- **Customer Lifetime Value (CLV)** : croiser risque churn × valeur client

### ⚖️ Déséquilibre
- Comparer `class_weight` vs SMOTE, ADASYN, Borderline-SMOTE
- Optimisation multi-objectif (maximiser Recall ET Precision simultanément)

### 🚀 Production
- **API REST** (FastAPI) : scoring en temps réel
- **Dashboard** (Streamlit / Dash) : visualisation pour équipes métier
- **Pipeline MLOps** : réentraînement automatique mensuel, monitoring drift
- **Docker** : déploiement reproductible

---

## 📌 Compétences Démontrées

- ✅ Feature engineering & gestion du déséquilibre de classes
- ✅ Benchmark multi-modèles & sélection argumentée
- ✅ Hyperparameter tuning (RandomizedSearchCV)
- ✅ Évaluation ML : Recall, AUC-ROC, F1, Precision
- ✅ Interprétabilité (feature importance)
- ✅ Raisonnement business orienté ROI
