# Customer Churn Prediction
**Prédiction du risque de départ des clients bancaires par Machine Learning**
Link: https://churn-prediction7.streamlit.app/

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

| Modèle        | AUC-ROC | Recall | Precision | F1-Score | Accuracy |
|---------------|---------|--------|-----------|----------|----------|
| **LightGBM**  | **0.8715** | **76.9%** | 51.9% | **0.620** | 80.8% |
| Random Forest | 0.8676  | 69.3%  | 58.6%     | 0.635    | 83.8%    |
| SVM           | 0.8597  | 78.4%  | 47.3%     | 0.590    | 77.9%    |

**Décision finale : LightGBM** - Perdre un client coûte plus cher que de le contacter inutilement. Dans ce contexte, maximiser le Recall prime sur la Precision. LightGBM rate 31 churners de moins que Random Forest.

> Random Forest reste acceptable si les campagnes de rétention sont particulièrement coûteuses et nécessitent un ciblage très précis.

---

## Stack Technique

- **ML** : scikit-learn · XGBoost · LightGBM · SVM
- **Data** : pandas · numpy · matplotlib · seaborn
- **Méthode** : Feature engineering · Hyperparameter tuning (RandomizedSearchCV) · Gestion déséquilibre (`class_weight='balanced'`)
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
- StandardScaler appliqué aux features numériques continues pour les modèles linéaires et SVM

### 3. Benchmark - 8 modèles comparés

| Modèle | AUC-ROC | Recall |
|--------|---------|--------|
| LightGBM | 0.8589 | 71.5% |
| Random Forest | 0.8537 | 43.0% |
| SVM | 0.8531 | 73.7% |
| XGBoost | 0.8263 | 63.9% |
| SGD | 0.7956 | 73.5% |
| Logistic Regression | 0.7946 | 71.3% |
| Naive Bayes | 0.7918 | 50.6% |
| Decision Tree | 0.6600 | 46.4% |

### 4. Sélection & Tuning
- **3 finalistes retenus** : LightGBM, Random Forest, SVM
- Optimisation via **RandomizedSearchCV + StratifiedKFold** (5 folds, scoring = `roc_auc`)
- Sauvegarde du meilleur modèle : `best_model.pkl` + `model_info.pkl`

---

## Lancer le projet

```bash
git clone https://github.com/EddieZIDA/churn-prediction.git
cd churn-prediction
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn jupyter
jupyter notebook notebooks/03_modeling.ipynb
```

---

Perspectives d'Amélioration

- **PR-AUC** : Ajouter cette métrique pour mieux évaluer le modèle, 
  car le ROC-AUC peut être trompeur quand les classes sont déséquilibrées (20/80).

- **SHAP values** : Comprendre pourquoi le modèle prédit un churn 
  pour chaque client individuellement, pas seulement en moyenne.

- **Calibration des probabilités** : S'assurer qu'un score de 70% 
  correspond vraiment à 70% de risque réel.

- **Segmentation par valeur client** : Prioriser les actions de rétention 
  sur les clients les plus rentables plutôt que de traiter tous 
  les churners identifiés de la même façon.
