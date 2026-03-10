🎯 Customer Churn Prediction
Prédiction du risque de départ des clients bancaires par Machine Learning.
Dataset : 10 000 clients | Meilleur modèle : Gradient Boosting | ROI : 630%

📊 Résultats
🏆 Gradient Boosting
├─ ROC-AUC  : 0.87  ⭐
├─ Recall   : 69%   (détecte 7 churns sur 10)
├─ Precision: 61%
└─ Impact   : 145 000€ de gain net | ROI 630%
Top 3 facteurs de churn : Âge (40-60 ans) · Inactivité · Solde = 0€

🔧 Stack Technique
ML : scikit-learn · XGBoost · Gradient Boosting
Data : pandas · numpy · matplotlib · seaborn
Méthode : Feature engineering · Hyperparameter tuning · class_weight (déséquilibre 80/20)

📈 Méthodologie

EDA → 12 variables, aucune valeur manquante, analyse bivariée
Feature Engineering → 3 nouvelles variables créées (has_zero_balance, products_activity, balance_per_product)
Modélisation → 5 algorithmes testés, Gradient Boosting retenu
Optimisation → Tuning hyperparamètres + seuil optimal (0.27)

Modèles comparés : Logistic Regression · Decision Tree · Random Forest · Gradient Boosting 🏆 · XGBoost

💼 Impact Business
MétriqueValeurChurns détectés279 / 407 (69%)Clients sauvés112 (taux rétention 40%)

📌 Compétences
✅ Feature engineering · Gestion déséquilibre · Hyperparameter tuning
✅ Évaluation ML (Recall, ROC-AUC, F1) · Interprétabilité (Feature importance)
✅ Calcul ROI · Recommandations business actionnables

▶️ Quick Start
bashgit clone https://github.com/[username]/customer-churn-prediction.git
pip install pandas numpy scikit-learn xgboost matplotlib seaborn jupyter
jupyter notebook customer_churn_analysis.ipynb
