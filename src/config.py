"""Configuration centralisée pour le projet Churn Prediction.

Ce module centralise :
- Chemins d'accès aux ressources
- Constantes métier
- Hyperparamètres réutilisables
- Random state pour reproductibilité
"""

from pathlib import Path

# --- CHEMINS ---
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

# Fichiers principaux
MODEL_PATH = MODELS_DIR / "best_model.pkl"
MODEL_INFO_PATH = MODELS_DIR / "model_info.pkl"
RAW_DATA_PATH = DATA_RAW_DIR / "Bank-Customer-Churn-Prediction.csv"
PROCESSED_DATA_PATH = DATA_PROCESSED_DIR / "churn_dataset_clean.csv"

# --- REPRODUCTIBILITÉ ---
RANDOM_STATE = 42
TEST_SIZE = 0.2
STRATIFIED = True

# --- CONSTANTES MÉTIER ---
DEFAULT_CREDIT_SCORE = 650
MIN_AGE = 18
MAX_AGE = 100
MIN_TENURE = 0.0
MAX_TENURE = 50.0
MIN_PRODUCTS = 1
MAX_PRODUCTS = 10
MIN_CREDIT_SCORE = 300
MAX_CREDIT_SCORE = 850
COUNTRIES = ["France", "Germany", "Spain"]
GENDERS = ["Female", "Male"]
YES_NO_OPTIONS = ["Oui", "Non"]

# --- FEATURE ENGINEERING ---
# Colonnes features après encodage/engineering
FEATURE_COLUMNS = [
    "credit_score",
    "age",
    "tenure",
    "balance",
    "products_number",
    "credit_card",
    "active_member",
    "estimated_salary",
    "country_Germany",
    "country_Spain",
    "gender_Male",
    "age_group_middle",
    "age_group_senior",
    "age_group_old",
    "age_group_very_old",
    "has_balance",
    "tenure_group_medium",
    "tenure_group_loyal",
    "active_products",
    "products_2",
    "products_3_plus",
]

# Le churn suit une courbe en U selon le nombre de produits détenus :
# 1 produit → 27,7 %, 2 → 7,6 %, 3 → 82,7 %, 4 → 100 %. Encodée en numérique
# brut, cette forme est inexprimable pour un modèle linéaire et coûte des
# découpages à un modèle à base d'arbres. Les deux modalités ci-dessous la
# rendent explicite, avec « 1 produit » comme référence (le cas le plus
# fréquent). Les modalités 3 et 4 sont regroupées : elles se comportent de la
# même façon et 4 produits ne concerne que 60 clients.
PRODUCTS_HIGH_RISK_MIN = 3

# Définition des groupes d'âge
AGE_GROUPS = {
    "young": (0, 30),
    "middle": (30, 40),
    "senior": (40, 50),
    "old": (50, 60),
    "very_old": (60, 150),
}

# Définition des groupes de tenure
TENURE_GROUPS = {
    "new": (0, 2),
    "medium": (2, 5),
    "loyal": (5, 100),
}

# --- THRESHOLDS PRÉDICTION ---
# Bandes d'affichage du risque. Elles n'ont de sens que parce que le modèle
# produit des probabilités calibrées : une prédiction à 50 % correspond bien à
# un client sur deux qui part. Le seuil bas est le seuil de décision optimisé
# sur le coût métier — il est lu depuis models/model_info.pkl à l'exécution,
# la valeur ci-dessous ne servant que de repli si le fichier est absent.
RISK_THRESHOLDS = {
    "high": 0.5,
    "medium": 0.1,
    "low": 0.0,
}

RISK_LABELS = {
    "high": ("Risque élevé", "#cc0000"),
    "medium": ("Risque modéré", "#ff7f0e"),
    "low": ("Risque faible", "#2ca02c"),
}

# --- ÉCONOMIE DE LA RÉTENTION ---
# Une action de rétention n'est rentable que si le gain attendu dépasse son
# coût. Le gain attendu vaut « probabilité de départ × valeur du client », ce
# qui suppose de savoir ce que vaut un client.
#
# ATTENTION : les deux paramètres de valeur ci-dessous sont des HYPOTHÈSES de
# travail, pas des mesures. Ils produisent une valeur moyenne d'environ 500 €
# sur ce jeu de données, cohérente avec le coût de churn utilisé pour calibrer
# le seuil. Ils doivent être remplacés par le modèle de valeur vie client réel
# de la banque avant tout usage en production.
CONTACT_COST_EUR = 50.0
CUSTOMER_VALUE_BASE_EUR = 200.0
CUSTOMER_VALUE_BALANCE_SHARE = 0.004

# --- LOGGING ---
LOG_LEVEL = "INFO"
LOG_FILE = LOGS_DIR / "churn_prediction.log"
