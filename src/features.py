"""Feature engineering pour la prédiction de churn.

Ce module exporte les fonctions pour construire les features
à partir des inputs utilisateur.
"""

import pandas as pd

from .config import (
    AGE_GROUPS,
    COUNTRIES,
    FEATURE_COLUMNS,
    GENDERS,
    PRODUCTS_HIGH_RISK_MIN,
    TENURE_GROUPS,
)

# Le premier groupe (young / new) est la catégorie de référence : elle est
# omise à l'encodage (drop_first) pour éviter la colinéarité parfaite.
AGE_GROUP_LABELS = list(AGE_GROUPS)[1:]
TENURE_GROUP_LABELS = list(TENURE_GROUPS)[1:]


def get_age_group(age: float) -> dict:
    """Classe l'âge en groupes catégoriels.

    Parameters
    ----------
    age : float
        Âge du client en années.

    Returns
    -------
    dict
        Dictionnaire avec clés age_group_* et valeurs binaires (0, 1).
        Ex: {"age_group_middle": 1, "age_group_senior": 0, ...}

    Examples
    --------
    >>> get_age_group(35)
    {
        "age_group_middle": 1,
        "age_group_senior": 0,
        "age_group_old": 0,
        "age_group_very_old": 0,
    }
    """
    return {
        f"age_group_{label}": int(AGE_GROUPS[label][0] < age <= AGE_GROUPS[label][1])
        for label in AGE_GROUP_LABELS
    }


def get_tenure_group(tenure: float) -> dict:
    """Classe l'ancienneté en groupes catégoriels.

    Parameters
    ----------
    tenure : float
        Ancienneté du client en années.

    Returns
    -------
    dict
        Dictionnaire avec clés tenure_group_* et valeurs binaires (0 ou 1).
        Exemple: {"tenure_group_medium": 1, "tenure_group_loyal": 0}

    Examples
    --------
    >>> get_tenure_group(3.0)
    {'tenure_group_medium': 1, 'tenure_group_loyal': 0}
    """
    return {
        f"tenure_group_{label}": int(
            TENURE_GROUPS[label][0] < tenure <= TENURE_GROUPS[label][1]
        )
        for label in TENURE_GROUP_LABELS
    }


def build_features(answers: dict) -> pd.DataFrame:
    """Construit un DataFrame avec features engineered pour prédiction.

    Transforme les inputs utilisateur en vecteur de features prêt
    pour être passé au modèle LightGBM. Inclut :
    - Encodage des catégories (country, gender)
    - Création de features dérivées (has_balance, active_products)
    - Grouping d'âge et d'ancienneté

    Parameters
    ----------
    answers : dict
        Dictionnaire contenant les clés suivantes :
        - age (float): Âge du client (18-100)
        - balance (float): Solde du compte (≥0)
        - tenure (float): Ancienneté en années (0-50)
        - products_number (int): Nombre de produits (1-10)
        - credit_card (str): "Oui" ou "Non"
        - active_member (str): "Oui" ou "Non"
        - estimated_salary (float): Salaire estimé (≥0)
        - country (str): "France", "Germany" ou "Spain"
        - gender (str): "Female" ou "Male"
        - credit_score (int): Score crédit (300-850)

    Returns
    -------
    pd.DataFrame
        Shape (1, len(FEATURE_COLUMNS)), colonnes dans l'ordre de FEATURE_COLUMNS.
        Prêt pour model.predict_proba(X).

    Raises
    ------
    KeyError
        Si une clé attendue manque dans 'answers'.
    ValueError
        Si une valeur est en dehors des plages acceptées.

    Examples
    --------
    >>> inputs = {
    ...     "age": 35, "balance": 10000, "tenure": 3,
    ...     "products_number": 2, "credit_card": "Oui",
    ...     "active_member": "Oui", "estimated_salary": 50000,
    ...     "country": "France", "gender": "Male", "credit_score": 650
    ... }
    >>> X = build_features(inputs)
    >>> X.shape[1] == len(FEATURE_COLUMNS)
    True
    >>> X["age"].values[0]
    35
    """
    age_group = get_age_group(answers["age"])
    tenure_group = get_tenure_group(answers["tenure"])
    active_member = int(answers["active_member"] == "Oui")

    feature_values = {
        "credit_score": answers["credit_score"],
        "age": answers["age"],
        "tenure": answers["tenure"],
        "balance": answers["balance"],
        "products_number": answers["products_number"],
        "credit_card": int(answers["credit_card"] == "Oui"),
        "active_member": active_member,
        "estimated_salary": answers["estimated_salary"],
        "country_Germany": int(answers["country"] == "Germany"),
        "country_Spain": int(answers["country"] == "Spain"),
        "gender_Male": int(answers["gender"] == "Male"),
        "has_balance": int(answers["balance"] > 0),
        "active_products": active_member * answers["products_number"],
        "products_2": int(answers["products_number"] == 2),
        "products_3_plus": int(answers["products_number"] >= PRODUCTS_HIGH_RISK_MIN),
    }
    feature_values.update(age_group)
    feature_values.update(tenure_group)

    return pd.DataFrame([feature_values], columns=FEATURE_COLUMNS)


def build_features_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Applique le feature engineering au dataset complet (côté entraînement).

    Version vectorisée de :func:`build_features`, appliquée au dataset brut
    plutôt qu'à un seul client. Les deux fonctions partagent les mêmes seuils
    (``AGE_GROUPS``, ``TENURE_GROUPS``, ``PRODUCTS_HIGH_RISK_MIN`` de
    config.py) : c'est ce qui garantit l'absence de train/serve skew, vérifié
    par les tests de parité.

    Parameters
    ----------
    raw : pd.DataFrame
        Dataset brut tel que lu depuis ``RAW_DATA_PATH``. La colonne
        ``customer_id`` est ignorée si présente, ``churn`` est conservée
        telle quelle si présente.

    Returns
    -------
    pd.DataFrame
        Dataset encodé : colonnes originales (hors catégorielles) suivies de
        ``country_*``, ``gender_*``, ``age_group_*``, ``has_balance``,
        ``tenure_group_*``, ``active_products``, ``products_2`` et
        ``products_3_plus``. Hors ``churn``, l'ordre suit ``FEATURE_COLUMNS``.

    Examples
    --------
    >>> df = pd.read_csv(RAW_DATA_PATH)
    >>> encoded = build_features_frame(df)
    >>> set(FEATURE_COLUMNS).issubset(encoded.columns)
    True
    """
    df = raw.drop(columns=["customer_id"], errors="ignore")

    # Modalités figées depuis la config : sans cela, get_dummies déduirait les
    # catégories de l'échantillon et changerait de référence (donc de colonnes)
    # dès qu'un pays ou un genre est absent des données passées.
    df["country"] = pd.Categorical(df["country"], categories=COUNTRIES)
    df["gender"] = pd.Categorical(df["gender"], categories=GENDERS)
    df = pd.get_dummies(df, columns=["country", "gender"], drop_first=True)
    encoded = [c for c in df.columns if c.startswith(("country_", "gender_"))]
    df[encoded] = df[encoded].astype(float)

    for label in AGE_GROUP_LABELS:
        low, high = AGE_GROUPS[label]
        df[f"age_group_{label}"] = (df["age"] > low) & (df["age"] <= high)

    df["has_balance"] = (df["balance"] > 0).astype(int)

    for label in TENURE_GROUP_LABELS:
        low, high = TENURE_GROUPS[label]
        df[f"tenure_group_{label}"] = (df["tenure"] > low) & (df["tenure"] <= high)

    df["active_products"] = df["active_member"] * df["products_number"]

    df["products_2"] = (df["products_number"] == 2).astype(int)
    df["products_3_plus"] = (df["products_number"] >= PRODUCTS_HIGH_RISK_MIN).astype(
        int
    )

    return df
