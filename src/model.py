"""Gestion du modèle LightGBM pour la prédiction de churn.

Ce module fournit des fonctions pour charger le modèle,
valider son intégrité et effectuer des prédictions.
"""

import os
import pickle
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np

from .config import (
    FEATURE_COLUMNS,
    MODEL_INFO_PATH,
    MODEL_PATH,
    RISK_THRESHOLDS,
)
from .exceptions import (
    InvalidInputError,
    ModelNotFoundError,
)
from .logger import get_logger

logger = get_logger(__name__)


def load_model(
    model_path: Optional[Path] = None,
    mlflow_uri: Optional[str] = None,
) -> Any:
    """Charge le modèle LightGBM depuis stockage local ou MLflow.

    Essaie les sources dans l'ordre :
    1. model_path local (param ou config)
    2. variable d'env MLFLOW_MODEL_URI
    3. MLflow production registry: models:/LightGBM_final/Production

    Parameters
    ----------
    model_path : Path, optional
        Chemin vers fichier .pkl. Défaut: CONFIG MODEL_PATH
    mlflow_uri : str, optional
        URI MLflow explicite (ex. "models:/LightGBM_final/Production")

    Returns
    -------
    object
        Modèle LightGBM chargé, avec méthode predict_proba()

    Raises
    ------
    ModelNotFoundError
        Si aucune source valide trouvée.
    Exception
        Si pickle.load() échoue ou MLflow non accessible.

    Notes
    -----
    Utilise un cache Streamlit si appelé depuis contexte Streamlit
    (via @st.cache_resource decorator dans streamlit_app.py).

    Examples
    --------
    >>> model = load_model()
    >>> model.predict_proba([[1, 2, 3, ...]])  # doctest: +SKIP
    array([[0.3, 0.7]])
    """
    if model_path is None:
        model_path = MODEL_PATH

    # 1. Essayer modèle local
    if model_path.exists():
        try:
            logger.info(f"Chargement modèle depuis {model_path}")
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            logger.info("Modèle chargé avec succès depuis fichier local")
            return model
        except Exception as e:
            logger.error(
                f"Erreur lors du chargement du fichier pickle {model_path}: {e}"
            )
            raise

    # 2. Essayer variable d'env MLflow
    mlflow_uri = mlflow_uri or os.environ.get("MLFLOW_MODEL_URI")
    if mlflow_uri:
        try:
            logger.info(f"Chargement modèle depuis MLflow: {mlflow_uri}")
            import mlflow

            model = mlflow.pyfunc.load_model(mlflow_uri)
            logger.info("Modèle chargé avec succès depuis MLflow")
            return model
        except Exception as e:
            logger.warning(f"MLflow URI fourni mais échec du chargement: {e}")

    # 3. Essayer production registry MLflow
    try:
        logger.info("Tentative chargement depuis MLflow production registry")
        import mlflow

        model = mlflow.pyfunc.load_model("models:/LightGBM_final/Production")
        logger.info("Modèle chargé depuis MLflow production registry")
        return model
    except Exception as e:
        logger.error(f"Échec MLflow production registry: {e}")

    # Échec complet
    error_msg = (
        f"Impossible de charger le modèle. "
        f"Vérifiez que :\n"
        f"  1. {model_path} existe (fichier local)\n"
        f"  2. MLFLOW_MODEL_URI est défini (variable d'env)\n"
        f"  3. MLflow est accessible et le modèle existe en production"
    )
    logger.error(error_msg)
    raise ModelNotFoundError(error_msg)


def validate_model(model: Any) -> bool:
    """Valide qu'un modèle peut effectuer des prédictions.

    Effectue des checks basiques :
    - Modèle a method predict_proba()
    - Prédiction sur dummy input réussit
    - Sorties dans plages valides [0, 1]
    - Proba sum to 1

    Parameters
    ----------
    model : object
        Modèle à valider (supposé avoir .predict_proba())

    Returns
    -------
    bool
        True si tous les checks passent, False sinon.

    Notes
    -----
    Les erreurs détectées sont loggées en ERROR level.
    """
    try:
        # Check 1: method exists
        if not hasattr(model, "predict_proba"):
            logger.error("Modèle n'a pas de méthode predict_proba()")
            return False

        # Check 2: dummy input
        dummy_input = np.random.randn(1, len(FEATURE_COLUMNS))
        pred = model.predict_proba(dummy_input)

        # Check 3: shape
        if pred.shape != (1, 2):
            logger.error(f"Forme de sortie incorrecte: {pred.shape} != (1, 2)")
            return False

        # Check 4: range [0, 1]
        if not (0 <= pred.min() and pred.max() <= 1):
            logger.error(
                f"Probabilités hors limites [0, 1]: min={pred.min()}, "
                f"max={pred.max()}"
            )
            return False

        # Check 5: sum to 1
        if not np.isclose(pred.sum(axis=1)[0], 1.0, atol=1e-5):
            logger.error(f"Proba ne somme pas à 1: {pred.sum(axis=1)[0]:.6f}")
            return False

        logger.info("Validation modèle réussie")
        return True

    except Exception as e:
        logger.error(f"Erreur validation modèle: {e}", exc_info=True)
        return False


def predict_churn(
    model: Any,
    X: Any,
) -> Tuple[float, float]:
    """Effectue une prédiction de churn sur un client.

    Parameters
    ----------
    model : object
        Modèle LightGBM avec .predict_proba()
    X : array-like
        Features du client, shape (1, n_features) ou (n_features,)

    Returns
    -------
    tuple[float, float]
        (proba_no_churn, proba_churn) où proba_churn ∈ [0, 1]

    Raises
    ------
    ValueError
        Si X n'a pas la bonne shape.
    """
    # Assurez que X est 2D
    if hasattr(X, "shape"):
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
    else:
        X = np.array(X).reshape(1, -1)

    if X.shape[1] != len(FEATURE_COLUMNS):
        error_msg = (
            f"Nombre de features incorrect: {X.shape[1]} != {len(FEATURE_COLUMNS)}"
        )
        logger.error(error_msg)
        raise InvalidInputError(error_msg)

    pred = model.predict_proba(X)
    return float(pred[0, 0]), float(pred[0, 1])


def get_decision_threshold(default: float = RISK_THRESHOLDS["medium"]) -> float:
    """Seuil de décision au-delà duquel contacter le client est rentable.

    Le seuil est calibré sur le coût métier pendant l'entraînement et stocké
    dans ``model_info.pkl``. Le lire à l'exécution évite qu'il diverge du
    modèle servi après un réentraînement.

    Parameters
    ----------
    default : float
        Valeur de repli si le fichier est absent ou ne contient pas le seuil.

    Returns
    -------
    float
        Seuil de décision dans [0, 1].
    """
    try:
        with open(MODEL_INFO_PATH, "rb") as f:
            seuil = pickle.load(f).get("decision_threshold")
    except (OSError, pickle.UnpicklingError) as e:
        logger.warning(f"model_info.pkl illisible ({e}), seuil par défaut utilisé")
        return default

    if not isinstance(seuil, (int, float)) or not 0 < seuil < 1:
        logger.warning(f"Seuil de décision invalide dans model_info: {seuil!r}")
        return default
    return float(seuil)
