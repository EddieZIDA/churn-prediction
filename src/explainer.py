"""Explicabilité des prédictions via SHAP.

Ce module fournit des fonctions pour calculer et formater
les explications SHAP pour une prédiction donnée.
"""

from typing import Any, Optional

import numpy as np
import pandas as pd
import shap

from .config import FEATURE_COLUMNS
from .exceptions import SHAPExplainerError
from .logger import get_logger

logger = get_logger(__name__)


class ShapExplainer:
    """Wrapper autour SHAP TreeExplainer avec caching optionnel.

    Attributes
    ----------
    model : object
        Modèle LightGBM
    explainer : shap.TreeExplainer
        Instance SHAP initialisée
    background_data : array-like, optional
        Données de background pour accélérer SHAP
    """

    def __init__(self, model: Any, background_data: Optional[Any] = None):
        """Initialise l'explicateur SHAP.

        Parameters
        ----------
        model : object
            Modèle LightGBM avec .predict() et .predict_proba()
        background_data : array-like, optional
            Ensemble de données pour background (sample de train)
            Si fourni, accélère les calculs SHAP.
            Recommandé: 100-1000 samples aléatoires du train set.
        """
        self.model = model
        logger.info("Initialisation SHAP TreeExplainer...")

        try:
            if background_data is not None:
                logger.info(
                    f"Background data fournie: " f"shape={background_data.shape}"
                )
                self.explainer = shap.TreeExplainer(model, data=background_data)
            else:
                logger.warning(
                    "Pas de background data: les calculs SHAP " "seront plus lents"
                )
                self.explainer = shap.TreeExplainer(model)
        except Exception as e:
            error_msg = f"Erreur initialisation SHAP: {e}"
            logger.error(error_msg)
            raise SHAPExplainerError(error_msg) from e

        self.background_data = background_data
        logger.info("SHAP explainer initialisé")

    def explain(self, X: Any) -> np.ndarray:
        """Calcule SHAP values pour une prédiction.

        Parameters
        ----------
        X : array-like
            Features, shape (1, n_features) ou compatible

        Returns
        -------
        np.ndarray
            SHAP values, shape (n_features,) pour un client unique

        Notes
        -----
        Gère les différentes structures de sortie selon type de modèle
        (classification multiclass vs binary).
        """
        logger.debug(f"Calcul SHAP pour shape {X.shape}")

        try:
            shap_values = self.explainer.shap_values(X)

            # Gestion classification binary vs multiclass
            if isinstance(shap_values, (list, tuple)):
                # Multiclass : prendre classe positive (index 1)
                logger.debug("Format multiclass, classe positive extraite")
                shap_values = shap_values[1]

            shap_values = np.asarray(shap_values)

            # Assurez que shape est (n_features,)
            if shap_values.ndim == 2:
                if shap_values.shape[0] == 1:
                    shap_values = shap_values[0]

            return shap_values

        except Exception as e:
            error_msg = f"Erreur calcul SHAP: {e}"
            logger.error(error_msg, exc_info=True)
            raise SHAPExplainerError(error_msg) from e

    def get_top_features(self, X: Any, top_n: int = 3) -> pd.DataFrame:
        """Retourne les N features les plus influentes pour une prédiction.

        Parameters
        ----------
        X : array-like
            Features, shape (1, n_features)
        top_n : int
            Nombre de features à retourner (défaut: 3)

        Returns
        -------
        pd.DataFrame
            Colonnes: ["Facteur", "Impact SHAP", "Valeur absolue"]
            Triées par Impact SHAP décroissant (en valeur absolue)
        """
        shap_values = self.explain(X)

        df = pd.DataFrame(
            {
                "Facteur": FEATURE_COLUMNS,
                "Impact SHAP": shap_values,
                "Valeur absolue": np.abs(shap_values),
            }
        ).sort_values(by="Valeur absolue", ascending=False)

        return df.head(top_n)


def create_explainer(
    model: Any, background_data: Optional[Any] = None
) -> ShapExplainer:
    """Factory pour créer une instance ShapExplainer.

    Parameters
    ----------
    model : object
        Modèle LightGBM
    background_data : array-like, optional
        Données de background SHAP

    Returns
    -------
    ShapExplainer
        Instance prête à utiliser
    """
    return ShapExplainer(model, background_data=background_data)
