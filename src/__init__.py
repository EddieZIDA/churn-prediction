"""Churn Prediction ML Package.

Modules principaux:
- config: Configuration centralisée
- features: Feature engineering
- model: Chargement et prédiction modèle
- explainer: Explicabilité SHAP
- validation: Validation inputs Pydantic
- logger: Logging centralisé
- exceptions: Exceptions personnalisées
- prediction_logger: Logging des prédictions
- decision: Décision de rétention (valeur attendue)
"""

from .config import (
    FEATURE_COLUMNS,
    RANDOM_STATE,
    MODEL_PATH,
    RISK_LABELS,
    RISK_THRESHOLDS,
)
from .exceptions import (
    ChurnPredictionError,
    InvalidInputError,
    ModelNotFoundError,
    ModelValidationError,
    PredictionLoggerError,
    SHAPExplainerError,
)
from .decision import (
    RetentionDecision,
    estimate_customer_value,
    retention_decision,
)
from .features import (
    build_features,
    build_features_frame,
    get_age_group,
    get_tenure_group,
)
from .logger import LoggerConfig, get_logger
from .model import load_model, predict_churn, validate_model
from .explainer import create_explainer, ShapExplainer
from .prediction_logger import PredictionLogger
from .validation import (
    ClientInputModel,
    validate_client_input,
)

__all__ = [
    "FEATURE_COLUMNS",
    "RANDOM_STATE",
    "MODEL_PATH",
    "RISK_LABELS",
    "RISK_THRESHOLDS",
    "RetentionDecision",
    "estimate_customer_value",
    "retention_decision",
    "build_features",
    "build_features_frame",
    "get_age_group",
    "get_tenure_group",
    "load_model",
    "validate_model",
    "predict_churn",
    "create_explainer",
    "ShapExplainer",
    "ClientInputModel",
    "validate_client_input",
    "ChurnPredictionError",
    "InvalidInputError",
    "ModelNotFoundError",
    "ModelValidationError",
    "PredictionLoggerError",
    "SHAPExplainerError",
    "LoggerConfig",
    "get_logger",
    "PredictionLogger",
]

__version__ = "1.0.0"
