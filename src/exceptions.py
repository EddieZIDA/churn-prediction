"""
Custom exceptions for churn prediction project.

Provides domain-specific exceptions for better error handling
and more informative error messages throughout the application.
"""


class ChurnPredictionError(Exception):
    """Base exception for churn prediction project.

    All custom exceptions inherit from this to enable:
    - Specific exception handling in try/except blocks
    - Easier debugging with context-aware messages
    - Cleaner error propagation in user-facing code
    """

    pass


class ModelNotFoundError(ChurnPredictionError):
    """Raised when model file or registry entry cannot be found.

    Typical causes:
    - models/best_model.pkl doesn't exist
    - MLflow model URI is invalid
    - Model not registered in MLflow
    """

    pass


class InvalidInputError(ChurnPredictionError):
    """Raised when input data fails validation or shape checks.

    Typical causes:
    - Client input doesn't match Pydantic schema
    - Features array has wrong shape
    - Numeric values out of valid range
    """

    pass


class ModelValidationError(ChurnPredictionError):
    """Raised when model fails internal validation checks.

    Typical causes:
    - Model doesn't have predict_proba method
    - Model output shape is wrong
    - Predicted probabilities not in [0, 1]
    - Probabilities don't sum to 1
    """

    pass


class SHAPExplainerError(ChurnPredictionError):
    """Raised when SHAP explanation generation fails.

    Typical causes:
    - Model type not compatible with TreeExplainer
    - Features array has wrong shape
    - Background data is invalid
    - SHAP computation timeout or memory error
    """

    pass


class PredictionLoggerError(ChurnPredictionError):
    """Raised when prediction logging fails.

    Typical causes:
    - Log file write permission denied
    - Log directory doesn't exist
    - Disk space exhausted
    - CSV serialization error
    """

    pass
