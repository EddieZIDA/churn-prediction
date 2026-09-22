"""
Prediction logging for churn model tracking and monitoring.

Logs every prediction made by the model to:
- Enable drift detection (Phase 5)
- Audit trail for business team
- Performance monitoring over time
- Feature importance analysis

Each prediction is logged with:
- Timestamp
- Client features (anonymized)
- Model prediction (probability)
- Model version
"""

import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from src.config import PROJECT_ROOT
from src.exceptions import PredictionLoggerError
from src.logger import get_logger

# Get logger for this module
logger = get_logger(__name__)


class PredictionLogger:
    """Log churn predictions for monitoring and drift detection.

    Stores predictions in CSV format with following columns:
    - timestamp: When prediction was made
    - client_id: Optional client identifier
    - age, tenure, balance, credit_score, products_number: Features
    - country, gender, has_credit_card, is_active_member: Categorical
    - proba_churn: Model's predicted churn probability
    - model_version: Model version tag

    CSV file: logs/predictions.csv (auto-created)

    Example:
        >>> pred_logger = PredictionLogger()
        >>> pred_logger.log_prediction(
        ...     client_id="C12345",
        ...     features={"age": 45, "tenure": 5, ...},
        ...     proba_churn=0.72,
        ...     model_version="v1.0"
        ... )
    """

    def __init__(self, log_file: Optional[str] = None):
        """Initialize prediction logger.

        Args:
            log_file: Path to CSV file for logging predictions.
                     Defaults to logs/predictions.csv
        """
        self.log_file = Path(log_file or PROJECT_ROOT / "logs" / "predictions.csv")
        self._ensure_log_file_exists()

    def _ensure_log_file_exists(self) -> None:
        """Create log directory and CSV file with headers if needed."""
        try:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

            if not self.log_file.exists():
                with open(self.log_file, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=[
                            "timestamp",
                            "client_id",
                            "age",
                            "tenure",
                            "balance",
                            "credit_score",
                            "products_number",
                            "country",
                            "gender",
                            "has_credit_card",
                            "is_active_member",
                            "proba_churn",
                            "model_version",
                        ],
                    )
                    writer.writeheader()
                logger.info(f"Created prediction log file: {self.log_file}")
        except Exception as e:
            error_msg = f"Failed to create prediction log file: {e}"
            logger.error(error_msg)
            raise PredictionLoggerError(error_msg) from e

    def log_prediction(
        self,
        features: Dict,
        proba_churn: float,
        model_version: str = "unknown",
        client_id: Optional[str] = None,
    ) -> None:
        """Log a single prediction.

        Args:
            features: Dictionary of client features from build_features()
            proba_churn: Model's predicted churn probability (0-1)
            model_version: Model version tag (default: "unknown")
            client_id: Optional client identifier for tracking

        Raises:
            PredictionLoggerError: If logging fails (file write, etc.)

        Example:
            >>> proba = predict_churn(model, X)
            >>> pred_logger.log_prediction(
            ...     features=client_features,
            ...     proba_churn=proba,
            ...     model_version="v1.0.0",
            ...     client_id="client_12345"
            ... )
        """
        try:
            row = {
                "timestamp": datetime.now().isoformat(),
                "client_id": client_id or "unknown",
                "age": features.get("age"),
                "tenure": features.get("tenure"),
                "balance": features.get("balance"),
                "credit_score": features.get("credit_score"),
                "products_number": features.get("products_number"),
                "country": features.get("country"),
                "gender": features.get("gender"),
                "has_credit_card": features.get("has_credit_card"),
                "is_active_member": features.get("is_active_member"),
                "proba_churn": proba_churn,
                "model_version": model_version,
            }

            with open(self.log_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "timestamp",
                        "client_id",
                        "age",
                        "tenure",
                        "balance",
                        "credit_score",
                        "products_number",
                        "country",
                        "gender",
                        "has_credit_card",
                        "is_active_member",
                        "proba_churn",
                        "model_version",
                    ],
                )
                writer.writerow(row)

            logger.debug(f"Prediction logged: {client_id}, proba={proba_churn:.3f}")

        except Exception as e:
            error_msg = f"Failed to log prediction: {e}"
            logger.error(error_msg)
            raise PredictionLoggerError(error_msg) from e

    def get_predictions_count(self) -> int:
        """Get total number of logged predictions.

        Returns:
            Number of prediction records (excluding header)
        """
        try:
            with open(self.log_file, "r", encoding="utf-8") as f:
                return sum(1 for _ in f) - 1  # -1 for header
        except Exception as e:
            logger.error(f"Failed to read prediction count: {e}")
            return 0

    def get_latest_predictions(self, n: int = 10) -> list:
        """Get latest N predictions from log.

        Args:
            n: Number of predictions to retrieve

        Returns:
            List of dictionaries containing prediction data
        """
        try:
            predictions = []
            with open(self.log_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    predictions.append(row)

            # Return last N records
            return predictions[-n:] if predictions else []
        except Exception as e:
            logger.error(f"Failed to read latest predictions: {e}")
            return []
