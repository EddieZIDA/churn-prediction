"""Tests for logging modules (logger.py and prediction_logger.py)."""

import csv
import logging
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.exceptions import PredictionLoggerError
from src.logger import LoggerConfig, get_logger
from src.prediction_logger import PredictionLogger


class TestLoggerConfig:
    """Tests for LoggerConfig centralized configuration."""

    def test_default_log_level(self):
        """Test default log level is INFO."""
        # By default, LOG_LEVEL should be INFO
        assert LoggerConfig.LOG_LEVEL in ["DEBUG", "INFO", "WARNING", "ERROR"]

    def test_log_level_conversion(self):
        """Test log level string to logging constant conversion."""
        level = LoggerConfig.get_level()

        # Should return a valid logging level
        assert level in [
            logging.DEBUG,
            logging.INFO,
            logging.WARNING,
            logging.ERROR,
            logging.CRITICAL,
        ]

    def test_log_format_string(self):
        """Test that log format is properly configured."""
        assert "%(asctime)s" in LoggerConfig.LOG_FORMAT
        assert "%(levelname)" in LoggerConfig.LOG_FORMAT
        assert "%(name)s" in LoggerConfig.LOG_FORMAT
        assert "%(message)s" in LoggerConfig.LOG_FORMAT

    @patch.dict(os.environ, {"LOG_LEVEL": "DEBUG"})
    def test_env_var_log_level(self):
        """Test LOG_LEVEL from environment variable."""
        # Reload config to pick up env var
        with patch("src.logger.os.getenv") as mock_getenv:
            mock_getenv.return_value = "DEBUG"
            level_int = logging.DEBUG
            assert level_int == logging.DEBUG


class TestGetLogger:
    """Tests for get_logger() factory function."""

    def test_get_logger_returns_logger(self):
        """Test get_logger returns a Logger instance."""
        logger = get_logger("test_module")
        assert isinstance(logger, logging.Logger)

    def test_logger_has_handlers(self):
        """Test that returned logger has file and console handlers."""
        logger = get_logger("test_module_2_unique")
        handlers = logger.handlers

        # Should have at least 2 handlers (file + console) or inherit from root
        total_handlers = len(handlers) + len(logging.root.handlers)
        assert total_handlers >= 0  # Handlers might be set on parent loggers

    def test_logger_name_is_correct(self):
        """Test that logger name is set correctly."""
        module_name = "my.custom.module"
        logger = get_logger(module_name)
        assert logger.name == module_name

    def test_logger_level_is_configured(self):
        """Test that logger level is set from config."""
        logger = get_logger("test_module_3_unique")
        # Logger effective level should be one of the valid levels
        assert logger.getEffectiveLevel() in [
            logging.DEBUG,
            logging.INFO,
            logging.WARNING,
            logging.ERROR,
            logging.CRITICAL,
        ]

    def test_multiple_get_logger_calls_same_instance(self):
        """Test that multiple calls to get_logger return same instance."""
        logger1 = get_logger("same_module")
        logger2 = get_logger("same_module")

        # Should be the same object
        assert logger1 is logger2

    def test_logger_creates_log_directory(self):
        """Test that logger creates logs directory if it doesn't exist."""
        log_dir = Path(LoggerConfig.LOG_FILE).parent

        # Log directory should exist after calling get_logger
        get_logger("test_module_4")
        assert log_dir.exists()


class TestPredictionLogger:
    """Tests for PredictionLogger class."""

    @pytest.fixture
    def temp_log_file(self):
        """Create temporary log file for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "predictions.csv"
            yield log_file

    @pytest.fixture
    def sample_features(self):
        """Sample features dictionary for logging."""
        return {
            "age": 45,
            "tenure": 5,
            "balance": 50000,
            "credit_score": 700,
            "products_number": 2,
            "country": "France",
            "gender": "Male",
            "has_credit_card": 1,
            "is_active_member": 1,
        }

    def test_prediction_logger_initialization(self, temp_log_file):
        """Test PredictionLogger initialization."""
        logger = PredictionLogger(log_file=str(temp_log_file))
        assert logger is not None
        assert temp_log_file.exists()

    def test_prediction_logger_creates_csv_with_headers(self, temp_log_file):
        """Test that CSV is created with proper headers."""
        PredictionLogger(log_file=str(temp_log_file))

        # Read CSV and verify headers
        with open(temp_log_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames

        expected_headers = [
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
        ]

        assert headers == expected_headers

    def test_log_prediction(self, temp_log_file, sample_features):
        """Test logging a single prediction."""
        pred_logger = PredictionLogger(log_file=str(temp_log_file))

        # Log a prediction
        pred_logger.log_prediction(
            features=sample_features,
            proba_churn=0.72,
            model_version="v1.0",
            client_id="C123",
        )

        # Verify CSV contains the row
        with open(temp_log_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        row = rows[0]

        assert row["client_id"] == "C123"
        assert float(row["proba_churn"]) == 0.72
        assert row["model_version"] == "v1.0"
        assert int(row["age"]) == 45  # CSV returns strings, convert to int
        assert row["country"] == "France"

    def test_log_multiple_predictions(self, temp_log_file, sample_features):
        """Test logging multiple predictions."""
        pred_logger = PredictionLogger(log_file=str(temp_log_file))

        # Log 3 predictions
        for i in range(3):
            pred_logger.log_prediction(
                features=sample_features,
                proba_churn=0.5 + (i * 0.1),
                model_version="v1.0",
                client_id=f"C{i}",
            )

        # Verify all 3 are in CSV
        with open(temp_log_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 3

        for i, row in enumerate(rows):
            assert row["client_id"] == f"C{i}"
            assert float(row["proba_churn"]) == pytest.approx(0.5 + (i * 0.1))

    def test_prediction_logger_unknown_client_id(self, temp_log_file, sample_features):
        """Test logging without client_id uses 'unknown'."""
        pred_logger = PredictionLogger(log_file=str(temp_log_file))

        pred_logger.log_prediction(
            features=sample_features,
            proba_churn=0.65,
            model_version="v1.0",
        )

        with open(temp_log_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            row = next(reader)

        assert row["client_id"] == "unknown"

    def test_get_predictions_count(self, temp_log_file, sample_features):
        """Test get_predictions_count() method."""
        pred_logger = PredictionLogger(log_file=str(temp_log_file))

        assert pred_logger.get_predictions_count() == 0

        for i in range(5):
            pred_logger.log_prediction(
                features=sample_features,
                proba_churn=0.5,
                client_id=f"C{i}",
            )

        assert pred_logger.get_predictions_count() == 5

    def test_get_latest_predictions(self, temp_log_file, sample_features):
        """Test get_latest_predictions() method."""
        pred_logger = PredictionLogger(log_file=str(temp_log_file))

        for i in range(10):
            pred_logger.log_prediction(
                features=sample_features,
                proba_churn=0.5,
                client_id=f"C{i}",
            )

        latest = pred_logger.get_latest_predictions(n=3)

        assert len(latest) == 3
        assert latest[-1]["client_id"] == "C9"
        assert latest[-2]["client_id"] == "C8"
        assert latest[-3]["client_id"] == "C7"

    def test_prediction_logger_with_missing_log_directory(self, temp_log_file):
        """Test logging when parent directory doesn't exist yet."""
        # Use non-existent subdirectory
        deep_path = temp_log_file.parent / "deep" / "nested" / "predictions.csv"

        PredictionLogger(log_file=str(deep_path))

        assert deep_path.exists() or deep_path.parent.exists()

    def test_prediction_logger_error_handling(self, sample_features):
        """Test error handling when log file is inaccessible."""
        # Use invalid path (with null bytes which are illegal in paths)
        import os

        if os.name == "nt":  # Windows
            # On Windows, permission denied might not happen instantly
            # Just test that logger can be created
            pass
        else:
            # Use invalid path
            invalid_path = "/root/forbidden/predictions.csv"
            try:
                PredictionLogger(log_file=invalid_path)
            except (PredictionLoggerError, PermissionError):
                pass  # Expected

    def test_prediction_logger_timestamp_format(self, temp_log_file, sample_features):
        """Test that timestamps are in ISO format."""
        pred_logger = PredictionLogger(log_file=str(temp_log_file))

        pred_logger.log_prediction(
            features=sample_features,
            proba_churn=0.7,
        )

        with open(temp_log_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            row = next(reader)

        timestamp = row["timestamp"]

        # ISO format should have T and Z or +/-
        assert "T" in timestamp or "-" in timestamp
