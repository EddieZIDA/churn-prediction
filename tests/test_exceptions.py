"""Tests for custom exceptions module."""

import pytest

from src.exceptions import (
    ChurnPredictionError,
    InvalidInputError,
    ModelNotFoundError,
    ModelValidationError,
    PredictionLoggerError,
    SHAPExplainerError,
)


class TestExceptionHierarchy:
    """Tests for exception class hierarchy."""

    def test_all_exceptions_inherit_from_base(self):
        """Test that all custom exceptions inherit from ChurnPredictionError."""
        exceptions = [
            InvalidInputError,
            ModelNotFoundError,
            ModelValidationError,
            PredictionLoggerError,
            SHAPExplainerError,
        ]

        for exc_class in exceptions:
            assert issubclass(exc_class, ChurnPredictionError)

    def test_base_exception_inherits_from_exception(self):
        """Test that ChurnPredictionError inherits from Exception."""
        assert issubclass(ChurnPredictionError, Exception)

    def test_all_exceptions_are_exceptions(self):
        """Test that all exceptions can be caught as Exception."""
        exceptions = [
            ChurnPredictionError("test"),
            InvalidInputError("test"),
            ModelNotFoundError("test"),
            ModelValidationError("test"),
            PredictionLoggerError("test"),
            SHAPExplainerError("test"),
        ]

        for exc in exceptions:
            assert isinstance(exc, Exception)


class TestExceptionInstantiation:
    """Tests for creating exception instances."""

    def test_churn_prediction_error_message(self):
        """Test ChurnPredictionError with message."""
        msg = "Test error message"
        exc = ChurnPredictionError(msg)

        assert str(exc) == msg

    def test_model_not_found_error_message(self):
        """Test ModelNotFoundError with message."""
        msg = "Model file not found at /path/to/model.pkl"
        exc = ModelNotFoundError(msg)

        assert str(exc) == msg

    def test_invalid_input_error_message(self):
        """Test InvalidInputError with message."""
        msg = "Age must be between 18 and 100"
        exc = InvalidInputError(msg)

        assert str(exc) == msg

    def test_model_validation_error_message(self):
        """Test ModelValidationError with message."""
        msg = "Model output shape is incorrect"
        exc = ModelValidationError(msg)

        assert str(exc) == msg

    def test_shap_explainer_error_message(self):
        """Test SHAPExplainerError with message."""
        msg = "SHAP calculation failed"
        exc = SHAPExplainerError(msg)

        assert str(exc) == msg

    def test_prediction_logger_error_message(self):
        """Test PredictionLoggerError with message."""
        msg = "Failed to write prediction log"
        exc = PredictionLoggerError(msg)

        assert str(exc) == msg


class TestExceptionCatching:
    """Tests for catching exceptions in different scenarios."""

    def test_catch_model_not_found_as_churn_error(self):
        """Test that ModelNotFoundError can be caught as ChurnPredictionError."""
        with pytest.raises(ChurnPredictionError):
            raise ModelNotFoundError("Model not found")

    def test_catch_invalid_input_as_churn_error(self):
        """Test that InvalidInputError can be caught as ChurnPredictionError."""
        with pytest.raises(ChurnPredictionError):
            raise InvalidInputError("Invalid input")

    def test_catch_model_validation_error_specifically(self):
        """Test catching ModelValidationError specifically."""
        with pytest.raises(ModelValidationError):
            raise ModelValidationError("Validation failed")

    def test_catch_shap_error_specifically(self):
        """Test catching SHAPExplainerError specifically."""
        with pytest.raises(SHAPExplainerError):
            raise SHAPExplainerError("SHAP failed")

    def test_multiple_except_clauses(self):
        """Test using multiple except clauses."""

        def raise_model_error():
            raise ModelNotFoundError("Model error")

        def raise_input_error():
            raise InvalidInputError("Input error")

        # Test first error
        with pytest.raises(ChurnPredictionError):
            try:
                raise_model_error()
            except ModelNotFoundError as e:
                assert "Model error" in str(e)
                raise

        # Test second error
        with pytest.raises(ChurnPredictionError):
            try:
                raise_input_error()
            except InvalidInputError as e:
                assert "Input error" in str(e)
                raise


class TestExceptionChainingFromOther:
    """Tests for exception chaining with 'from' clause."""

    def test_exception_chain_preserves_cause(self):
        """Test that exception chaining preserves the original error."""
        original_error = ValueError("Original error")

        try:
            raise original_error
        except ValueError as e:
            with pytest.raises(ModelValidationError):
                raise ModelValidationError("Validation failed") from e

    def test_caught_exception_has_cause(self):
        """Test that chained exceptions have __cause__ set."""
        original_error = ValueError("Original error")

        try:
            try:
                raise original_error
            except ValueError as e:
                raise ModelValidationError("Validation failed") from e
        except ModelValidationError as e:
            assert e.__cause__ is not None
            assert isinstance(e.__cause__, ValueError)
            assert "Original error" in str(e.__cause__)


class TestExceptionUseInCodeFlow:
    """Tests for realistic exception usage patterns."""

    def test_load_model_error_pattern(self):
        """Test typical pattern for load_model errors."""

        def mock_load_model():
            raise ModelNotFoundError(
                "Model not found at /path/to/model.pkl. "
                "Verify the file exists or set MLFLOW_MODEL_URI."
            )

        with pytest.raises(ModelNotFoundError) as exc_info:
            mock_load_model()

        error_msg = str(exc_info.value)
        assert "Model not found" in error_msg
        assert "MLFLOW_MODEL_URI" in error_msg

    def test_validation_error_pattern(self):
        """Test typical pattern for validation errors."""

        def mock_validate_input(age):
            if not (18 <= age <= 100):
                raise InvalidInputError(f"Age must be between 18 and 100, got {age}")

        # Valid age
        try:
            mock_validate_input(45)
        except InvalidInputError:
            pytest.fail("Should not raise for valid age")

        # Invalid age
        with pytest.raises(InvalidInputError) as exc_info:
            mock_validate_input(150)

        error_msg = str(exc_info.value)
        assert "Age must be between" in error_msg
        assert "150" in error_msg

    def test_shap_error_pattern(self):
        """Test typical pattern for SHAP errors."""

        def mock_calculate_shap(model_type):
            if model_type not in ["LightGBM", "XGBoost"]:
                raise SHAPExplainerError(
                    f"SHAP TreeExplainer does not support {model_type}. "
                    f"Supported types: LightGBM, XGBoost"
                )

        # Valid model type
        try:
            mock_calculate_shap("LightGBM")
        except SHAPExplainerError:
            pytest.fail("Should not raise for LightGBM")

        # Invalid model type
        with pytest.raises(SHAPExplainerError) as exc_info:
            mock_calculate_shap("RandomForest")

        error_msg = str(exc_info.value)
        assert "does not support" in error_msg
        assert "RandomForest" in error_msg
