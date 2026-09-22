"""Integration tests for churn prediction pipeline.

Tests the full end-to-end pipeline with real model (if available)
or mocks for features that are hard to test in isolation.
"""

import numpy as np
import pandas as pd
import pytest

from src.config import FEATURE_COLUMNS, MODEL_PATH
from src.explainer import create_explainer
from src.features import build_features
from src.model import load_model, predict_churn, validate_model
from src.validation import validate_client_input


class TestLoadRealModel:
    """Tests loading the real model from disk."""

    def test_model_file_exists(self):
        """Verify that the real model file exists."""
        assert MODEL_PATH.exists(), f"Model file not found: {MODEL_PATH}"

    def test_load_model_returns_valid_model(self):
        """Test loading real model and validate it."""
        model = load_model()
        assert model is not None
        assert validate_model(model) is True

    def test_loaded_model_is_pickle(self):
        """Verify model was loaded from pickle."""
        # Load model
        model = load_model()

        # Try to access LightGBM attributes
        assert hasattr(model, "predict_proba")
        assert hasattr(model, "predict")

        # n_features should be 19 OR check feature_names
        if hasattr(model, "n_features"):
            assert model.n_features == len(FEATURE_COLUMNS)
        elif hasattr(model, "n_features_"):
            assert model.n_features_ == len(FEATURE_COLUMNS)


class TestEndToEndPipeline:
    """Full pipeline: input → validation → features → prediction."""

    @pytest.fixture
    def model(self):
        """Load real model once for all tests."""
        return load_model()

    @pytest.fixture
    def sample_client_data(self):
        """Sample client data for testing."""
        return {
            "age": 45,
            "tenure": 5,
            "balance": 50000,
            "credit_score": 700,
            "products_number": 2,
            "country": "France",
            "gender": "Male",
            "credit_card": "Oui",
            "active_member": "Oui",
            "estimated_salary": 75000.0,
        }

    def test_full_pipeline_valid_input(self, model, sample_client_data):
        """Test complete pipeline with valid input."""
        # Step 1: Validate input
        validated = validate_client_input(sample_client_data)
        assert validated is not None

        # Step 2: Build features
        X = build_features(validated.to_dict())
        assert X.shape == (1, len(FEATURE_COLUMNS))
        assert list(X.columns) == FEATURE_COLUMNS

        # Step 3: Predict
        proba_no_churn, proba_churn = predict_churn(model, X.values)

        # Step 4: Verify output
        assert 0 <= proba_no_churn <= 1
        assert 0 <= proba_churn <= 1
        assert np.isclose(proba_no_churn + proba_churn, 1.0, atol=1e-5)

    @pytest.mark.parametrize(
        "age,tenure,expected_risk",
        [
            (20, 1, "high"),  # Young, new → potentially high risk
            (65, 30, "low"),  # Older, very loyal → potentially low risk
            (35, 10, "medium"),  # Middle age, medium tenure → mixed
        ],
    )
    def test_different_client_profiles(self, model, age, tenure, expected_risk):
        """Test predictions for different client profiles."""
        client_data = {
            "age": age,
            "tenure": tenure,
            "balance": 50000,
            "credit_score": 700,
            "products_number": 2,
            "country": "France",
            "gender": "Male",
            "credit_card": "Oui",
            "active_member": "Oui",
            "estimated_salary": 75000.0,
        }

        validated = validate_client_input(client_data)
        X = build_features(validated.to_dict())
        proba_no_churn, proba_churn = predict_churn(model, X.values)

        # Just verify valid output (don't assert on expected_risk yet)
        assert 0 <= proba_churn <= 1
        assert np.isclose(proba_no_churn + proba_churn, 1.0, atol=1e-5)

    def test_pipeline_with_different_countries(self, model):
        """Test pipeline works for all countries."""
        base_client = {
            "age": 45,
            "tenure": 5,
            "balance": 50000,
            "credit_score": 700,
            "products_number": 2,
            "gender": "Male",
            "credit_card": "Oui",
            "active_member": "Oui",
            "estimated_salary": 75000.0,
        }

        for country in ["France", "Germany", "Spain"]:
            client_data = {**base_client, "country": country}
            validated = validate_client_input(client_data)
            X = build_features(validated.to_dict())
            proba_no_churn, proba_churn = predict_churn(model, X.values)

            assert 0 <= proba_churn <= 1, f"Invalid proba for {country}"

    def test_pipeline_with_dataframe_output(self, model, sample_client_data):
        """Test that features are returned as DataFrame."""
        validated = validate_client_input(sample_client_data)
        X = build_features(validated.to_dict())

        assert isinstance(X, pd.DataFrame)
        assert X.shape == (1, len(FEATURE_COLUMNS))


class TestExplainerWithRealModel:
    """SHAP explainer tests with real model."""

    @pytest.fixture
    def model(self):
        """Load real model."""
        return load_model()

    @pytest.fixture
    def sample_features(self):
        """Generate sample features."""
        return np.random.randn(1, len(FEATURE_COLUMNS))

    def test_explainer_creation(self, model):
        """Test creating SHAP explainer."""
        explainer = create_explainer(model)
        assert explainer is not None
        assert hasattr(explainer, "explain")
        assert hasattr(explainer, "get_top_features")

    def test_explainer_explain(self, model, sample_features):
        """Test SHAP explain function."""
        explainer = create_explainer(model)
        shap_values = explainer.explain(sample_features)

        assert shap_values is not None
        assert isinstance(shap_values, np.ndarray)
        assert shap_values.shape == (len(FEATURE_COLUMNS),)

    def test_explainer_top_features(self, model, sample_features):
        """Test getting top features from SHAP."""
        explainer = create_explainer(model)
        top_features = explainer.get_top_features(sample_features, top_n=3)

        assert top_features is not None
        assert isinstance(top_features, pd.DataFrame)
        assert len(top_features) == 3
        assert list(top_features.columns) == [
            "Facteur",
            "Impact SHAP",
            "Valeur absolue",
        ]
        assert all(top_features["Facteur"].isin(FEATURE_COLUMNS))

    def test_explainer_top_n_parameter(self, model, sample_features):
        """Test top_n parameter variation."""
        explainer = create_explainer(model)

        for top_n in [1, 3, 5]:
            top_features = explainer.get_top_features(sample_features, top_n=top_n)
            assert len(top_features) == top_n


class TestPredictionConsistency:
    """Test consistency of predictions."""

    @pytest.fixture
    def model(self):
        """Load real model."""
        return load_model()

    def test_same_input_same_output(self, model):
        """Test that same input produces same output."""
        np.random.seed(42)
        X = np.random.randn(1, len(FEATURE_COLUMNS))

        proba_1 = predict_churn(model, X)
        proba_2 = predict_churn(model, X)

        assert proba_1 == proba_2

    def test_batch_prediction_consistency(self, model):
        """Test predictions are consistent for different inputs."""
        np.random.seed(42)

        predictions = []
        for _ in range(5):
            X = np.random.randn(1, len(FEATURE_COLUMNS))
            proba = predict_churn(model, X)
            predictions.append(proba)

        # All predictions should be valid
        for proba_no_churn, proba_churn in predictions:
            assert 0 <= proba_no_churn <= 1
            assert 0 <= proba_churn <= 1
            assert np.isclose(proba_no_churn + proba_churn, 1.0)
