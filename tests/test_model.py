"""Tests pour le module model (chargement et prédiction)."""

import pickle
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
from src.config import FEATURE_COLUMNS
from src.model import (
    get_decision_threshold,
    load_model,
    predict_churn,
    validate_model,
)
from src.exceptions import ModelNotFoundError, InvalidInputError


class TestLoadModel:
    """Tests pour load_model()."""

    def test_load_model_file_not_found(self, monkeypatch):
        """ModelNotFoundError quand ni le fichier local ni MLflow ne répondent.

        MLflow est simulé : sans cela, load_model() interrogerait le vrai
        registry, ce qui crée un mlflow.db dans le dépôt et ralentit la suite.
        """
        monkeypatch.delenv("MLFLOW_MODEL_URI", raising=False)
        failing_mlflow = Mock()
        failing_mlflow.pyfunc.load_model.side_effect = RuntimeError("indisponible")

        with patch.dict("sys.modules", {"mlflow": failing_mlflow}):
            with pytest.raises(ModelNotFoundError):
                load_model(model_path=Path("/nonexistent/path/model.pkl"))

    def test_load_model_with_mlflow_uri_env(self, monkeypatch):
        """Repli sur MLFLOW_MODEL_URI quand le fichier local est absent."""
        expected_model = Mock()
        fake_mlflow = Mock()
        fake_mlflow.pyfunc.load_model.return_value = expected_model
        monkeypatch.setenv("MLFLOW_MODEL_URI", "models:/test/1")

        with patch.dict("sys.modules", {"mlflow": fake_mlflow}):
            model = load_model(model_path=Path("/nonexistent/path/model.pkl"))

        assert model is expected_model
        fake_mlflow.pyfunc.load_model.assert_called_once_with("models:/test/1")


class TestValidateModel:
    """Tests pour validate_model()."""

    def test_validate_model_missing_predict_proba(self):
        """Test validation échoue si pas predict_proba()."""
        fake_model = Mock(spec=[])  # Pas de predict_proba
        result = validate_model(fake_model)
        assert result is False

    def test_validate_model_wrong_output_shape(self):
        """Test validation échoue si shape incorrecte."""
        fake_model = Mock()
        fake_model.predict_proba = Mock(return_value=np.array([[0.3, 0.7, 0.0]]))
        result = validate_model(fake_model)
        assert result is False

    def test_validate_model_proba_out_of_range(self):
        """Test validation échoue si proba < 0 ou > 1."""
        fake_model = Mock()
        fake_model.predict_proba = Mock(return_value=np.array([[1.2, 0.0]]))
        result = validate_model(fake_model)
        assert result is False

    def test_validate_model_proba_not_sum_to_1(self):
        """Test validation échoue si proba ne somme pas à 1."""
        fake_model = Mock()
        fake_model.predict_proba = Mock(return_value=np.array([[0.4, 0.4]]))
        result = validate_model(fake_model)
        assert result is False

    def test_validate_model_valid(self):
        """Test validation réussit avec modèle correct."""
        fake_model = Mock()
        fake_model.predict_proba = Mock(return_value=np.array([[0.3, 0.7]]))
        result = validate_model(fake_model)
        assert result is True


class TestPredictChurn:
    """Tests pour predict_churn()."""

    @pytest.fixture
    def mock_model(self):
        """Modèle mock pour tests."""
        model = Mock()
        model.predict_proba = Mock(
            return_value=np.array([[0.3, 0.7]])  # 30% no-churn, 70% churn
        )
        return model

    def test_predict_churn_2d_input(self, mock_model):
        """Test prédiction avec input 2D."""
        X = np.random.randn(1, len(FEATURE_COLUMNS))
        proba_no_churn, proba_churn = predict_churn(mock_model, X)
        assert proba_no_churn == 0.3
        assert proba_churn == 0.7

    def test_predict_churn_1d_input(self, mock_model):
        """Test prédiction avec input 1D (reshapé)."""
        X = np.random.randn(len(FEATURE_COLUMNS))
        proba_no_churn, proba_churn = predict_churn(mock_model, X)
        assert proba_no_churn == 0.3
        assert proba_churn == 0.7

    def test_predict_churn_wrong_n_features(self, mock_model):
        """Test erreur si nombre de features incorrect."""
        X = np.random.randn(1, len(FEATURE_COLUMNS) - 1)  # une feature manquante
        with pytest.raises(InvalidInputError):
            predict_churn(mock_model, X)

    def test_predict_churn_returns_float(self, mock_model):
        """Test que les retours sont des floats."""
        X = np.random.randn(1, len(FEATURE_COLUMNS))
        proba_no_churn, proba_churn = predict_churn(mock_model, X)
        assert isinstance(proba_no_churn, float)
        assert isinstance(proba_churn, float)


class TestDecisionThreshold:
    """Le seuil d'action doit suivre le modèle, jamais diverger de lui."""

    def test_reads_threshold_from_model_info(self, tmp_path, monkeypatch):
        """Le seuil stocké à l'entraînement est bien celui qui est appliqué."""
        chemin = tmp_path / "model_info.pkl"
        chemin.write_bytes(pickle.dumps({"decision_threshold": 0.23}))
        monkeypatch.setattr("src.model.MODEL_INFO_PATH", chemin)

        assert get_decision_threshold() == 0.23

    def test_falls_back_when_file_missing(self, tmp_path, monkeypatch):
        """Sans fichier, on retombe sur la valeur de repli plutôt que crasher."""
        monkeypatch.setattr("src.model.MODEL_INFO_PATH", tmp_path / "absent.pkl")

        assert get_decision_threshold(default=0.4) == 0.4

    @pytest.mark.parametrize("valeur", [0, 1, 1.5, -0.2, "0.3", None])
    def test_rejects_invalid_threshold(self, valeur, tmp_path, monkeypatch):
        """Un seuil hors de ]0, 1[ ou mal typé ne doit pas être appliqué."""
        chemin = tmp_path / "model_info.pkl"
        chemin.write_bytes(pickle.dumps({"decision_threshold": valeur}))
        monkeypatch.setattr("src.model.MODEL_INFO_PATH", chemin)

        assert get_decision_threshold(default=0.4) == 0.4

    def test_real_model_info_has_usable_threshold(self):
        """Le model_info livré expose un seuil exploitable."""
        seuil = get_decision_threshold()

        assert 0 < seuil < 1
