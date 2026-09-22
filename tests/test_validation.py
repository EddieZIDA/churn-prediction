"""Tests pour le module validation (Pydantic)."""

import pytest
from pydantic import ValidationError
from src.validation import ClientInputModel, validate_client_input


class TestClientInputModel:
    """Tests pour modèle Pydantic ClientInputModel."""

    @pytest.fixture
    def valid_input(self):
        """Input valide."""
        return {
            "age": 35,
            "tenure": 3.0,
            "balance": 10000.0,
            "products_number": 2,
            "credit_card": "Oui",
            "active_member": "Oui",
            "estimated_salary": 50000.0,
            "country": "France",
            "gender": "Male",
            "credit_score": 650,
        }

    def test_valid_input(self, valid_input):
        """Test input valide."""
        model = ClientInputModel(**valid_input)
        assert model.age == 35
        assert model.credit_score == 650

    def test_age_too_low(self, valid_input):
        """Test âge < 18."""
        valid_input["age"] = 15
        with pytest.raises(ValidationError):
            ClientInputModel(**valid_input)

    def test_age_too_high(self, valid_input):
        """Test âge > 100."""
        valid_input["age"] = 120
        with pytest.raises(ValidationError):
            ClientInputModel(**valid_input)

    def test_credit_score_out_of_range(self, valid_input):
        """Test credit_score hors limites."""
        valid_input["credit_score"] = 250  # < 300
        with pytest.raises(ValidationError):
            ClientInputModel(**valid_input)

    def test_negative_balance(self, valid_input):
        """Test balance négative."""
        valid_input["balance"] = -1000.0
        with pytest.raises(ValidationError):
            ClientInputModel(**valid_input)

    def test_invalid_country(self, valid_input):
        """Test country invalide."""
        valid_input["country"] = "Italy"
        with pytest.raises(ValidationError):
            ClientInputModel(**valid_input)

    def test_invalid_gender(self, valid_input):
        """Test gender invalide."""
        valid_input["gender"] = "Other"
        with pytest.raises(ValidationError):
            ClientInputModel(**valid_input)

    def test_invalid_credit_card(self, valid_input):
        """Test credit_card invalide."""
        valid_input["credit_card"] = "Maybe"
        with pytest.raises(ValidationError):
            ClientInputModel(**valid_input)

    def test_to_dict(self, valid_input):
        """Test conversion to dict."""
        model = ClientInputModel(**valid_input)
        d = model.to_dict()
        assert isinstance(d, dict)
        assert d["age"] == 35
        assert d["credit_score"] == 650

    def test_default_credit_score(self, valid_input):
        """Test credit_score par défaut."""
        del valid_input["credit_score"]
        model = ClientInputModel(**valid_input)
        from src.config import DEFAULT_CREDIT_SCORE

        assert model.credit_score == DEFAULT_CREDIT_SCORE


class TestValidateClientInput:
    """Tests pour la fonction validate_client_input()."""

    @pytest.fixture
    def valid_input(self):
        return {
            "age": 35,
            "tenure": 3.0,
            "balance": 10000.0,
            "products_number": 2,
            "credit_card": "Oui",
            "active_member": "Oui",
            "estimated_salary": 50000.0,
            "country": "France",
            "gender": "Male",
            "credit_score": 650,
        }

    def test_validate_client_input_valid(self, valid_input):
        """Test validation d'input valide."""
        validated = validate_client_input(valid_input)
        assert isinstance(validated, ClientInputModel)
        assert validated.age == 35

    def test_validate_client_input_invalid(self, valid_input):
        """Test validation d'input invalide."""
        valid_input["age"] = 15  # Invalide
        with pytest.raises(ValueError):
            validate_client_input(valid_input)
