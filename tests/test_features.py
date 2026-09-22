"""Tests pour le module features (feature engineering)."""

import pandas as pd
import pytest
from src.config import FEATURE_COLUMNS
from src.features import (
    build_features,
    build_features_frame,
    get_age_group,
    get_tenure_group,
)


class TestAgeGroup:
    """Tests pour categorisation d'âge."""

    def test_age_young(self):
        """Âge ≤ 30 -> pas de groupe"""
        result = get_age_group(25)
        assert result["age_group_middle"] == 0
        assert result["age_group_senior"] == 0
        assert result["age_group_old"] == 0
        assert result["age_group_very_old"] == 0

    def test_age_middle(self):
        """Âge 30-40 -> age_group_middle"""
        result = get_age_group(35)
        assert result["age_group_middle"] == 1
        assert result["age_group_senior"] == 0
        assert result["age_group_old"] == 0
        assert result["age_group_very_old"] == 0

    def test_age_senior(self):
        """Âge 40-50 -> age_group_senior"""
        result = get_age_group(45)
        assert result["age_group_middle"] == 0
        assert result["age_group_senior"] == 1
        assert result["age_group_old"] == 0
        assert result["age_group_very_old"] == 0

    def test_age_old(self):
        """Âge 50-60 -> age_group_old"""
        result = get_age_group(55)
        assert result["age_group_old"] == 1
        assert result["age_group_very_old"] == 0

    def test_age_very_old(self):
        """Âge > 60 -> age_group_very_old"""
        result = get_age_group(65)
        assert result["age_group_very_old"] == 1


class TestTenureGroup:
    """Tests pour categorisation d'ancienneté."""

    def test_tenure_new(self):
        """Tenure ≤ 2 -> pas de groupe"""
        result = get_tenure_group(1.5)
        assert result["tenure_group_medium"] == 0
        assert result["tenure_group_loyal"] == 0

    def test_tenure_medium(self):
        """Tenure 2-5 -> tenure_group_medium"""
        result = get_tenure_group(3.5)
        assert result["tenure_group_medium"] == 1
        assert result["tenure_group_loyal"] == 0

    def test_tenure_loyal(self):
        """Tenure > 5 -> tenure_group_loyal"""
        result = get_tenure_group(8.0)
        assert result["tenure_group_medium"] == 0
        assert result["tenure_group_loyal"] == 1


class TestBuildFeatures:
    """Tests pour la construction des features."""

    @pytest.fixture
    def sample_input(self):
        """Input client typique."""
        return {
            "age": 35,
            "balance": 10000.0,
            "tenure": 3.0,
            "products_number": 2,
            "credit_card": "Oui",
            "active_member": "Oui",
            "estimated_salary": 50000.0,
            "country": "France",
            "gender": "Male",
            "credit_score": 650,
        }

    def test_build_features_shape(self, sample_input):
        """La forme suit le contrat de features."""
        df = build_features(sample_input)
        assert df.shape == (1, len(FEATURE_COLUMNS))

    def test_build_features_columns(self, sample_input):
        """Test que toutes les colonnes attendues sont présentes."""
        from src.config import FEATURE_COLUMNS

        df = build_features(sample_input)
        assert list(df.columns) == FEATURE_COLUMNS

    def test_build_features_values(self, sample_input):
        """Test que valeurs sont correctement transformées."""
        df = build_features(sample_input)
        # Input encoding
        assert df["age"].iloc[0] == 35
        assert df["credit_score"].iloc[0] == 650
        assert df["balance"].iloc[0] == 10000.0
        # Binary encoding
        assert df["credit_card"].iloc[0] == 1  # "Oui"
        assert df["active_member"].iloc[0] == 1  # "Oui"
        assert df["gender_Male"].iloc[0] == 1  # "Male"
        # Derived features
        assert df["has_balance"].iloc[0] == 1  # balance > 0
        assert df["active_products"].iloc[0] == 2  # 1 * 2
        # Country encoding
        assert df["country_Germany"].iloc[0] == 0  # "France"
        assert df["country_Spain"].iloc[0] == 0  # "France"

    def test_build_features_no_balance(self, sample_input):
        """Test avec solde = 0."""
        sample_input["balance"] = 0.0
        df = build_features(sample_input)
        assert df["has_balance"].iloc[0] == 0
        # Note: active_products n'est pas affecté par balance
        assert df["active_products"].iloc[0] == 2

    def test_build_features_germany(self, sample_input):
        """Test avec country = Germany."""
        sample_input["country"] = "Germany"
        df = build_features(sample_input)
        assert df["country_Germany"].iloc[0] == 1
        assert df["country_Spain"].iloc[0] == 0

    def test_build_features_spain(self, sample_input):
        """Test avec country = Spain."""
        sample_input["country"] = "Spain"
        df = build_features(sample_input)
        assert df["country_Germany"].iloc[0] == 0
        assert df["country_Spain"].iloc[0] == 1

    def test_build_features_female(self, sample_input):
        """Test avec gender = Female."""
        sample_input["gender"] = "Female"
        df = build_features(sample_input)
        assert df["gender_Male"].iloc[0] == 0

    def test_build_features_inactive(self, sample_input):
        """Test avec membre inactif."""
        sample_input["active_member"] = "Non"
        df = build_features(sample_input)
        assert df["active_member"].iloc[0] == 0
        assert df["active_products"].iloc[0] == 0


# Profils couvrant chaque groupe d'âge, d'ancienneté, pays et genre.
PARITY_PROFILES = [
    (25, 1.0, 0.0, 1, "France", "Female", "Non", "Non"),
    (35, 3.0, 10000.0, 2, "Germany", "Male", "Oui", "Oui"),
    (45, 6.0, 25000.5, 3, "Spain", "Female", "Oui", "Non"),
    (55, 0.0, 0.0, 4, "France", "Male", "Non", "Oui"),
    (65, 10.0, 99999.0, 1, "Germany", "Female", "Oui", "Oui"),
    (30, 2.0, 1.0, 2, "Spain", "Male", "Non", "Non"),
    (60, 5.0, 500.0, 2, "France", "Female", "Oui", "Oui"),
]


class TestTrainServeParity:
    """Le chemin entraînement et le chemin service doivent coïncider.

    build_features (1 client, app Streamlit) et build_features_frame
    (dataset complet, notebooks) sont deux implémentations distinctes.
    Toute divergence entre elles produirait un train/serve skew silencieux.
    """

    @pytest.mark.parametrize(
        "age,tenure,balance,products,country,gender,card,active", PARITY_PROFILES
    )
    def test_frame_matches_single_row(
        self, age, tenure, balance, products, country, gender, card, active
    ):
        """Chaque profil donne le même vecteur par les deux chemins."""
        served = build_features(
            {
                "age": age,
                "tenure": tenure,
                "balance": balance,
                "products_number": products,
                "credit_card": card,
                "active_member": active,
                "estimated_salary": 50000.0,
                "country": country,
                "gender": gender,
                "credit_score": 650,
            }
        )

        raw = pd.DataFrame(
            [
                {
                    "customer_id": 1,
                    "credit_score": 650,
                    "country": country,
                    "gender": gender,
                    "age": age,
                    "tenure": tenure,
                    "balance": balance,
                    "products_number": products,
                    "credit_card": int(card == "Oui"),
                    "active_member": int(active == "Oui"),
                    "estimated_salary": 50000.0,
                }
            ]
        )
        # get_dummies n'invente pas les modalités absentes d'un échantillon
        # d'une seule ligne : on les rétablit à 0 avant comparaison.
        trained = build_features_frame(raw).reindex(
            columns=FEATURE_COLUMNS, fill_value=0
        )

        pd.testing.assert_frame_equal(
            trained.astype(float), served.astype(float), check_dtype=False
        )

    def test_frame_produces_expected_columns(self):
        """Le dataset encodé contient bien toutes les features du modèle."""
        raw = pd.DataFrame(
            [
                {
                    "customer_id": i,
                    "credit_score": 650,
                    "country": country,
                    "gender": gender,
                    "age": 40,
                    "tenure": 4.0,
                    "balance": 100.0,
                    "products_number": 2,
                    "credit_card": 1,
                    "active_member": 1,
                    "estimated_salary": 50000.0,
                    "churn": 0,
                }
                for i, (country, gender) in enumerate(
                    [("France", "Female"), ("Germany", "Male"), ("Spain", "Female")]
                )
            ]
        )
        encoded = build_features_frame(raw)

        assert set(FEATURE_COLUMNS).issubset(encoded.columns)
        assert "customer_id" not in encoded.columns
        assert "churn" in encoded.columns

    def test_frame_keeps_zero_tenure_in_reference_group(self):
        """Une ancienneté de 0 reste dans le groupe de référence 'new'."""
        raw = pd.DataFrame(
            [
                {
                    "credit_score": 650,
                    "country": "France",
                    "gender": "Male",
                    "age": 40,
                    "tenure": 0,
                    "balance": 0.0,
                    "products_number": 1,
                    "credit_card": 0,
                    "active_member": 0,
                    "estimated_salary": 1000.0,
                }
            ]
        )
        encoded = build_features_frame(raw)

        assert not encoded["tenure_group_medium"].iloc[0]
        assert not encoded["tenure_group_loyal"].iloc[0]
