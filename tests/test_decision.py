"""Tests de la décision de rétention par valeur attendue."""

import pytest
from src.config import CONTACT_COST_EUR, CUSTOMER_VALUE_BASE_EUR
from src.decision import estimate_customer_value, retention_decision


class TestEstimateCustomerValue:
    """Valeur estimée d'un client."""

    def test_zero_balance_keeps_base_value(self):
        """Un client sans solde garde la valeur de base, jamais zéro."""
        assert estimate_customer_value(0) == CUSTOMER_VALUE_BASE_EUR

    def test_value_increases_with_balance(self):
        """Un solde plus élevé donne une valeur plus élevée."""
        assert estimate_customer_value(200_000) > estimate_customer_value(10_000)

    def test_negative_balance_is_floored(self):
        """Un solde négatif ne doit pas produire une valeur inférieure à la base."""
        assert estimate_customer_value(-5_000) == CUSTOMER_VALUE_BASE_EUR


class TestRetentionDecision:
    """La règle de décision doit suivre l'économie, pas la seule probabilité."""

    def test_contacts_when_expected_loss_exceeds_cost(self):
        """Perte attendue supérieure au coût du contact : on agit."""
        decision = retention_decision(0.9, balance=100_000)

        assert decision.contact
        assert decision.net_gain > 0

    def test_skips_when_expected_loss_below_cost(self):
        """Perte attendue inférieure au coût : on n'agit pas."""
        decision = retention_decision(0.02, balance=0)

        assert not decision.contact
        assert decision.net_gain < 0

    def test_high_value_client_contacted_despite_low_risk(self):
        """Un client à faible risque mais forte valeur reste rentable à contacter.

        C'est tout l'intérêt de la règle : un seuil unique sur la probabilité
        laisserait passer ce client.
        """
        decision = retention_decision(0.08, balance=150_000)

        assert decision.contact

    def test_low_value_client_skipped_despite_higher_risk(self):
        """Un risque plus élevé sur un client peu rentable ne suffit pas."""
        risque_faible = retention_decision(0.08, balance=150_000)
        risque_eleve = retention_decision(0.16, balance=0)

        assert risque_eleve.expected_loss < risque_faible.expected_loss
        assert not risque_eleve.contact

    def test_net_gain_matches_definition(self):
        """Le gain net est bien la perte évitée moins le coût du contact."""
        decision = retention_decision(0.5, balance=50_000)

        assert decision.expected_loss == pytest.approx(0.5 * decision.customer_value)
        assert decision.net_gain == pytest.approx(
            decision.expected_loss - CONTACT_COST_EUR
        )

    def test_custom_contact_cost_shifts_decision(self):
        """Une campagne plus chère rend l'action moins souvent rentable."""
        assert retention_decision(0.2, 50_000, contact_cost=10).contact
        assert not retention_decision(0.2, 50_000, contact_cost=1_000).contact

    @pytest.mark.parametrize("proba", [-0.1, 1.1, 2.0])
    def test_rejects_probability_out_of_range(self, proba):
        """Une probabilité hors de [0, 1] est une erreur, pas un cas limite."""
        with pytest.raises(ValueError):
            retention_decision(proba, balance=1_000)
