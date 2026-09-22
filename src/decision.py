"""Décision de rétention : faut-il contacter ce client, et pour quel gain ?

Le modèle produit une probabilité de départ. Cette probabilité ne dit pas
quoi faire : contacter un client à 60 % de risque ne vaut rien s'il ne
rapporte que 30 €, alors qu'un client à 8 % de risque et 5 000 € de valeur
mérite une campagne.

Ce module sépare donc la prédiction de la décision. La règle appliquée est
celle de la valeur attendue : agir lorsque
``probabilité × valeur du client > coût du contact``.
"""

from typing import NamedTuple

from .config import (
    CONTACT_COST_EUR,
    CUSTOMER_VALUE_BALANCE_SHARE,
    CUSTOMER_VALUE_BASE_EUR,
)


class RetentionDecision(NamedTuple):
    """Recommandation chiffrée pour un client.

    Attributes
    ----------
    contact : bool
        True si l'action de rétention est rentable en espérance.
    customer_value : float
        Valeur estimée du client, en euros.
    expected_loss : float
        Perte attendue en l'absence d'action (probabilité × valeur).
    net_gain : float
        Gain net attendu de l'action (perte évitée moins coût du contact).
        Négatif lorsque le contact coûte plus qu'il ne rapporte.
    """

    contact: bool
    customer_value: float
    expected_loss: float
    net_gain: float


def estimate_customer_value(balance: float) -> float:
    """Estime ce que représente un client, en euros.

    Modèle volontairement simple et transparent : une base fixe (produits
    détenus, frais de tenue de compte) plus une part du solde déposé.

    Parameters
    ----------
    balance : float
        Solde du compte, en euros.

    Returns
    -------
    float
        Valeur estimée du client, en euros.

    Notes
    -----
    Paramètres définis dans ``config.py``. Ce sont des hypothèses de travail
    à remplacer par le modèle de valeur vie client de la banque.

    Examples
    --------
    >>> estimate_customer_value(0)
    200.0
    >>> estimate_customer_value(100_000)
    600.0
    """
    return CUSTOMER_VALUE_BASE_EUR + CUSTOMER_VALUE_BALANCE_SHARE * max(balance, 0.0)


def retention_decision(
    proba_churn: float,
    balance: float,
    contact_cost: float = CONTACT_COST_EUR,
) -> RetentionDecision:
    """Décide si une action de rétention est rentable, et chiffre le gain.

    Un seuil unique sur la probabilité traite tous les clients comme s'ils
    valaient la même chose. Cette règle-ci adapte le seuil à chaque client :
    plus il vaut cher, plus une probabilité faible suffit à justifier l'action.

    Parameters
    ----------
    proba_churn : float
        Probabilité de départ prédite, dans [0, 1].
    balance : float
        Solde du compte, en euros.
    contact_cost : float
        Coût d'une action de rétention, en euros.

    Returns
    -------
    RetentionDecision
        Recommandation et montants associés.

    Raises
    ------
    ValueError
        Si la probabilité est hors de [0, 1].

    Examples
    --------
    >>> d = retention_decision(0.08, balance=150_000)
    >>> d.contact  # faible risque, mais client à forte valeur
    True
    >>> retention_decision(0.16, balance=0).contact  # risque plus élevé, peu de valeur
    False
    """
    if not 0 <= proba_churn <= 1:
        raise ValueError(f"Probabilité hors de [0, 1] : {proba_churn}")

    valeur = estimate_customer_value(balance)
    perte_attendue = proba_churn * valeur
    return RetentionDecision(
        contact=perte_attendue > contact_cost,
        customer_value=valeur,
        expected_loss=perte_attendue,
        net_gain=perte_attendue - contact_cost,
    )
