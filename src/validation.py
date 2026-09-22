"""Validation des inputs utilisateur via Pydantic.

Ce module définit des modèles Pydantic pour valider et transformer
les entrées utilisateur avant de créer les features.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .config import (
    DEFAULT_CREDIT_SCORE,
    MAX_AGE,
    MAX_CREDIT_SCORE,
    MAX_PRODUCTS,
    MAX_TENURE,
    MIN_AGE,
    MIN_CREDIT_SCORE,
    MIN_PRODUCTS,
    MIN_TENURE,
)


class ClientInputModel(BaseModel):
    """Modèle Pydantic pour validation des inputs client.

    Valide :
    - Types de données
    - Plages de valeurs
    - Énumérations (country, gender, options yes/no)

    Attributes
    ----------
    age : int
        Âge du client, entre MIN_AGE et MAX_AGE
    tenure : float
        Ancienneté en années, entre MIN_TENURE et MAX_TENURE
    balance : float
        Solde du compte, ≥ 0
    products_number : int
        Nombre de produits, entre MIN_PRODUCTS et MAX_PRODUCTS
    credit_card : str
        "Oui" ou "Non"
    active_member : str
        "Oui" ou "Non"
    estimated_salary : float
        Salaire estimé, ≥ 0
    country : str
        "France", "Germany" ou "Spain"
    gender : str
        "Female" ou "Male"
    credit_score : int
        Score crédit, entre MIN_CREDIT_SCORE et MAX_CREDIT_SCORE

    Examples
    --------
    >>> input_data = ClientInputModel(
    ...     age=35,
    ...     tenure=3.0,
    ...     balance=10000.0,
    ...     products_number=2,
    ...     credit_card="Oui",
    ...     active_member="Oui",
    ...     estimated_salary=50000.0,
    ...     country="France",
    ...     gender="Male",
    ...     credit_score=650
    ... )
    >>> input_data.age
    35
    >>> input_data.credit_score
    650
    """

    age: int = Field(..., ge=MIN_AGE, le=MAX_AGE, description="Âge du client")
    tenure: float = Field(
        ..., ge=MIN_TENURE, le=MAX_TENURE, description="Ancienneté (années)"
    )
    balance: float = Field(..., ge=0.0, description="Solde du compte (€)")
    products_number: int = Field(
        ...,
        ge=MIN_PRODUCTS,
        le=MAX_PRODUCTS,
        description="Nombre de produits",
    )
    credit_card: Literal["Oui", "Non"] = Field(
        ..., description="Dispose d'une carte de crédit"
    )
    active_member: Literal["Oui", "Non"] = Field(..., description="Membre actif")
    estimated_salary: float = Field(..., ge=0.0, description="Salaire estimé (€)")
    country: Literal["France", "Germany", "Spain"] = Field(
        ..., description="Pays de résidence"
    )
    gender: Literal["Female", "Male"] = Field(..., description="Genre")
    credit_score: int = Field(
        default=DEFAULT_CREDIT_SCORE,
        ge=MIN_CREDIT_SCORE,
        le=MAX_CREDIT_SCORE,
        description="Score de crédit",
    )

    model_config = ConfigDict(populate_by_name=True)

    def to_dict(self) -> dict:
        """Convertit en dictionnaire pour build_features().

        Returns
        -------
        dict
            Dictionnaire avec clés attendues par build_features()
        """
        return self.model_dump()


def validate_client_input(data: dict) -> ClientInputModel:
    """Valide et retourne un modèle client.

    Parameters
    ----------
    data : dict
        Données client brutes (ex. depuis formulaire Streamlit)

    Returns
    -------
    ClientInputModel
        Modèle validé et typé

    Raises
    ------
    ValueError
        Si validation échoue (erreur Pydantic détaillée)

    Examples
    --------
    >>> raw_input = {
    ...     "age": 35,
    ...     "tenure": 3.0,
    ...     "balance": 10000,
    ...     "products_number": 2,
    ...     "credit_card": "Oui",
    ...     "active_member": "Oui",
    ...     "estimated_salary": 50000,
    ...     "country": "France",
    ...     "gender": "Male",
    ...     "credit_score": 650
    ... }
    >>> validated = validate_client_input(raw_input)
    >>> validated.age
    35
    """
    try:
        return ClientInputModel(**data)
    except Exception as e:
        # Pydantic va lever ValidationError avec détails
        raise ValueError(f"Erreur validation données client: {e}") from e
