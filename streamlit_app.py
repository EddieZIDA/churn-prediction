"""Application Streamlit pour prédiction de churn bancaire.

Interface utilisateur qui utilise les modules src/ pour :
- Charger le modèle
- Valider les inputs
- Effectuer les prédictions
- Afficher les explications SHAP
"""

import pandas as pd
import streamlit as st

from src import __version__
from src.config import (
    CONTACT_COST_EUR,
    COUNTRIES,
    DEFAULT_CREDIT_SCORE,
    GENDERS,
    RISK_LABELS,
    RISK_THRESHOLDS,
)
from src.decision import retention_decision
from src.exceptions import (
    ModelNotFoundError,
    ModelValidationError,
    PredictionLoggerError,
    SHAPExplainerError,
)
from src.explainer import create_explainer
from src.features import build_features
from src.logger import get_logger
from src.model import (
    get_decision_threshold,
    load_model,
    predict_churn,
    validate_model,
)
from src.prediction_logger import PredictionLogger
from src.validation import validate_client_input

# --- LOGGING SETUP ---
logger = get_logger(__name__)


# --- STREAMLIT CONFIG ---
st.set_page_config(
    page_title="Prédiction de churn bancaire",
    layout="centered",
)


# --- CACHE FUNCTIONS ---
@st.cache_resource(show_spinner="Chargement du modèle en cours...")
def load_and_validate_model():
    """Charge et valide le modèle ML."""
    try:
        model = load_model()
        if not validate_model(model):
            st.error("Validation du modèle échouée")
            st.stop()
        logger.info("Modèle chargé et validé")
        return model
    except ModelNotFoundError as e:
        logger.error(f"Modèle introuvable: {e}")
        st.error(
            "Modèle introuvable. Exécutez notebooks/03_modeling.ipynb "
            "pour le régénérer, ou définissez MLFLOW_MODEL_URI."
        )
        st.stop()
    except ModelValidationError as e:
        logger.error(f"Modèle invalide: {e}")
        st.error(f"Le modèle chargé est invalide: {e}")
        st.stop()


@st.cache_resource(show_spinner="Préparation du journal de prédictions...")
def get_prediction_logger():
    """Initialise le logger de prédictions (traçabilité / drift)."""
    return PredictionLogger()


@st.cache_resource(show_spinner="Préparation de l'explicateur SHAP...")
def load_explainer(_model):
    """Initialise l'explicateur SHAP.

    Le préfixe underscore de ``_model`` indique à Streamlit de ne pas hacher
    cet argument pour construire la clé de cache. Sans lui, l'application
    plante au chargement : un modèle LightGBM entraîné contient des objets
    non sérialisables, et le cache échoue avant même d'appeler la fonction.
    """
    try:
        explainer = create_explainer(_model)
        logger.info("Explicateur SHAP initialisé")
        return explainer
    except SHAPExplainerError as e:
        logger.error(f"Erreur initialisation SHAP: {e}")
        st.warning(
            "Les explications SHAP ne seront pas disponibles " "pour cette session"
        )
        return None


# --- DISPLAY FUNCTIONS ---
def display_prediction_result(proba_churn: float):
    """Affiche le résultat de la prédiction de manière visuelle.

    Parameters
    ----------
    proba_churn : float
        Probabilité de churn [0, 1]
    """
    # Le seuil d'action vient du modèle lui-même (calibré sur le coût métier),
    # pas d'une constante figée : il suit automatiquement un réentraînement.
    seuil_action = get_decision_threshold()

    if proba_churn >= RISK_THRESHOLDS["high"]:
        label, color = RISK_LABELS["high"]
    elif proba_churn >= seuil_action:
        label, color = RISK_LABELS["medium"]
    else:
        label, color = RISK_LABELS["low"]

    # Affichage styled
    st.divider()
    st.subheader("Résultat de la prédiction")
    st.markdown(
        f"""
        <div style='background-color:{color}; padding:20px;
                    border-radius:10px; color:white; text-align:center;'>
            <h2 style='color:white; margin:0;'>{proba_churn:.1%}</h2>
            <p style='font-size:1.2em; margin:0;'>
                <strong>{label}</strong>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Progress bar
    progress_value = min(max(int(proba_churn * 100), 0), 100)
    st.progress(progress_value / 100)

    st.caption(
        f"Seuil de risque de référence : {seuil_action:.0%}, calibré sur le coût "
        "métier moyen. La recommandation ci-dessous affine ce seuil selon la "
        "valeur du client."
    )


def display_retention_advice(proba_churn: float, balance: float):
    """Affiche la recommandation d'action et son gain attendu.

    Un seuil unique sur la probabilité traite tous les clients comme s'ils
    valaient la même chose. On décide ici par valeur attendue : agir tant que
    la perte évitée dépasse le coût du contact.

    Parameters
    ----------
    proba_churn : float
        Probabilité de churn [0, 1]
    balance : float
        Solde du compte, qui détermine la valeur estimée du client
    """
    decision = retention_decision(proba_churn, balance)

    st.subheader("Recommandation d'action")
    col1, col2, col3 = st.columns(3)
    col1.metric("Valeur estimée du client", f"{decision.customer_value:,.0f} €")
    col2.metric("Perte attendue sans action", f"{decision.expected_loss:,.0f} €")
    col3.metric(
        "Gain net de la campagne",
        f"{decision.net_gain:+,.0f} €",
        delta="rentable" if decision.contact else "non rentable",
        delta_color="normal" if decision.contact else "inverse",
    )

    if decision.contact:
        st.success(
            f"**Contacter ce client.** La perte attendue "
            f"({decision.expected_loss:,.0f} €) dépasse le coût d'une campagne "
            f"({CONTACT_COST_EUR:,.0f} €)."
        )
    else:
        st.info(
            f"**Pas d'action.** La perte attendue "
            f"({decision.expected_loss:,.0f} €) ne couvre pas le coût d'une "
            f"campagne ({CONTACT_COST_EUR:,.0f} €), malgré un risque de "
            f"{proba_churn:.0%}."
        )
    st.caption(
        "La valeur client repose sur une hypothèse de travail (part du solde "
        "déposé) à remplacer par le modèle de valeur vie client de la banque."
    )


def display_client_features(X: pd.DataFrame):
    """Affiche les features encodées du client.

    Parameters
    ----------
    X : pd.DataFrame
        Features du client, une ligne par prédiction
    """
    with st.expander("Voir les détails du profil client encodé"):
        st.dataframe(
            X.T.rename(columns={0: "Valeur"}),
            width="stretch",
        )


def display_shap_explanation(explainer, X: pd.DataFrame):
    """Affiche l'explication SHAP pour une prédiction.

    Parameters
    ----------
    explainer : ShapExplainer
        Explicateur SHAP initialisé
    X : pd.DataFrame
        Features du client
    """
    if explainer is None:
        st.warning("Explicateur SHAP non disponible")
        return

    try:
        st.write("#### Top 3 facteurs influençant cette prédiction")

        # Obtenir les features influentes
        top_features = explainer.get_top_features(X, top_n=3)

        # Affichage en colonnes
        cols = st.columns(3)
        for i, (_, row) in enumerate(top_features.iterrows()):
            direction = "Augmente" if row["Impact SHAP"] > 0 else "Diminue"
            color = "normal" if row["Impact SHAP"] > 0 else "inverse"

            with cols[i]:
                st.metric(
                    label=row["Facteur"],
                    value=f"{row['Impact SHAP']:.3f}",
                    delta=direction,
                    delta_color=color,
                )

        # Graphique
        chart_data = top_features.set_index("Facteur")["Impact SHAP"]
        st.bar_chart(chart_data)

    except SHAPExplainerError as e:
        logger.error(f"Erreur calcul SHAP: {e}", exc_info=True)
        st.error("Impossible de calculer les explications SHAP pour ce client")


# --- MAIN APPLICATION ---
def main():
    """Fonction principale de l'application Streamlit."""
    st.title("Prédiction du churn client bancaire")
    st.markdown(
        "Cette application prédit la probabilité de départ d'un client "
        "bancaire et met en avant les principaux facteurs de risque."
    )

    # Charger modèle et explainer
    model = load_and_validate_model()
    explainer = load_explainer(model)

    # Formulaire de saisie
    st.subheader("Profil du client")
    with st.form(key="client_form"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input(
                "Âge",
                min_value=18,
                max_value=100,
                value=35,
                step=1,
            )
            balance = st.number_input(
                "Solde du compte (€)",
                min_value=0.0,
                value=12000.0,
                step=100.0,
                format="%.2f",
            )
            tenure = st.number_input(
                "Ancienneté (années)",
                min_value=0.0,
                max_value=50.0,
                value=3.0,
                step=0.5,
            )
            products_number = st.number_input(
                "Nombre de produits",
                min_value=1,
                max_value=10,
                value=2,
                step=1,
            )
            credit_card = st.selectbox(
                "Carte de crédit",
                ["Oui", "Non"],
                index=1,
            )

        with col2:
            active_member = st.selectbox(
                "Membre actif",
                ["Oui", "Non"],
                index=0,
            )
            estimated_salary = st.number_input(
                "Salaire estimé (€)",
                min_value=0.0,
                value=50000.0,
                step=1000.0,
                format="%.2f",
            )
            country = st.selectbox(
                "Pays",
                COUNTRIES,
                index=0,
            )
            gender = st.selectbox(
                "Genre",
                GENDERS,
                index=0,
            )
            st.write("---")
            credit_score = st.slider(
                "Score de crédit",
                min_value=300,
                max_value=850,
                value=DEFAULT_CREDIT_SCORE,
                step=1,
            )

        submit_button = st.form_submit_button(
            "Prédire le churn",
            width="stretch",
        )

    # Exécution de la prédiction
    if submit_button:
        # Valider les inputs
        try:
            raw_input = {
                "age": int(age),
                "balance": float(balance),
                "tenure": float(tenure),
                "products_number": int(products_number),
                "credit_card": credit_card,
                "active_member": active_member,
                "estimated_salary": float(estimated_salary),
                "country": country,
                "gender": gender,
                "credit_score": int(credit_score),
            }

            validated = validate_client_input(raw_input)
            logger.info(
                f"Input validé: age={validated.age}, "
                f"credit_score={validated.credit_score}"
            )

        except ValueError as e:
            logger.warning(f"Validation échouée: {e}")
            st.error(f"Erreur validation données: {e}")
            st.stop()

        # Construire features
        X = build_features(validated.to_dict())

        # Prédiction
        _, proba = predict_churn(model, X)
        logger.info(f"Prédiction: probabilité churn = {proba:.3f}")

        # Traçabilité : alimente logs/predictions.csv (détection de drift)
        try:
            get_prediction_logger().log_prediction(
                features=X.iloc[0].to_dict(),
                proba_churn=proba,
                model_version=__version__,
            )
        except PredictionLoggerError as e:
            logger.error(f"Échec du log de prédiction: {e}")

        # Afficher résultats
        display_prediction_result(proba)
        display_retention_advice(proba, validated.balance)
        display_client_features(X)
        display_shap_explanation(explainer, X)

        # Footer
        st.divider()
        st.caption(
            "Cette prédiction est basée sur le modèle LightGBM "
            "entraîné sur 10,000 clients bancaires (AUC-ROC: 0.87)."
        )


if __name__ == "__main__":
    main()
