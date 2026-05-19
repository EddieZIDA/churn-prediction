import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import streamlit as st

# --- CONSTANTES ---
MODEL_PATH = Path("models/best_model.pkl")
DEFAULT_CREDIT_SCORE = 650

FEATURE_COLUMNS = [
    "credit_score", "age", "tenure", "balance", "products_number",
    "credit_card", "active_member", "estimated_salary",
    "country_Germany", "country_Spain", "gender_Male",
    "age_group_middle", "age_group_senior", "age_group_old", "age_group_very_old",
    "has_balance", "tenure_group_medium", "tenure_group_loyal", "active_products",
]

# --- CHARGEMENT DU MODÈLE (AVEC CACHE) ---
@st.cache_resource(show_spinner="Chargement du modèle en cours...")
def load_model():
    """Charge le modèle depuis le stockage local ou MLflow avec mise en cache."""
    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)

    mlflow_uri = os.environ.get("MLFLOW_MODEL_URI")
    if mlflow_uri:
        import mlflow
        return mlflow.pyfunc.load_model(mlflow_uri)

    try:
        import mlflow
        return mlflow.pyfunc.load_model("models:/LightGBM_final/Production")
    except Exception as e:
        raise FileNotFoundError(
            "Aucun modèle local trouvé et MLflow n'a pas pu charger de modèle. "
            f"Détails : {e}"
        )

@st.cache_resource(show_spinner="Préparation de l'explicateur SHAP...")
def get_shap_explainer(_model):
    """Crée et met en cache l'explicateur SHAP pour éviter de le recalculer."""
    return shap.TreeExplainer(_model)

# --- INGENIERIE DES CARACTÉRISTIQUES ---
def get_age_group(age: float) -> dict:
    return {
        "age_group_middle": int(30 < age <= 40),
        "age_group_senior": int(40 < age <= 50),
        "age_group_old": int(50 < age <= 60),
        "age_group_very_old": int(age > 60),
    }

def get_tenure_group(tenure: float) -> dict:
    return {
        "tenure_group_medium": int(2 < tenure <= 5),
        "tenure_group_loyal": int(tenure > 5),
    }

def build_features(answers: dict) -> pd.DataFrame:
    """Transforme les entrées utilisateur en DataFrame prêt pour la prédiction."""
    age_group = get_age_group(answers["age"])
    tenure_group = get_tenure_group(answers["tenure"])
    active_member = int(answers["active_member"] == "Oui")

    feature_values = {
        "credit_score": DEFAULT_CREDIT_SCORE,
        "age": answers["age"],
        "tenure": answers["tenure"],
        "balance": answers["balance"],
        "products_number": answers["products_number"],
        "credit_card": int(answers["credit_card"] == "Oui"),
        "active_member": active_member,
        "estimated_salary": answers["estimated_salary"],
        "country_Germany": int(answers["country"] == "Germany"),
        "country_Spain": int(answers["country"] == "Spain"),
        "gender_Male": int(answers["gender"] == "Male"),
        "has_balance": int(answers["balance"] > 0),
        "active_products": active_member * answers["products_number"],
    }
    feature_values.update(age_group)
    feature_values.update(tenure_group)

    return pd.DataFrame([feature_values], columns=FEATURE_COLUMNS)

def format_probability(prob: float) -> tuple[str, str]:
    if prob >= 0.7:
        return "Risque élevé", "#cc0000"
    if prob >= 0.4:
        return "Risque modéré", "#ff7f0e"
    return "Risque faible", "#2ca02c"

# --- AFFICHAGE DES RÉSULTATS ---
def display_shap_analysis(model, X: pd.DataFrame):
    """Calcule et affiche l'explication SHAP pour la prédiction."""
    try:
        explainer = get_shap_explainer(model)
        shap_values = explainer.shap_values(X)
        
        # Gestion des différentes structures de sortie SHAP selon le modèle
        if isinstance(shap_values, (list, tuple)):
            shap_values = shap_values[1]
        shap_values = np.asarray(shap_values)
        if shap_values.ndim == 2 and shap_values.shape[0] == 1:
            shap_values = shap_values[0]

        shap_df = pd.DataFrame({
            "Facteur": FEATURE_COLUMNS,
            "Impact SHAP": shap_values,
            "Valeur absolue": np.abs(shap_values)
        }).sort_values(by="Valeur absolue", ascending=False)

        top_risks = shap_df.head(3)

        st.write("#### Top 3 facteurs influençant cette prédiction")
        
        # Affichage plus moderne avec des colonnes
        cols = st.columns(3)
        for i, (_, row) in enumerate(top_risks.iterrows()):
            direction = "🔺 Augmente" if row["Impact SHAP"] > 0 else "🔻 Diminue"
            color = "normal" if row["Impact SHAP"] > 0 else "inverse"
            
            with cols[i]:
                st.metric(
                    label=row["Facteur"], 
                    value=f"{row['Impact SHAP']:.3f}", 
                    delta=direction,
                    delta_color=color
                )

        # Graphique
        st.bar_chart(top_risks.set_index("Facteur")["Impact SHAP"])

    except Exception as exc:
        st.error("Impossible de calculer les explications SHAP pour ce client.")
        st.exception(exc)

# --- APPLICATION PRINCIPALE ---
def main():
    st.set_page_config(
        page_title="Prédiction de churn bancaire",
        page_icon="💳",
        layout="centered",
    )

    st.title("Prédiction du churn client bancaire")
    st.markdown("Cette application prédit la probabilité de churn d’un client bancaire et met en avant les principaux facteurs de risque.")

    # Formulaire de saisie
    with st.form(key="client_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Âge", min_value=18, max_value=100, value=35, step=1)
            balance = st.number_input("Solde du compte", min_value=0.0, value=12000.0, step=100.0, format="%.2f")
            tenure = st.number_input("Ancienneté (années)", min_value=0.0, max_value=50.0, value=3.0, step=0.5)
            products_number = st.number_input("Nombre de produits", min_value=1, max_value=10, value=2, step=1)
            credit_card = st.selectbox("Carte de crédit", ["Oui", "Non"], index=1)
        with col2:
            active_member = st.selectbox("Membre actif", ["Oui", "Non"], index=0)
            estimated_salary = st.number_input("Salaire estimé", min_value=0.0, value=50000.0, step=1000.0, format="%.2f")
            country = st.selectbox("Pays", ["France", "Germany", "Spain"], index=0)
            gender = st.selectbox("Genre", ["Female", "Male"], index=0)
            
            st.write("---")
            st.info("Le score de crédit est fixé à 650 par défaut.")

        submit_button = st.form_submit_button("Prédire le churn", use_container_width=True)

    # Exécution de la prédiction
    if submit_button:
        try:
            model = load_model()
        except FileNotFoundError as e:
            st.error(str(e))
            st.stop()

        inputs = {
            "age": float(age),
            "balance": float(balance),
            "tenure": float(tenure),
            "products_number": int(products_number),
            "credit_card": credit_card,
            "active_member": active_member,
            "estimated_salary": float(estimated_salary),
            "country": country,
            "gender": gender,
        }
        
        X = build_features(inputs)

        # Calcul de la probabilité
        proba = model.predict_proba(X)[:, 1][0]
        label, color = format_probability(proba)

        # Affichage des résultats
        st.divider()
        st.subheader("Résultat de la prédiction")
        st.markdown(
            f"""
            <div style='background-color:{color}; padding:20px; border-radius:10px; color:white; text-align:center;'>
                <h2 style='color:white; margin:0;'>{proba:.1%}</h2>
                <p style='font-size:1.2em; margin:0;'><strong>{label}</strong></p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(min(max(int(proba * 100), 0), 100))
        
        with st.expander("Voir les détails du profil client encodé"):
            st.dataframe(X.T.rename(columns={0: "Valeur"}), use_container_width=True)

        # Explications SHAP
        display_shap_analysis(model, X)

if __name__ == "__main__":
    main()