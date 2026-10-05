import streamlit as st
import pandas as pd
import joblib


# =========================
# Configuration
# =========================

st.set_page_config(
    page_title="Segmentation RFM",
    page_icon="👥",
    layout="centered"
)


# =========================
# Charger le modèle
# =========================

model = joblib.load("../model/random_forest_rfm.joblib")


# =========================
# Segments
# =========================

segments = {
    0: "Clients fidèles",
    1: "Clients inactifs",
    2: "Clients occasionnels"
}


# =========================
# Interface
# =========================

st.title("👥 Segmentation des clients RFM")

st.write(
    "Entrez les caractéristiques RFM d'un nouveau client "
    "pour déterminer automatiquement son segment."
)


# =========================
# Inputs RFM
# =========================

st.subheader("Caractéristiques du client")

recency = st.number_input(
    "Récence",
    min_value=0.0,
    value=20.0,
    step=1.0,
    help="Nombre de jours depuis le dernier achat."
)

frequency = st.number_input(
    "Fréquence",
    min_value=0.0,
    value=10.0,
    step=1.0,
    help="Nombre total d'achats effectués."
)

monetary = st.number_input(
    "Montant total dépensé",
    min_value=0.0,
    value=5000.0,
    step=100.0,
    help="Montant total dépensé par le client."
)


# =========================
# Prediction
# =========================

if st.button("🔮 Prédire le segment"):

    # Création du DataFrame
    nouveau_client = pd.DataFrame([{
        "Recency": recency,
        "Frequency": frequency,
        "Monetary": monetary
    }])

    # Prediction
    cluster_predit = model.predict(nouveau_client)[0]

    # Nom du segment
    segment = segments.get(
        cluster_predit,
        "Segment inconnu"
    )

    # =========================
    # Résultat
    # =========================

    st.success("Prédiction effectuée avec succès !")

    st.subheader("🎯 Résultat")

    st.write(
        f"**Cluster prédit :** {cluster_predit}"
    )

    st.write(
        f"**Segment :** {segment}"
    )

    # =========================
    # Caractéristiques RFM
    # =========================

    st.subheader("📊 Caractéristiques RFM")

    st.dataframe(
        nouveau_client,
        use_container_width=True
    )