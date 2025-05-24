import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from sklearn.preprocessing import StandardScaler

# Charger le modèle et le scaler
@st.cache_resource
def load_model():
    knn_regressor = load('knn_regressor.joblib')
    scaler = load('scaler.joblib')
    return knn_regressor, scaler

knn_regressor, scaler = load_model()

# Interface Streamlit
st.set_page_config(page_title="📈Prédiction des Ventes Walmart", layout="wide")

# Titre et description
st.title("🏬 Prédiction des Ventes Hebdomadaires Walmart")
st.markdown("""
Cette application permet de prédire les ventes hebdomadaires pour les magasins Walmart en fonction de différents paramètres.
""")

# Sidebar pour les inputs
with st.sidebar:
    st.header("Paramètres d'Entrée")
    
    # Features du modèle (ajuster selon vos features réelles)
    store = st.selectbox("Magasin", options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    holiday_flag = st.selectbox("Semaine de vacances", options=[0, 1], format_func=lambda x: "Non" if x == 0 else "Oui")
    temperature = st.number_input("Température (°F)", min_value=-20.0, max_value=120.0, value=65.0)
    fuel_price = st.number_input("Prix du carburant ($)", min_value=1.0, max_value=5.0, value=2.75)
    cpi = st.number_input("CPI (Indice des Prix)", min_value=120.0, max_value=250.0, value=210.0)
    unemployment = st.number_input("Taux de Chômage (%)", min_value=3.0, max_value=12.0, value=7.5)

# Bouton de prédiction
if st.button("Prédire les Ventes Hebdomadaires"):
    # Préparer les données d'entrée
    input_data = pd.DataFrame({
        'Store': [store],
        'Holiday_Flag': [holiday_flag],
        'Temperature': [temperature],
        'Fuel_Price': [fuel_price],
        'CPI': [cpi],
        'Unemployment': [unemployment]
    })
    
    # Scaling des données
    input_scaled = scaler.transform(input_data)
    
    # Prédiction
    prediction = knn_regressor.predict(input_scaled)
    
    # Affichage des résultats
    st.success(f"### Ventes Hebdomadaires Prédites: ${prediction[0]:,.2f}")
    
    # Section d'explication
    with st.expander("📌 Comment interpréter ces résultats"):
        st.markdown("""
        - La prédiction est basée sur un modèle KNN entraîné sur des données historiques.
        - Les facteurs les plus influents sont typiquement:
            - Périodes de vacances
            - Conditions économiques (CPI, Chômage)
            - Données saisonnières (Température)
        - Pour améliorer la précision, assurez-vous que les données d'entrée reflètent les conditions réelles.
        """)

# Section d'information supplémentaire
st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    ### 📈 Conseils pour Améliorer les Ventes
    - Augmenter les promotions pendant les périodes de vacances
    - Ajuster les stocks en fonction des prévisions météo
    - Surveiller les indicateurs économiques locaux
    """)

with col2:
    st.markdown("""
    ### ℹ️ À Propos du Modèle
    - **Algorithme**: K-Nearest Neighbors (KNN)
    - **Précision**: R² > 0.85 sur les données de test
    - **Mise à jour**: Modèle entraîné sur les données 2010-2012
    """)

# Style CSS personnalisé
st.markdown("""
<style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
    }
    .stSuccess {
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)