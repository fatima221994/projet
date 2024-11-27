import streamlit as st
import pandas as pd
import requests
import numpy as np
import os

API_URL = 'https://project-science-free-014cfbe31914.herokuapp.com/predict'  # Remplacez par l'URL de votre API Flask

import streamlit as st
import requests
import json

# URL de l'API Flask (en supposant que l'API est déployée sur Heroku ou un autre serveur)
api_url = 'https://project-science-free-014cfbe31914.herokuapp.com/predict'  # Remplacez par l'URL de votre API

# Liste des clients simulée (vous devez probablement remplacer ceci par des données réelles)
clients = [
    {'id': 1, 'name': 'Client 1', 'data': {'SK_ID_CURR': 1, 'NAME_CONTRACT_TYPE': 'Cash loans', 'CODE_GENDER': 'M', 'AMT_INCOME_TOTAL': 10000, 'AMT_CREDIT': 50000}},
    {'id': 2, 'name': 'Client 2', 'data': {'SK_ID_CURR': 2, 'NAME_CONTRACT_TYPE': 'Revolving loans', 'CODE_GENDER': 'F', 'AMT_INCOME_TOTAL': 12000, 'AMT_CREDIT': 80000}},
    # Ajoutez plus de clients ici...
]

# Fonction pour appeler l'API Flask et obtenir les résultats
def get_prediction(client_data):
    try:
        response = requests.post(api_url, json=client_data)  # Envoie des données sous forme JSON
        response.raise_for_status()  # Vérifie si la requête a réussi
        return response.json()  # Retourne les résultats sous forme de JSON
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur lors de la connexion à l'API: {e}")
        return None

# Interface utilisateur Streamlit
st.title('Prédiction de client - API Machine Learning')

# Affichage d'une liste déroulante pour choisir un client
client_names = [client['name'] for client in clients]
selected_client_name = st.selectbox('Sélectionnez un client:', client_names)

# Récupérer les données du client sélectionné
selected_client = next(client for client in clients if client['name'] == selected_client_name)
client_data = selected_client['data']

# Afficher les données du client pour confirmation
st.write("Données du client sélectionné:", client_data)

# Faire la prédiction lorsque l'utilisateur clique sur le bouton
if st.button('Obtenir la prédiction'):
    result = get_prediction(client_data)

    if result:
        # Affichage des résultats de la prédiction
        st.subheader('Résultats de la prédiction')
        st.write(f"Prédiction (0 ou 1): {result['prediction']}")
        st.write(f"Probabilité: {result['probability']:.4f}")
        st.write(f"Seuil optimal: {result['best_threshold']}")
        st.write(f"Coût métier: {result['cost']}")
        
        # Affichage des détails du coût
        st.write("Détails du coût métier:")
        st.write(f"TN: {result['cost_details']['TN']}")
        st.write(f"FP: {result['cost_details']['FP']}")
        st.write(f"FN: {result['cost_details']['FN']}")
        st.write(f"TP: {result['cost_details']['TP']}")
        
        # Affichage des métriques
        st.write("Métriques de performance du modèle:")
        st.write(result['metrics'])
