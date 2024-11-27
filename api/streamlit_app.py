import streamlit as st
import pandas as pd
import requests
import numpy as np

# URL de l'API Flask déployée (remplacez par l'URL correcte de votre API)
API_URL = 'http://127.0.0.1:5005/predict'  # Remplacez par l'URL de votre API Flask

# Charger les données de test (application_test.csv) depuis un chemin local
df = pd.read_csv('/projet/api/data/application_test.csv')  # Remplacez par le chemin correct du fichier

# Liste des clients disponibles dans le fichier CSV
clients = df['SK_ID_CURR'].tolist()

# Interface Streamlit : Sélectionner un client parmi la liste
st.title("Simulation de scoring client")
client_id = st.selectbox("Sélectionnez un client", clients)

# Filtrer les données du client sélectionné
client_data = df[df['SK_ID_CURR'] == client_id].iloc[0]

# Affichage des informations du client
st.write(f"Informations du client {client_id}:")
st.write(client_data)

# Préparer les données du client pour l'API sous forme de dictionnaire
client_data_dict = client_data.to_dict()

# Fonction pour nettoyer les données (remplacer NaN et inf par des valeurs par défaut)
def clean_data(data):
    for key, value in data.items():
        if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
            data[key] = 0.0  # Remplace NaN ou inf par 0.0
    return data

client_data_dict = clean_data(client_data_dict)

# Afficher les données envoyées à l'API pour débogage
st.write(f"Données envoyées à l'API : {client_data_dict}")

# Bouton pour faire la prédiction
if st.button('Faire la prédiction'):
    try:
        # Envoi de la requête POST à l'API Flask avec les données du client
        response = requests.post(API_URL, json=client_data_dict)
        
        # Vérification de la réponse de l'API
        response_data = response.json()
        
        if 'error' not in response_data:
            # Récupérer la probabilité, la classe de crédit, le seuil optimal et le coût métier
            prediction_probability = response_data.get('probability', None)
            prediction_class = response_data.get('prediction', None)
            best_threshold = response_data.get('best_threshold', None)
            cost = response_data.get('cost', None)

            # Affichage des résultats de la prédiction
            if prediction_class is not None:
                st.write(f"Classe de crédit : {'Accordé' if prediction_class == 1 else 'Refusé'}")
            else:
                st.write("Classe de crédit : Non disponible")
                
            if prediction_probability is not None:
                st.write(f"Probabilité de défaut : {prediction_probability:.2f}")
            else:
                st.write("Probabilité de défaut : Non disponible")

            if cost is not None:
                st.write(f"Coût métier : {cost}")
            else:
                st.write("Coût métier : Non disponible")

            if best_threshold is not None:
                st.write(f"Seuil optimal : {best_threshold:.4f}")
            else:
                st.write("Seuil optimal : Non disponible")

        else:
            st.write(f"Erreur : {response_data.get('error', 'Inconnue')}")
    
    except Exception as e:
        # Gestion des erreurs liées à la requête API
        st.write(f"Erreur lors de la requête API : {str(e)}")
