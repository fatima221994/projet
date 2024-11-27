import streamlit as st
import pandas as pd
import requests
import numpy as np
import os


# Définir le port à partir de la variable d'environnement de Heroku
port = os.getenv("PORT", "8501")

st.write(f"App is running on port {port}")

# URL de l'API déployée
API_URL = 'http://127.0.0.1:5005/predict'  # Remplacez par l'URL de votre API Flask

# Charger les données de test ou des exemples de clients
df = pd.read_csv('/api/data/application_test.csv')  # Remplacez par le chemin de votre fichier CSV

# Liste des clients
clients = df['SK_ID_CURR'].tolist()

# Interface Streamlit pour sélectionner un client
st.title("Simulation de scoring client")
client_id = st.selectbox("Sélectionnez un client", clients)

# Filtrer les données du client sélectionné
client_data = df[df['SK_ID_CURR'] == client_id].iloc[0]

# Afficher les informations du client
st.write(f"Informations du client {client_id}:")
st.write(client_data)

# Préparer les données du client pour l'API
client_data_dict = client_data.to_dict()

# Nettoyer les valeurs infinies et NaN dans les données
def clean_data(data):
    # Remplacer NaN et valeurs infinies par des valeurs par défaut (0 dans cet exemple)
    for key, value in data.items():
        if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
            data[key] = 0.0  # Remplace NaN ou inf par 0.0
    return data

client_data_dict = clean_data(client_data_dict)

# Afficher les données envoyées à l'API pour debug
st.write(f"Données envoyées à l'API : {client_data_dict}")

# Envoyer les données du client à l'API et récupérer la réponse
if st.button('Faire la prédiction'):
    try:
        response = requests.post(API_URL, json=client_data_dict)
        response_data = response.json()

        # Vérifier si la prédiction a réussi
        if 'error' not in response_data:
            # Récupérer la probabilité, la classe de crédit et le coût
            prediction_probability = response_data.get('probability', None)
            prediction_class = response_data.get('prediction', None)
            best_threshold = response_data.get('best_threshold', None)
            cost = response_data.get('cost', None)

            # Affichage des résultats
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
        st.write(f"Erreur lors de la requête API : {str(e)}")

# Lancer Streamlit en écoutant sur le port correct
if __name__ == "__main__":
    st.run(port=port)