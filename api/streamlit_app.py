import streamlit as st
import pandas as pd
import requests
import numpy as np
import os

# Définir le port à partir de la variable d'environnement de Heroku
port = os.getenv("PORT", "8501")
#st.write(f"App is running on port {port}")

# URL de l'API déployée
API_URL = 'https://project-science-free-014cfbe31914.herokuapp.com/predict'  #  l'URL de l API Flask

#  le répertoire de travail actuel
current_dir = os.getcwd()
#st.write(f"Répertoire actuel : {current_dir}")

#  le chemin absolu
file_path = os.path.join(current_dir,  'data', 'application_test.csv')

if os.path.exists(file_path):
    df = pd.read_csv(file_path)
   # st.write("Fichier chargé avec succès.")
else:
    st.write(f"Le fichier {file_path} est introuvable.")

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
#st.write(f"Données envoyées à l'API : {client_data_dict}")

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

            # Affichage des résultats
            if prediction_class is not None:
                st.write(f"Classe de crédit : {'Accordé' if prediction_class == 0 else 'Refusé'}")
            else:
                st.write("Classe de crédit : Non disponible")
                
            if prediction_probability is not None:
                st.write(f"Probabilité de défaut : {prediction_probability:.2f}")
            else:
                st.write("Probabilité de défaut : Non disponible")


        else:
            st.write(f"Erreur : {response_data.get('error', 'Inconnue')}")
    except Exception as e:
        st.write(f"Erreur lors de la requête API : {str(e)}")
