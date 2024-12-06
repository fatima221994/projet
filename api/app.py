from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from flask_cors import CORS
import pickle
from sklearn.preprocessing import LabelEncoder
import os

# Initialisation de l'application Flask
app = Flask(__name__)

# Ajouter CORS à l'application Flask
CORS(app)

# Charger le modèle et le préprocesseur
model = joblib.load('models/xgb_model_with_smote_and_score_metier_etape_par_etape.pkl')

preprocessor = joblib.load('models/preprocessor.pkl')

# Vérifier si le modèle a la méthode 'predict_proba'
print(f"Type du modèle chargé : {type(model)}")

# Optionnellement, ajouter une vérification plus stricte :
if not hasattr(model, 'predict_proba'):
    raise ValueError("Le modèle chargé n'a pas la méthode 'predict_proba'. Vérifiez le fichier du modèle.")

# Route d'accueil pour tester si l'API fonctionne
@app.route('/')
def home():
    return "API is running!"

# Fonction pour transformer les données entrantes
def preprocess_data(data):
    # Définir les colonnes requises à partir des features du préprocesseur
    expected_columns = list(preprocessor.feature_names_in_)
    
    # Vérifier les colonnes manquantes et ajouter des valeurs par défaut
    for col in expected_columns:
        if col not in data:
            data[col] = None

    # Créer un DataFrame avec les données reçues
    df = pd.DataFrame([data], columns=expected_columns)
    
    # Appliquer l'encodage des labels pour les colonnes catégorielles
    categorical_columns = ['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']
    for col in categorical_columns:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].fillna('unknown')  # Remplir les valeurs manquantes
            df[col] = LabelEncoder().fit_transform(df[col])

    return df

# Fonction de coût métier (10 * FN + FP)
def cost_function(y_true, y_pred_proba, threshold=0.5):
    # Conversion des probabilités en classes binaires
    y_pred_bin = (y_pred_proba >= threshold).astype(int)
    
    # Calculer la matrice de confusion
    cm = confusion_matrix(y_true, y_pred_bin)
    
    if cm.shape == (2, 2):  # Vérifier si la matrice est bien 2x2
        tn, fp, fn, tp = cm.ravel()  # Extraire les éléments de la matrice de confusion
        return 10 * fn + fp  # Coût métier
    else:
        return 15.0  # Valeur par défaut si la matrice n'est pas valide

# Route de prédiction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Vérifier si le contenu est bien du JSON
        if not request.is_json:
            return jsonify({'error': "Le format des données est invalide, JSON attendu."}), 400

        # Récupérer les données envoyées
        data = request.get_json(silent=True)  # Utiliser silent=True pour éviter les exceptions si ce n'est pas du JSON
        
        # Si les données ne sont pas valides (par exemple une chaîne non-JSON)
        if data is None:
            return jsonify({'error': "Le format des données est invalide, JSON attendu."}), 400

        # Vérifier si les données sont vides
        if not data:
            return jsonify({'error': "Aucune donnée fournie."}), 400

        # Prétraiter les données
        processed_data = preprocess_data(data)
        
        # Effectuer la prédiction des probabilités
        y_pred_proba = model.predict_proba(processed_data)[:, 1]

        # Utiliser le seuil optimal pour le score métier
        best_threshold = 0.4000
        
        # Calculer la prédiction binaire
        y_pred_bin = (y_pred_proba >= best_threshold).astype(int)
        
        # Construire la réponse
        response = {
            'prediction': int(y_pred_bin[0]),
            'probability': float(y_pred_proba[0])
        }
        
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': f"Une erreur s'est produite : {str(e)}"}), 500



# Lancer l'application
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)), debug=True)
