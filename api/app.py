from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from flask_cors import CORS
import os

# Initialisation de l'application Flask
app = Flask(__name__)
CORS(app)

# Charger le modèle et le préprocesseur
model = joblib.load('models/xgb_model_with_smote_and_score_metier_etape_par_etape.pkl')
preprocessor = joblib.load('models/preprocessor.pkl')

# Définir les colonnes requises à partir des features du préprocesseur
required_columns = list(preprocessor.feature_names_in_)

# Route d'accueil
@app.route('/')
def home():
    return "API is running!"

# Fonction de validation et de prétraitement
def validate_and_preprocess_input(data):
    """
    Valide et prétraite les données pour les rendre compatibles avec le modèle.
    """
    if not isinstance(data, dict):
        raise ValueError("Les données envoyées doivent être au format JSON.")
    
    # Vérifier les colonnes manquantes
    missing_columns = [col for col in required_columns if col not in data]
    
    # Ajouter des colonnes manquantes avec des valeurs par défaut
    for col in missing_columns:
        data[col] = None
    
    # Créer un DataFrame avec les colonnes dans l'ordre attendu
    df = pd.DataFrame([data], columns=required_columns)
    
    # Appliquer le préprocesseur
    processed_data = preprocessor.transform(df)
    
    return processed_data, missing_columns

# Route de prédiction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': "Aucune donnée fournie."}), 400
        
        # Valider et prétraiter les données
        processed_data, missing_columns = validate_and_preprocess_input(data)
        
        if missing_columns:
            return jsonify({
                'error': "Certaines colonnes sont manquantes.",
                'missing_columns': missing_columns
            }), 400
        
        # Effectuer la prédiction
        y_pred_proba = model.predict_proba(processed_data)[:, 1]
        best_threshold = 0.4000
        y_pred_bin = (y_pred_proba >= best_threshold).astype(int)
        
        # Construire la réponse
        response = {
            'prediction': int(y_pred_bin[0]),
            'probability': float(y_pred_proba[0])
        }
        
        return jsonify(response)
    
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f"Une erreur s'est produite : {str(e)}"}), 500

# Lancer l'application
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)), debug=True)
