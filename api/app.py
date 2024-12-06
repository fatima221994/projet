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

# Ajouter CORS à l' application Flask
CORS(app)

# Charger le modèle et le préprocesseur
model = joblib.load('models/xgb_model_with_smote_and_score_metier_etape_par_etape.pkl')
#with open('models/xgb_model_with_smote_and_score_metier_etape_par_etape.pkl', 'rb') as f:
    #model = pickle.load(f)

# Vérifier si le modèle a la méthode 'predict_proba'
print(f"Type du modèle chargé : {type(model)}")

# Optionnellement, ajouter une vérification plus stricte :
if not hasattr(model, 'predict_proba'):
    raise ValueError("Le modèle chargé n'a pas la méthode 'predict_proba'. Vérifiez le fichier du modèle.")

preprocessor = joblib.load('models/preprocessor.pkl')
#with open('models/preprocessor.pkl', 'rb') as f:
    #preprocessor = pickle.load(f)    

# Route d'accueil pour tester si l'API fonctionne
@app.route('/')
def home():
    return "API is running!"

# Fonction pour transformer les données entrantes
def preprocess_data(data):
    expected_columns = [
        'SK_ID_CURR', 'NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
        'CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
        'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS',
        'NAME_HOUSING_TYPE', 'REGION_POPULATION_RELATIVE', 'DAYS_BIRTH', 'DAYS_EMPLOYED',
        'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'OWN_CAR_AGE', 'FLAG_MOBIL', 'FLAG_EMP_PHONE',
        'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL', 'OCCUPATION_TYPE',
        'CNT_FAM_MEMBERS', 'REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY', 'WEEKDAY_APPR_PROCESS_START',
        'HOUR_APPR_PROCESS_START', 'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION',
        'LIVE_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY',
        'LIVE_CITY_NOT_WORK_CITY', 'ORGANIZATION_TYPE', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
        'APARTMENTS_AVG', 'BASEMENTAREA_AVG', 'YEARS_BEGINEXPLUATATION_AVG', 'YEARS_BUILD_AVG',
        'COMMONAREA_AVG', 'ELEVATORS_AVG', 'ENTRANCES_AVG', 'FLOORSMAX_AVG', 'FLOORSMIN_AVG',
        'LANDAREA_AVG', 'LIVINGAPARTMENTS_AVG', 'LIVINGAREA_AVG', 'NONLIVINGAPARTMENTS_AVG',
        'NONLIVINGAREA_AVG', 'APARTMENTS_MODE', 'BASEMENTAREA_MODE', 'YEARS_BEGINEXPLUATATION_MODE',
        'YEARS_BUILD_MODE', 'COMMONAREA_MODE', 'ELEVATORS_MODE', 'ENTRANCES_MODE', 'FLOORSMAX_MODE',
        'FLOORSMIN_MODE', 'LANDAREA_MODE', 'LIVINGAPARTMENTS_MODE', 'LIVINGAREA_MODE',
        'NONLIVINGAPARTMENTS_MODE', 'NONLIVINGAREA_MODE', 'APARTMENTS_MEDI', 'BASEMENTAREA_MEDI',
        'YEARS_BEGINEXPLUATATION_MEDI', 'YEARS_BUILD_MEDI', 'COMMONAREA_MEDI', 'ELEVATORS_MEDI',
        'ENTRANCES_MEDI', 'FLOORSMAX_MEDI', 'FLOORSMIN_MEDI', 'LANDAREA_MEDI', 'LIVINGAPARTMENTS_MEDI',
        'LIVINGAREA_MEDI', 'NONLIVINGAPARTMENTS_MEDI', 'NONLIVINGAREA_MEDI', 'FONDKAPREMONT_MODE',
        'HOUSETYPE_MODE', 'TOTALAREA_MODE', 'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE',
        'OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE',
        'DEF_60_CNT_SOCIAL_CIRCLE', 'DAYS_LAST_PHONE_CHANGE', 'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3',
        'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8',
        'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12',
        'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16',
        'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20',
        'FLAG_DOCUMENT_21', 'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY',
        'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_QRT',
        'AMT_REQ_CREDIT_BUREAU_YEAR', 'BUREAU_AMT_CREDIT_SUM_SUM', 'BUREAU_AMT_CREDIT_SUM_MEAN',
        'BUREAU_AMT_CREDIT_SUM_MAX', 'BUREAU_AMT_CREDIT_SUM_DEBT_SUM', 'BUREAU_AMT_CREDIT_SUM_DEBT_MEAN',
        'BUREAU_AMT_CREDIT_SUM_DEBT_MAX', 'BUREAU_DAYS_CREDIT_MEAN', 'BUREAU_DAYS_CREDIT_MIN',
        'POS_SK_DPD', 'POS_SK_DPD_DEF', 'POS_CNT_INSTALMENT', 'POS_CNT_INSTALMENT_FUTURE',
        'CREDIT_CARD_AMT_BALANCE', 'CREDIT_CARD_AMT_CREDIT_LIMIT_ACTUAL',
        'CREDIT_CARD_AMT_DRAWINGS_ATM_CURRENT', 'CREDIT_CARD_AMT_PAYMENT_CURRENT', 'PREVIOUS_AMT_APPLICATION',
        'PREVIOUS_AMT_CREDIT', 'PREVIOUS_CNT_PAYMENT', 'INSTALLMENTS_AMT_INSTALMENT', 'INSTALLMENTS_AMT_PAYMENT',
        'INSTALLMENTS_NUM_INSTALMENT_NUMBER', 'income_credit_ratio', 'age', 'years_employed', 'log_AMT_INCOME_TOTAL',
        'annuity_income_ratio', 'log_AMT_CREDIT', 'is_employed', 'credit_income_ratio', 'debt_to_income_ratio'
    ]
    
    # Vérification des colonnes manquantes
    missing_columns = [col for col in expected_columns if col not in data]
    if missing_columns:
        return jsonify({'error': f"Clés manquantes : {', '.join(missing_columns)}"}), 400
    
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

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
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

    except ValueError as ve:
        return jsonify({'error': f"Valeurs invalides : {str(ve)}"}), 400
    except Exception as e:
        return jsonify({'error': f"Une erreur s'est produite : {str(e)}"}), 500
