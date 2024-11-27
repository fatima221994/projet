from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from flask_cors import CORS
import pickle

# Initialisation de l'application Flask
app = Flask(__name__)

# Ajouter CORS à votre application Flask
CORS(app, resources={r"/predict": {"origins": "*"}})

# Charger le modèle et le préprocesseur
with open('models/xgb_model_with_smote_and_score_metier_etape_par_etape.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)    

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
    
    df = pd.DataFrame([data], columns=expected_columns)

    # Exemple d'encodage des variables catégorielles
    df['NAME_CONTRACT_TYPE'] = df['NAME_CONTRACT_TYPE'].astype('category').cat.codes
    df['CODE_GENDER'] = df['CODE_GENDER'].map({'M': 1, 'F': 0})  # Par exemple, 'M' devient 1 et 'F' devient 0
    df['NAME_INCOME_TYPE'] = df['NAME_INCOME_TYPE'].astype('category').cat.codes
    df['NAME_FAMILY_STATUS'] = df['NAME_FAMILY_STATUS'].astype('category').cat.codes

    # Convertir les colonnes numériques en valeurs appropriées
    df['AMT_INCOME_TOTAL'] = pd.to_numeric(df['AMT_INCOME_TOTAL'], errors='coerce')
    df['AMT_CREDIT'] = pd.to_numeric(df['AMT_CREDIT'], errors='coerce')
    df['AMT_ANNUITY'] = pd.to_numeric(df['AMT_ANNUITY'], errors='coerce')

    # Remplacer les NaN par 0 ou une valeur par défaut
    df.fillna(0, inplace=True)
    
    return df

# Fonction de coût métier (10 * FN + FP)
def cost_function(y_true, y_pred_proba, threshold=0.5):
    y_pred_bin = (y_pred_proba >= threshold).astype(int)
    
    cm = confusion_matrix(y_true, y_pred_bin)
    
    if cm.size == 4:  # Vérifier que la matrice de confusion est bien 2x2
        tn, fp, fn, tp = cm.ravel()
        return 10 * fn + fp  # Coût métier
    else:
        print(f"Avertissement: matrice de confusion invalide: {cm}")
        return 1.0  # Coût métier par défaut

# Route de prédiction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    try:
        # Prétraiter les données
        processed_data = preprocess_data(data)
        
        # Effectuer la prédiction des probabilités
        y_pred_proba = model.predict_proba(processed_data)[:, 1]
        
        # Calculer la prédiction binaire en fonction du meilleur seuil
        best_threshold = 0.4000
        y_pred_bin = (y_pred_proba >= best_threshold).astype(int)

        # Calculer la matrice de confusion et le coût métier
        y_true = np.array([0])  # Ajustez cela si vous avez des vraies étiquettes
        cm = confusion_matrix(y_true, y_pred_bin)
        
        if cm.size == 4:  # Vérifier que la matrice de confusion est bien 2x2
            tn, fp, fn, tp = cm.ravel()
            cost = 10 * fn + fp  # Coût métier
        else:
            tn, fp, fn, tp = 0, 0, 0, 0
            cost = 1.0
        
        # Retourner la réponse avec la probabilité de défaut et le coût métier
        return jsonify({
            'probability': float(y_pred_proba[0]),
            'prediction': int(y_pred_bin[0]),
            'cost': cost
        })
    except Exception as e:
        print(f"Erreur lors de la prédiction: {e}")
        return jsonify({'error': 'Erreur lors du traitement des données'}), 500

# Lancer l'application Flask
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
