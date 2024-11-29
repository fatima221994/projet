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

# Ajouter CORS à votre application Flask
CORS(app)

# Charger le modèle et le préprocesseur
model = joblib.load('models/xgb_model_with_smote_and_score_metier_etape_par_etape.pkl')
#with open('models/xgb_model_with_smote_and_score_metier_etape_par_etape.pkl', 'rb') as f:
    #model = pickle.load(f)

# Vérifier si le modèle a la méthode 'predict_proba'
print(f"Type du modèle chargé : {type(model)}")

# Optionnellement, vous pouvez ajouter une vérification plus stricte :
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
    
    # Créer un DataFrame avec les données reçues
    df = pd.DataFrame([data], columns=expected_columns)
    
    # Appliquer l'encodage des labels pour les colonnes catégorielles
    categorical_columns = ['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS']  # Liste des colonnes catégorielles
    le = LabelEncoder()

    for col in categorical_columns:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])

    return df


# Fonction de coût métier (10 * FN + FP)

def cost_function(y_true, y_pred_proba, threshold=0.5):
    # Conversion des probabilités en classes binaires
    y_pred_bin = (y_pred_proba >= threshold).astype(int)
    
    # Calculer la matrice de confusion
    cm = confusion_matrix(y_true, y_pred_bin)
    
    print(f"Matrice de confusion : {cm}")  # Pour aider à déboguer
    print(f"Classes réelles: {set(y_true)}")
    print(f"Classes prédites: {set(y_pred_bin)}")

    # Vérifier que la matrice de confusion est bien 2x2
    if cm.shape == (2, 2):  # Vérifier si la matrice est bien 2x2
        tn, fp, fn, tp = cm.ravel()  # Extraire les éléments de la matrice de confusion
        
        # Calculer le coût métier
        return 10 * fn + fp  # Coût : 10 * FN + FP
    else:
        print(f"Avertissement: matrice de confusion invalide: {cm}")
        return 15.0  # Valeur par défaut si la matrice n'est pas valide


        

# Route de prédiction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    try:
        # Prétraiter les données
        processed_data = preprocess_data(data)
        
        # Effectuer la prédiction des probabilités
        y_pred_proba = model.predict_proba(processed_data)[:, 1]

        print(f"Prediction Probabilities: {y_pred_proba}")  # Debug: afficher les probabilités de prédiction

        # Tester le coût pour chaque seuil
        for threshold in np.arange(0.0, 1.05, 0.05):
            y_pred_bin = (y_pred_proba >= threshold).astype(int)
            cost = cost_function(y_true, y_pred_proba, threshold)
            print(f"Threshold: {threshold:.2f} - Cost: {cost}")
        
        # Utiliser le seuil optimal fourni pour le score métier
        best_threshold = 0.4000
        
        # Calculer la prédiction binaire en fonction du meilleur seuil
        y_pred_bin = (y_pred_proba >= best_threshold).astype(int)
        
        # Calculer la matrice de confusion et le coût métier
        #y_true = np.array([0])  # À ajuster si vous avez les vraies étiquettes dans les données
        # Charger les étiquettes réelles y_true
        y_true = pd.read_csv('api/data/y_val.csv')['label'].values
        cm = confusion_matrix(y_true, y_pred_bin)
        if cm.size == 4:  # Vérifier que la matrice de confusion est bien 2x2
            tn, fp, fn, tp = cm.ravel()
            cost = 10 * fn + fp  # Coût métier
        else:
            tn, fp, fn, tp = 0, 0, 0, 0
            cost = 1.0  # Coût par défaut en cas de problème avec la matrice de confusion
        
        # Retourner les résultats sous forme JSON
        response = {
            'prediction': int(y_pred_bin[0]),
            'probability': float(y_pred_proba[0]),
            'best_threshold': float(best_threshold),
            'cost': float(cost),
            'cost_details': {
                'TN': int(tn),
                'FP': int(fp),
                'FN': int(fn),
                'TP': int(tp)
            },
            'metrics': {
                'AUC': 0.7625,
                'Accuracy': 0.7992,
                'F1-Score': 0.3039,
                'Precision': 0.2110,
                'Recall': 0.5430,
                'Best Cost Score': -21746.8
            }
        }
        
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)), debug=True)
