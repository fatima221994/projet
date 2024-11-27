from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import make_scorer

app = Flask(__name__)

# Charger le modèle pré-entrainé (remplacez par le chemin de votre modèle)
model = joblib.load('/projet/models/xgb_model_with_smote_and_score_metier_etape_par_etape.pkl')  # Assurez-vous que le modèle est déjà sauvegardé en utilisant joblib
# Charger le préprocesseur (si vous en avez un)
preprocessor = joblib.load('/projet/models/preprocessor.pkl')





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
    
    return df


# Fonction de coût métier (10 * FN + FP)
def cost_function(y_true, y_pred_proba, threshold=0.5):
    y_pred_bin = (y_pred_proba >= threshold).astype(int)
    
    # Calculer la matrice de confusion
    cm = confusion_matrix(y_true, y_pred_bin)
    
    if cm.size == 4:  # Vérifier que la matrice de confusion est bien 2x2
        tn, fp, fn, tp = cm.ravel()
        # Coût métier : 10 * FN + FP
        return 10 * fn + fp
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

        print(f"Prediction Probabilities: {y_pred_proba}")  # Debug: afficher les probabilités de prédiction

        # Tester le coût pour chaque seuil
        for threshold in np.arange(0.0, 1.05, 0.05):
            y_pred_bin = (y_pred_proba >= threshold).astype(int)
            cost = cost_function(np.array([0]), y_pred_proba, threshold)
            print(f"Threshold: {threshold:.2f} - Cost: {cost}")
        
        # Utiliser le seuil optimal fourni pour le score métier
        best_threshold = 0.4000
        
        # Calculer la prédiction binaire en fonction du meilleur seuil
        y_pred_bin = (y_pred_proba >= best_threshold).astype(int)

        # Calculer la matrice de confusion et le coût métier
        y_true = np.array([0])  # À ajuster si vous avez les vraies étiquettes dans les données
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
    app.run(host='127.0.0.1', port=5005, debug=True)