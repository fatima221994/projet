# tests/test_model.py

import pytest
from api.app import model, preprocess_data
import numpy as np
def test_model_prediction():
    # Données fictives pour tester la prédiction
    input_data = {
        'SK_ID_CURR': 12345,
        'NAME_CONTRACT_TYPE': 'Cash loans',
        'CODE_GENDER': 'M',
        'FLAG_OWN_CAR': 'Y',
        'FLAG_OWN_REALTY': 'Y',
        'CNT_CHILDREN': 0,
        'AMT_INCOME_TOTAL': 120000.0,
        'AMT_CREDIT': 100000.0,
        'AMT_ANNUITY': 5000.0,
        'AMT_GOODS_PRICE': 120000.0,
        # Ajouter les autres colonnes nécessaires...
    }
    
    # Prétraiter les données
    processed_data = preprocess_data(input_data)
    
    # Faire une prédiction
    y_pred_proba = model.predict_proba(processed_data)[:, 1]
    
    # Vérifier que la prédiction est un tableau de probabilité
    assert isinstance(y_pred_proba, np.ndarray)
    assert len(y_pred_proba) == 1  # Une seule prédiction pour une entrée
    assert 0 <= y_pred_proba[0] <= 1  # La probabilité doit être entre 0 et 1

