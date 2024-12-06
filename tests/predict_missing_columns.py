import pytest
import json

# Test des données manquantes
@pytest.fixture
def client():
    from api.app import app 
    with app.test_client() as client:
        yield client

def test_predict_with_missing_columns(client):
    # Exemple de données avec des colonnes manquantes
    data = {
        'SK_ID_CURR': 123456,
        'NAME_CONTRACT_TYPE': 'Cash loans',
        'CODE_GENDER': 'F',
        'AMT_INCOME_TOTAL': 100000,
        'AMT_CREDIT': 500000,
        'AMT_ANNUITY': 5000,
        'AMT_GOODS_PRICE': 200000,
        'NAME_EDUCATION_TYPE': 'Higher education',
        'NAME_FAMILY_STATUS': 'Single / not married',
        'NAME_HOUSING_TYPE': 'House / apartment',
        'REGION_POPULATION_RELATIVE': 0.02,
        'DAYS_BIRTH': -4000,
        'DAYS_EMPLOYED': -2000,
        'EXT_SOURCE_1': 0.5,
        'EXT_SOURCE_2': 0.5,
        'EXT_SOURCE_3': 0.5,
        # Certaines colonnes attendues sont manquantes, par exemple : 'NAME_TYPE_SUITE', 'OWN_CAR_AGE', etc.
    }
    
    # Envoie de la requête POST avec des données manquantes
    response = client.post('/predict', json=data)
    
    # Vérification du statut de la réponse
    assert response.status_code == 200

    # Vérification de la réponse
    response_data = json.loads(response.data)
    assert 'prediction' in response_data
    assert 'probability' in response_data

    # Vérifier que les valeurs sont bien présentes dans la réponse (le modèle peut toujours faire une prédiction même avec des colonnes manquantes)
    assert isinstance(response_data['prediction'], int)
    assert isinstance(response_data['probability'], float)
