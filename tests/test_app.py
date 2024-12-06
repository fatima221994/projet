import pytest
import json
from api.app import app  # Assurez-vous d'importer votre application Flask ici

@pytest.fixture
def client():
    # Créez un client de test pour Flask
    with app.test_client() as client:
        yield client

def test_predict_valid(client):
    data = {
        'SK_ID_CURR': 123456,
        'NAME_CONTRACT_TYPE': 'Cash loans',
        'CODE_GENDER': 'F',
        'FLAG_OWN_CAR': 'Y',
        'FLAG_OWN_REALTY': 'Y',
        'CNT_CHILDREN': 2,
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
        'EXT_SOURCE_3': 0.5
    }

    response = client.post('/predict', json=data)

    print(response.data)  # Affichez le message d'erreur pour diagnostiquer
    assert response.status_code == 200
    response_json = response.get_json()
    assert 'prediction' in response_json
    assert 'probability' in response_json

