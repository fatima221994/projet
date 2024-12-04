import pytest
import json
from api.app import app  

@pytest.fixture
def client():
    """Fixture pour configurer un client Flask pour les tests."""
    with app.test_client() as client:
        app.testing = True
        yield client

def test_predict(client):
    """Test de la route /predict."""
    # Exemple de données JSON pour la requête POST
    input_data = {
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
        'NAME_TYPE_SUITE': 'Unaccompanied',
        'NAME_INCOME_TYPE': 'Working',
        'NAME_EDUCATION_TYPE': 'Higher education',
        'NAME_FAMILY_STATUS': 'Single / not married',
        'NAME_HOUSING_TYPE': 'House / apartment',
        'REGION_POPULATION_RELATIVE': 0.02,
        'DAYS_BIRTH': -4000,
        'DAYS_EMPLOYED': -2000,
        'DAYS_REGISTRATION': -1000,
        'DAYS_ID_PUBLISH': -500,
        'OWN_CAR_AGE': 5,
        'FLAG_MOBIL': 1,
        'FLAG_EMP_PHONE': 1,
        'FLAG_WORK_PHONE': 1,
        'FLAG_CONT_MOBILE': 1,
        'FLAG_PHONE': 1,
        'FLAG_EMAIL': 1,
        'OCCUPATION_TYPE': 'Laborers',
        'CNT_FAM_MEMBERS': 3,
        'REGION_RATING_CLIENT': 3,
        'REGION_RATING_CLIENT_W_CITY': 2,
        'WEEKDAY_APPR_PROCESS_START': 1,
        'HOUR_APPR_PROCESS_START': 9,
        'REG_REGION_NOT_LIVE_REGION': 1,
        'REG_REGION_NOT_WORK_REGION': 1,
        'LIVE_REGION_NOT_WORK_REGION': 1,
        'REG_CITY_NOT_LIVE_CITY': 1,
        'REG_CITY_NOT_WORK_CITY': 1,
        'LIVE_CITY_NOT_WORK_CITY': 1,
        'ORGANIZATION_TYPE': 'Commercial',
        'EXT_SOURCE_1': 0.5,
        'EXT_SOURCE_2': 0.5,
        'EXT_SOURCE_3': 0.5,
    }

    # requête POST à la route /predict avec des données JSON
    response = client.post('/predict', 
                           data=json.dumps(input_data),
                           content_type='application/json')

    # Vérification que la réponse est OK (code 200)
    assert response.status_code == 200

    # Vérification du contenu de la réponse 
    response_json = response.get_json()

    # Vérification de la présence des clés dans la réponse
    assert 'prediction' in response_json
    assert 'probability' in response_json
