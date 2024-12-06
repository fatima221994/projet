import pytest
from flask import Flask
from api.app import app  # Modifier selon l'emplacement de votre application
import json

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

def test_home_route(client):
    """Tester si la route d'accueil fonctionne."""
    response = client.get('/')
    assert response.status_code == 200
    assert b"API is running!" in response.data

def test_predict_valid_input(client):
    """Tester la route de prédiction avec des données valides."""
    valid_data = {
        "SK_ID_CURR": 100001,
        "NAME_CONTRACT_TYPE": "Cash loans",
        "CODE_GENDER": "M",
        "FLAG_OWN_CAR": "Y",
        "FLAG_OWN_REALTY": "N",
        "CNT_CHILDREN": 0,
        "AMT_INCOME_TOTAL": 200000,
        "AMT_CREDIT": 600000,
        "AMT_ANNUITY": 25000,
        "AMT_GOODS_PRICE": 500000,
        # Ajouter d'autres colonnes obligatoires avec des valeurs par défaut si nécessaires
    }
    response = client.post('/predict', data=json.dumps(valid_data), content_type='application/json')
    assert response.status_code == 200
    response_json = response.get_json()
    assert 'prediction' in response_json
    assert 'probability' in response_json

def test_predict_missing_data(client):
    """Tester la route de prédiction avec des colonnes manquantes."""
    incomplete_data = {
        "SK_ID_CURR": 100002,
        "NAME_CONTRACT_TYPE": "Revolving loans",
        "CODE_GENDER": "F",
        # 'FLAG_OWN_CAR' et autres colonnes manquent
    }
    response = client.post('/predict', data=json.dumps(incomplete_data), content_type='application/json')
    assert response.status_code == 400
    response_json = response.get_json()
    assert 'error' in response_json
    assert 'missing_columns' in response_json

def test_predict_invalid_data_format(client):
    """Tester la route de prédiction avec des données mal formées."""
    invalid_data = "Ceci n'est pas un JSON valide"
    response = client.post('/predict', data=invalid_data, content_type='application/json')
    assert response.status_code == 400
    response_json = response.get_json()
    assert 'error' in response_json

