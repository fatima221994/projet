import pytest
import json

# Test des données manquantes
@pytest.fixture
def client():
    from api.app import app  
    with app.test_client() as client:
        yield client


def test_invalid_json(client):
    # Test si l'API retourne une erreur si le JSON est invalide
    response = client.post('/predict', data="Invalid data")
    assert response.status_code == 400
    response_data = json.loads(response.data)
    assert 'error' in response_data
    assert response_data['error'] == "Le format des données est invalide, JSON attendu."


