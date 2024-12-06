import pytest
import json
from api.app import app  # Assurez-vous que le nom de votre fichier API est app.py

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

def test_predict_invalid_data_format(client):
    """Tester la route de prédiction avec des données mal formées (non-JSON)."""
    
    # Envoi d'une chaîne de caractères non-JSON
    invalid_data = "Ceci n'est pas un JSON valide"
    
    # Effectuer une requête POST
    response = client.post('/predict', data=invalid_data, content_type='application/json')
    
    # Vérifier que la réponse est une erreur 400
    assert response.status_code == 400
    
    # Vérifier que le message d'erreur est bien celui attendu
    json_response = json.loads(response.data)
    assert json_response['error'] == "Le format des données est invalide, JSON attendu."

