import pytest
from api.app import app  # importer application Flask 

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

