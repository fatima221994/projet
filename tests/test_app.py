import unittest
from api.app import app  # Modifier l'importation pour refléter le bon chemin

class TestApp(unittest.TestCase):
    def test_home(self):
        # Exemple de test unitaire pour la route home
        with app.test_client() as client:
            response = client.get('/')
            self.assertEqual(response.status_code, 200)

if __name__ == '__main__':
    unittest.main()

