# API de Prédiction (Moteur d'Inférence)

Ce projet contient une API permettant de réaliser des prédictions à l'aide d'un modèle de machine learning.

## Fonctionnalités
- Expose une API REST pour la prédiction (FastAPI).
- Implémente un moteur d'inférence pour renvoyer une classe et une probabilité.
- Tests unitaires pour l'API et le modèle.
- Pipeline CI/CD pour le déploiement continu.

## Installation
1. Clonez ce dépôt :
   ```bash
   git clone https://github.com/<votre-repo>
   cd project

Installez les dépendances :
pip install -r api/requirements.txt

Lancez l'API localement :
uvicorn api.app:app --reload

