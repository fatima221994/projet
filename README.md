# API de Scoring de Crédit

## **Objectif du Projet**
Cette API a pour but de prédire la probabilité de défaut de paiement d'un client en fonction de ses caractéristiques. Elle utilise un modèle de machine learning (XGBoost) entraîné avec des données prétraitées, en intégrant une logique métier 10 * fn + fp.

## **Fonctionnalités**
- **Prétraitement des données** : L'API prend en entrée un JSON avec les caractéristiques du client, puis les prétraite avant de les envoyer au modèle.
- **Prédiction binaire avec seuil** : L'API effectue une prédiction binaire (Accordé/refusé) en fonction de la probabilité calculée par le modèle et d'un seuil métier.
- **Retour des probabilités** : L'API retourne la probabilité que le client fasse défaut ainsi que la prédiction finale (1 pour défaut, 0 pour non-défectueux).
- **Fonction de Coût Métier**

Dans ce projet, nous utilisons une **fonction de coût métier** qui permet de donner un poids spécifique aux erreurs de type **faux négatifs (FN)** et **faux positifs (FP)**. Cette fonction est particulièrement utile lorsque le coût des erreurs n'est pas équivalent. La formule du coût est la suivante :

\[
\text{Coût} = 10 \times \text{FN} + \text{FP}
\]

### Détails sur la Fonction de Coût
- **Faux Négatifs (FN)** : Représente les clients qui devraient faire défaut mais qui sont classés comme solvables. Ces erreurs sont considérées comme plus coûteuses, avec un poids de 10.
- **Faux Positifs (FP)** : Représente les clients qui sont classés comme faisant défaut mais qui sont en réalité solvables. Ces erreurs ont un poids de 1.


## **Découpage des Dossiers**
- **`/models/`** : Contient le modèle entraîné et le préprocesseur utilisés par l'API.
  - `xgb_model_with_smote_and_score_metier_etape_par_etape.pkl` : Modèle XGBoost.
  - `preprocessor.pkl` : Préprocesseur pour transformer les données.
- **`/api/`** : Dossier contenant le code de l'API Flask.
  - `app.py` : Fichier principal contenant le code de l'API Flask.
- **`/notebooks/`** : Contient les notebooks Jupyter utilisés pour l'analyse exploratoire des données, le prétraitement, l'entraînement et l'évaluation du modèle.
  - **`exploration_data.ipynb`** : Notebook d'exploration des données.
  - **`code_etape_par_etape_openclassroom_feature_engineering_model_SHAP_MLFLOW.ipynb`** : Notebook de l'entraînement du modèle.
- **`/tests/`** : Contient les tests unitaires pour valider le bon fonctionnement de l'API.
- **`requirements.txt`** : Liste des packages nécessaires pour exécuter le projet.
- **`Dockerfile`** : Fichier pour construire l'image Docker et déployer l'application.
- **`Procfile`** : Fichier de configuration pour le déploiement sur Heroku.

## **Exemple de Flux d'Utilisation**
1. **Lancer l'API** :
   Pour lancer l'API localement, exécutez le fichier `app.py` :
   ```bash
   python app.py


Exemple de Flux d'Utilisation
1. Lancer l'API

Exécutez le fichier principal Flask pour démarrer l'API :

python api/app.py

L'API sera accessible à http://localhost:5000.
2. Tester l'API

Envoyez une requête POST avec des données client. Par exemple, utilisez curl ou postman :

curl -X POST https://project-science-free-014cfbe31914.herokuapp.com/predict \
     -H "Content-Type: application/json" \
     -d '{
         "SK_ID_CURR": 100001,
         "NAME_CONTRACT_TYPE": "Cash loans",
         "CODE_GENDER": "M",
         "FLAG_OWN_CAR": "Y",
         "FLAG_OWN_REALTY": "Y",
         "CNT_CHILDREN": 1,
         "AMT_INCOME_TOTAL": 25000,
         "AMT_CREDIT": 200000,
         "AMT_ANNUITY": 10000,
         "AMT_GOODS_PRICE": 150000,
         "NAME_TYPE_SUITE": "Unaccompanied",
         "NAME_INCOME_TYPE": "Working",
         "NAME_EDUCATION_TYPE": "Secondary / secondary special",
         "NAME_FAMILY_STATUS": "Single / not married",
         "NAME_HOUSING_TYPE": "Rented apartment",
         "REGION_POPULATION_RELATIVE": 0.001234,
         "DAYS_BIRTH": -15000,
         "DAYS_EMPLOYED": -1000,
         "DAYS_REGISTRATION": -2000,
         "DAYS_ID_PUBLISH": -3000,
         "OWN_CAR_AGE": 5,
         "FLAG_MOBIL": 1,
         "FLAG_EMP_PHONE": 1,
         "FLAG_WORK_PHONE": 0,
         "FLAG_CONT_MOBILE": 1,
         "FLAG_PHONE": 0,
         "FLAG_EMAIL": 1,
         "OCCUPATION_TYPE": "Laborers",
         "CNT_FAM_MEMBERS": 3,
         "REGION_RATING_CLIENT": 3,
         "REGION_RATING_CLIENT_W_CITY": 2,
         "WEEKDAY_APPR_PROCESS_START": 1,
         "HOUR_APPR_PROCESS_START": 10,
         "REG_REGION_NOT_LIVE_REGION": 0,
         "REG_REGION_NOT_WORK_REGION": 0,
         "LIVE_REGION_NOT_WORK_REGION": 0,
         "REG_CITY_NOT_LIVE_CITY": 0,
         "REG_CITY_NOT_WORK_CITY": 0,
         "LIVE_CITY_NOT_WORK_CITY": 0,
         "ORGANIZATION_TYPE": "Business Entity Type 1",
         "EXT_SOURCE_1": 0.5,
         "EXT_SOURCE_2": 0.3,
         "EXT_SOURCE_3": 0.4,
         "APARTMENTS_AVG": 1,
         "BASEMENTAREA_AVG": 1,
         "YEARS_BEGINEXPLUATATION_AVG": 10,
         "YEARS_BUILD_AVG": 20,
         "COMMONAREA_AVG": 0.2,
         "ELEVATORS_AVG": 2,
         "ENTRANCES_AVG": 1,
         "FLOORSMAX_AVG": 5,
         "FLOORSMIN_AVG": 1,
         "LANDAREA_AVG": 200,
         "LIVINGAPARTMENTS_AVG": 1,
         "LIVINGAREA_AVG": 80,
         "NONLIVINGAPARTMENTS_AVG": 0,
         "NONLIVINGAREA_AVG": 20
     }'


3. Réponse Exemple

{"prediction":1,"probability":0.5714335441589355}



Tests

Les tests unitaires permettent de vérifier le bon fonctionnement de l'API. Ils se trouvent dans le dossier tests/.

Pour exécuter les tests avec pytest :

pytest

Exécution avec Docker

    Construisez l'image Docker :

docker build -t flask-app .

Lancez le conteneur :

    sudo docker run -p 5000:5000 flask-app

Prérequis

    Python 3.9 ou version ultérieure.
    Les packages listés dans requirements.txt.

Pour installer les dépendances :

pip install -r requirements.txt


Streamlit :

Lancer l'interface Streamlit : Pour exécuter l'application Streamlit, allez dans le dossier contenant le fichier streamlit-app.py et exécutez la commande suivante :

streamlit run streamlit-app.py


Accédez à l'API déployée :

https://project-science-free-014cfbe31914.herokuapp.com

Tests

Pour tester l'API, vous pouvez utiliser les tests unitaires présents dans le dossier /tests/. Exécutez les tests avec pytest :

pytest


## Installation
1. Clonez ce dépôt :
   ```bash
   git clone https://github.com/fatima221994
   cd project

Installez les dépendances :
pip install -r api/requirements.txt

Lancez l'API localement :
uvicorn api.app:app --reload



