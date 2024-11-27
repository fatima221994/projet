#!/bin/bash

# 1. Mettre à jour le système et installer les dépendances
echo "Mise à jour du système..."
sudo apt-get update -y
sudo apt-get install -y python3-pip python3-dev git

# 2. Cloner le dépôt Git dans le répertoire de travail
echo "Clonage du dépôt..."
if [ -d "/opt/render/project/src" ]; then
  echo "Le répertoire existe déjà, on va le supprimer..."
  rm -rf /opt/render/project/src
fi
git clone https://github.com/votre-utilisateur/votre-depot.git /opt/render/project/src

# 3. Naviguer dans le répertoire du projet
cd /opt/render/project/src

# 4. Installer les dépendances Python via le fichier requirements.txt
echo "Installation des dépendances..."
pip install --no-cache-dir -r requirements.txt

# 5. Lancer l'application Flask
echo "Lancement de l'application Flask..."
export FLASK_APP=app.py
export FLASK_ENV=production
flask run --host=0.0.0.0 --port=5000

