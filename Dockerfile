# Étape 1 : Utiliser une image de base Python optimisée
FROM python:3.9-slim

# Étape 2 : Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Étape 3 : Copier les fichiers nécessaires dans le conteneur
COPY . /app

# Étape 4 : Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Étape 5 : Exposer le port de l'application Flask
EXPOSE 5000

# Étape 6 : Définir la commande pour exécuter l'application
CMD ["python", "api/app.py"]

