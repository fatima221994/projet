## Utilisation de l'image Python officielle comme base
FROM python:3.9-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier le fichier requirements.txt dans le conteneur
COPY requirements.txt /app/

# Installer les dépendances nécessaires
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste des fichiers du projet dans le conteneur
COPY . /app/

# Exposer le port 5000 pour Flask
EXPOSE 5005

# Commande pour démarrer l'application Flask avec gunicorn
CMD ["gunicorn", "app:app", "--bind", "127.0.0.1:5005"]

