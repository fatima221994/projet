# Utilisation de l'image officielle Python
FROM python:3.9-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier les fichiers nécessaires dans le conteneur
COPY requirements.txt /app/

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le code source dans le conteneur
COPY . /app/

# Exposer le port sur lequel l'application écoutera
EXPOSE 5000

# Commande pour démarrer l'application Flask avec gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]

