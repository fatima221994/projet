# Utiliser une image de base Python
FROM python:3.11-slim

# Installer des dépendances systèmes nécessaires
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Répertoire de travail dans le conteneur
WORKDIR /app

# Copier le fichier requirements.txt dans le conteneur
COPY requirements.txt /app/

# Mettre à jour pip et installer les dépendances
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copier tout le code source dans le conteneur
COPY . /app/

# Exposer le port utilisé par Flask
EXPOSE 5000

# Démarrer l'application avec Gunicorn
CMD ["gunicorn", "-w", "3", "app:app"]

