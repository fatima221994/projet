# Utiliser l'image officielle Python 3.9 slim
FROM python:3.9-slim

# Mettre à jour les packages système et installer gcc, libatlas, etc.
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    libatlas-base-dev \
    libssl-dev \
    libffi-dev \
    python3-dev

# Définir le répertoire de travail
WORKDIR /api

# Copier d'abord les fichiers nécessaires à l'installation des dépendances
COPY requirements.txt /api/

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste de l'application dans le conteneur
COPY . /ap/

# Exposer le port 5005 pour Flask
EXPOSE 5005

# Commande pour démarrer l'application Flask avec gunicorn
# Utiliser 0.0.0.0 pour que l'application soit accessible de l'extérieur du conteneur
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5005"]

