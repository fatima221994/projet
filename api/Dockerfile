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
WORKDIR /app

# Copier le fichier requirements.txt
COPY requirements.txt /app/

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste de l'application dans le conteneur
COPY . /app/


# Exposer le port 5000 pour Flask
EXPOSE 5005

# Commande pour démarrer l'application Flask avec gunicorn
CMD ["gunicorn", "app:app", "--bind", "127.0.0.1:5005"]

