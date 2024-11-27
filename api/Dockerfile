# Utiliser l'image officielle Python 3.9 slim
FROM python:3.9-slim

# Définir un serveur DNS explicite pour résoudre les noms de domaine
RUN echo "nameserver 8.8.8.8" > /etc/resolv.conf

# Remplacer le dépôt par un miroir Debian fiable
RUN sed -i 's|http://deb.debian.org|http://ftp.debian.org|g' /etc/apt/sources.list

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

# Copier le fichier requirements.txt dans le conteneur
COPY requirements.txt /app/

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste de l'application dans le conteneur
COPY . /api/

# Exposer le port 5005 pour Flask
EXPOSE 5005

# Commande pour démarrer l'application Flask avec gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5005"]

