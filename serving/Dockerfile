
# 1. Image Python de base
FROM python:3.10-slim

# 2. Dossier de travail dans le conteneur
WORKDIR /app

# 3. Copier les dépendances
COPY serving/requirements.txt /app/requirements.txt

# 4. Installer les bibliothèques
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copier le code FastAPI et le modèle exporté
COPY serving/ /app/serving/

# 6. Configurer le chemin du modèle
ENV MODEL_PATH=/app/serving/model
ENV MODEL_VERSION=1

# 7. Exposer le port de l'API
EXPOSE 8000

# 8. Démarrer FastAPI
CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
