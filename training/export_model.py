
import os
import shutil
import mlflow
import mlflow.sklearn

MODEL_NAME = "bank-marketing-classifier"
MODEL_VERSION = "1"

MODEL_URI = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
OUTPUT_PATH = "serving/model"


def export_model():
    print(f"Chargement du modèle : {MODEL_URI}")

    # 1. Charger la version précise depuis MLflow
    model = mlflow.sklearn.load_model(MODEL_URI)

    # 2. Préparer le dossier de destination
    os.makedirs("serving", exist_ok=True)

    if os.path.exists(OUTPUT_PATH):
        shutil.rmtree(OUTPUT_PATH)

    # 3. Exporter le modèle
    mlflow.sklearn.save_model(
        sk_model=model,
        path=OUTPUT_PATH,
        skops_trusted_types=["numpy.dtype"]
    )

    print(f"Modèle exporté dans : {OUTPUT_PATH}")
    print("Export terminé avec succès !")


if __name__ == "__main__":
    export_model()
