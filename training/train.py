import argparse
import glob
import json
import os

import mlflow
import mlflow.sklearn
import pandas as pd

from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    balanced_accuracy_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


MODEL_NAME = "bank-marketing-classifier"


def load_data():
    files = glob.glob("data/batches/*.csv")

    if not files:
        raise FileNotFoundError(
            "Aucun fichier CSV trouvé dans data/batches/"
        )

    dataframes = [pd.read_csv(f, sep=None, engine="python") for f in files]
    data = pd.concat(dataframes, ignore_index=True)

    if "y" not in data.columns:
        raise ValueError("La colonne cible 'y' est absente.")

    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--degrade", action="store_true")
    args = parser.parse_args()

    data = load_data()

    X = data.drop(columns=["y"])
    y = data["y"].astype(str).str.lower()

    # Encodage de la cible : no = 0, yes = 1
    y = y.map({"no": 0, "yes": 1})

    if y.isna().any():
        raise ValueError("La cible doit contenir uniquement yes et no.")

    y = y.astype(int)

    # Démonstration : mélanger volontairement les labels
    if args.degrade:
        y = y.sample(frac=1, random_state=42).reset_index(drop=True)

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])

    model = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("bank-marketing-training")

    with mlflow.start_run() as run:
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, zero_division=0),
            "recall": recall_score(y_test, predictions, zero_division=0),
            "f1": f1_score(y_test, predictions, zero_division=0),
        }

        mlflow.log_params({
            "model": "LogisticRegression",
            "class_weight": "balanced",
            "degraded_demo": args.degrade,
        })

        mlflow.log_metrics(metrics)

        mlflow.set_tags({
            "data_version": "initial",
            "git_sha": os.getenv("GITHUB_SHA", "local"),
        })

        signature = infer_signature(X_train, model.predict(X_train))

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            registered_model_name=MODEL_NAME,
            skops_trusted_types=["numpy.dtype"],
        )

        client = mlflow.MlflowClient()
        versions = client.search_model_versions(
            f"name='{MODEL_NAME}'",
            order_by=["version_number DESC"],
        )
        latest_version = int(versions[0].version)

        release = {
            "model_uri": f"models:/{MODEL_NAME}/{latest_version}",
            "model_version": latest_version,
            "data_version": "initial",
            "git_sha": os.getenv("GITHUB_SHA", "local"),
            "run_id": run.info.run_id,
            "metrics": metrics,
        }

        os.makedirs("artifacts", exist_ok=True)

        with open("artifacts/release.json", "w") as file:
            json.dump(release, file, indent=4)

        print("\nRésultats de l'entraînement :")
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")

        print("\nModèle enregistré dans MLflow :")
        print(release["model_uri"])


if __name__ == "__main__":
    main()