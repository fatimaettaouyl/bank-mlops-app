
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import mlflow.sklearn
import pandas as pd

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    make_asgi_app,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

MODEL_NAME = "bank-marketing-classifier"
MODEL_VERSION = os.getenv("MODEL_VERSION", "1")
MODEL_PATH = os.getenv("MODEL_PATH", "serving/model")


# ============================================================
# 2. PROMETHEUS METRICS
# ============================================================

HTTP_REQUESTS = Counter(
    "http_requests_total",
    "Nombre total de requêtes HTTP",
    ["method", "endpoint", "status"],
)

HTTP_DURATION = Histogram(
    "http_request_duration_seconds",
    "Durée des requêtes HTTP en secondes",
    ["endpoint"],
)

MODEL_PREDICTIONS = Counter(
    "model_predictions_total",
    "Nombre total de prédictions",
    ["prediction"],
)

MODEL_LABELED_PREDICTIONS = Counter(
    "model_labeled_predictions_total",
    "Prédictions comparées aux résultats réels",
    ["actual", "predicted"],
)

MODEL_INFO = Gauge(
    "model_info",
    "Informations sur le modèle chargé",
    ["model_name", "model_version"],
)


# ============================================================
# 3. MODELE DE DONNEES POUR L'API
# ============================================================

class CustomerData(BaseModel):

    model_config = ConfigDict(populate_by_name=True)

    age: int
    job: str
    marital: str
    education: str
    default: str
    housing: str
    loan: str
    contact: str
    month: str
    day_of_week: str

    duration: int
    campaign: int
    pdays: int
    previous: int
    poutcome: str

    emp_var_rate: float = Field(alias="emp.var.rate")
    cons_price_idx: float = Field(alias="cons.price.idx")
    cons_conf_idx: float = Field(alias="cons.conf.idx")
    euribor3m: float
    nr_employed: float = Field(alias="nr.employed")

    # Facultatif : permet de comparer la prédiction
    # avec le résultat réel si celui-ci est connu.
    actual_y: Optional[str] = None


# ============================================================
# 4. CHARGEMENT DU MODELE
# ============================================================

model = None


@asynccontextmanager
async def lifespan(app: FastAPI):

    global model

    try:
        print(f"Chargement du modèle depuis : {MODEL_PATH}")

        model = mlflow.sklearn.load_model(MODEL_PATH)

        MODEL_INFO.labels(
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION,
        ).set(1)

        print("Modèle chargé avec succès !")

    except Exception as error:
        print(f"Erreur de chargement du modèle : {error}")
        raise RuntimeError(
            "Impossible de charger le modèle ML."
        ) from error

    yield

    print("Arrêt de l'application FastAPI.")


# ============================================================
# 5. CREATION DE L'APPLICATION FASTAPI
# ============================================================

app = FastAPI(
    title="Bank Marketing ML API",
    description="API de prédiction de souscription bancaire",
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================
# 6. MIDDLEWARE : SUIVI DES REQUETES HTTP
# ============================================================

@app.middleware("http")
async def monitor_requests(request: Request, call_next):

    start_time = time.perf_counter()

    response = await call_next(request)

    duration = time.perf_counter() - start_time

    endpoint = request.url.path

    HTTP_REQUESTS.labels(
        method=request.method,
        endpoint=endpoint,
        status=str(response.status_code),
    ).inc()

    HTTP_DURATION.labels(
        endpoint=endpoint
    ).observe(duration)

    return response


# ============================================================
# 7. ENDPOINT : HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy" if model is not None else "unavailable",
        "model": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "loaded": model is not None,
    }


# ============================================================
# 8. ENDPOINT : PREDICTION
# ============================================================

@app.post("/predict")
def predict(customer: CustomerData):

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Le modèle n'est pas encore chargé.",
        )

    try:
        # Convertir les données reçues en DataFrame
        customer_data = customer.model_dump(
            by_alias=True,
            exclude={"actual_y"},
        )

        df = pd.DataFrame([customer_data])

        # Effectuer la prédiction
        prediction_value = int(model.predict(df)[0])

        # Calculer la probabilité de souscription (classe 1 = yes)
        probability_yes = float(
            model.predict_proba(df)[0][1]
        )

        prediction = "yes" if prediction_value == 1 else "no"

        # Mettre à jour les métriques Prometheus
        MODEL_PREDICTIONS.labels(
            prediction=prediction
        ).inc()

        # Comparer avec le résultat réel si disponible
        if customer.actual_y is not None:

            actual = customer.actual_y.strip().lower()

            if actual not in ["yes", "no"]:
                raise HTTPException(
                    status_code=422,
                    detail="actual_y doit être 'yes' ou 'no'.",
                )

            MODEL_LABELED_PREDICTIONS.labels(
                actual=actual,
                predicted=prediction,
            ).inc()

        return {
            "prediction": prediction,
            "probability_yes": round(probability_yes, 4),
            "model": MODEL_NAME,
            "model_version": MODEL_VERSION,
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur pendant la prédiction : {str(error)}",
        )


# ============================================================
# 9. ENDPOINT : PROMETHEUS METRICS
# ============================================================

metrics_app = make_asgi_app()

app.mount("/metrics", metrics_app)
