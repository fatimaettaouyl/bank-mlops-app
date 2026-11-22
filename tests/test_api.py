
import pytest
from fastapi.testclient import TestClient

from serving.app import app


# Données fictives d'un client bancaire
CUSTOMER_DATA = {
    "age": 35,
    "job": "admin.",
    "marital": "married",
    "education": "university.degree",
    "default": "no",
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "month": "may",
    "day_of_week": "mon",
    "duration": 200,
    "campaign": 1,
    "pdays": 999,
    "previous": 0,
    "poutcome": "nonexistent",
    "emp.var.rate": 1.1,
    "cons.price.idx": 93.994,
    "cons.conf.idx": -36.4,
    "euribor3m": 4.857,
    "nr.employed": 5191.0
}


@pytest.fixture
def client():
    """Créer un client de test avec le modèle chargé."""

    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    """Vérifier que l'API et le modèle sont opérationnels."""

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["loaded"] is True
    assert data["model"] == "bank-marketing-classifier"


def test_predict(client):
    """Vérifier que l'API réalise une prédiction."""

    response = client.post(
        "/predict",
        json=CUSTOMER_DATA
    )

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] in ["yes", "no"]
    assert 0 <= data["probability_yes"] <= 1
    assert data["model"] == "bank-marketing-classifier"
    assert data["model_version"] == "1"


def test_predict_invalid_data(client):
    """Vérifier le rejet des données incomplètes."""

    response = client.post(
        "/predict",
        json={"age": 35}
    )

    assert response.status_code == 422


def test_predict_with_actual_y(client):
    """Vérifier la comparaison avec le résultat réel."""

    customer = CUSTOMER_DATA.copy()
    customer["actual_y"] = "yes"

    response = client.post(
        "/predict",
        json=customer
    )

    assert response.status_code == 200


def test_metrics(client):
    """Vérifier que Prometheus expose ses métriques."""

    response = client.get("/metrics/")

    assert response.status_code == 200
    assert "http_requests_total" in response.text
    assert "model_predictions_total" in response.text
