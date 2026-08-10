import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock
import app.main

# 1. FIXTURE DE MONKEYPATCH: Roda ANTES de criar o cliente e simula o modelo online
@pytest.fixture(autouse=True)
def mock_modelo_mlflow(monkeypatch):
    """
    Injeta de forma automática um modelo simulado (Mock) no FastAPI 
    para que os testes de integração passem mesmo sem a pasta mlruns/ na nuvem.
    """
    # Criamos um objeto falso que simula o comportamento do modelo do XGBoost
    mock_model = MagicMock()
    # Ensinamos o modelo falso a retornar um array de float (ex: valor 1.95%) ao receber dados
    mock_model.predict.return_value = [1.95]
    
    # Forçamos o FastAPI a usar o nosso modelo simulado no lugar do original
    monkeypatch.setattr(app.main, "modelo_sensor_virtual", mock_model)


# Inicializa o cliente de testes virtual do FastAPI após a configuração do patch
client = TestClient(app.main.app)


@pytest.fixture
def payload_sensores_valido():
    """
    FIXTURE DO PYTEST: Fornece um payload com dados operacionais médios.
    """
    return {
        "% Iron Feed": 55.0,
        "Starch Flow": 2500.0,
        "Amina Flow": 500.0,
        "Ore Pulp Flow": 400.0,
        "Ore Pulp Density": 1.68,
        "Flotation Column 01 Air Flow": 250.0,
        "Flotation Column 02 Air Flow": 250.0,
        "Flotation Column 03 Air Flow": 250.0,
        "Flotation Column 04 Air Flow": 250.0,
        "Flotation Column 05 Air Flow": 250.0,
        "Flotation Column 06 Air Flow": 250.0,
        "Flotation Column 07 Air Flow": 250.0,
        "Flotation Column 01 Level": 500.0,
        "Flotation Column 02 Level": 500.0,
        "Flotation Column 03 Level": 500.0,
        "Flotation Column 04 Level": 500.0,
        "Flotation Column 05 Level": 500.0,
        "Flotation Column 06 Level": 500.0,
        "Flotation Column 07 Level": 500.0,
        "Starch Flow_lag20": 2480.0,
        "Amina Flow_lag20": 495.0,
        "Ore Pulp Flow_lag20": 398.0,
        "Ore Pulp Density_lag20": 1.67,
        "% Silica Concentrate_lag20": 1.9
    }


def test_endpoint_health_check():
    """
    Garante que o endpoint base '/' está respondendo com sucesso (200 OK)
    e que o Sensor Virtual simulado responde como ONLINE.
    """
    resposta = client.get("/")
    
    assert resposta.status_code == 200
    dados = resposta.json()
    assert dados["status_sistema"] == "OPERACIONAL"
    assert dados["sensor_virtual_silica"] == "ONLINE"


def test_endpoint_predict_com_payload_valido(payload_sensores_valido):
    """
    Garante que o endpoint '/predict' aceita dados válidos, executa a 
    validação e o modelo simulado, retornando a predição com sucesso (200 OK).
    """
    resposta = client.post("/predict", json=payload_sensores_valido)
    
    assert resposta.status_code == 200
    dados = resposta.json()
    assert "silica_concentrate_predicted" in dados
    assert dados["unidade"] == "%"
    assert dados["silica_concentrate_predicted"] == 1.95
