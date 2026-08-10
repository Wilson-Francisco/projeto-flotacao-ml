import pytest
from fastapi.testclient import TestClient
from app.main import app

# Inicializa o cliente de testes virtual do FastAPI
client = TestClient(app)

@pytest.fixture
def payload_sensores_valido():
    """
    FIXTURE DO PYTEST: Centraliza e fornece um payload com dados operacionais 
    médios e reais dos sensores da planta de flotação para os testes.
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
    e que o Sensor Virtual está online na inicialização da API.
    """
    resposta = client.get("/")
    
    assert resposta.status_code == 200
    dados = resposta.json()
    assert dados["status_sistema"] == "OPERACIONAL"
    assert dados["sensor_virtual_silica"] == "ONLINE"


def test_endpoint_predict_com_payload_valido(payload_sensores_valido):
    """
    Garante que o endpoint '/predict' aceita dados válidos, executa a 
    validação e o modelo XGBoost, retornando a predição da sílica com sucesso (200 OK).
    Nota: Recebe a fixture 'payload_sensores_valido' como argumento do PyTest.
    """
    resposta = client.post("/predict", json=payload_sensores_valido)
    
    assert resposta.status_code == 200
    dados = resposta.json()
    assert "silica_concentrate_predicted" in dados
    assert dados["unidade"] == "%"
    assert isinstance(dados["silica_concentrate_predicted"], float)
