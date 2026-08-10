import os
import pandas as pd
import numpy as np
import mlflow.pyfunc
from fastapi import FastAPI, HTTPException
from app.schemas import PayloadSensoresPlanta
from src.utils.logger import configurando_logger

logger = configurando_logger()

# Inicializa a aplicação FastAPI
app = FastAPI(
    title="Sensor Virtual de Flotação - ML",
    description="API de predição em tempo real conectada ao servidor remoto do MLflow.",
    version="1.0.0"
)

# Captura as credenciais e endereços do servidor externo do MLflow
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "local")
RUN_ID = os.getenv("MLFLOW_RUN_ID")

# Inicializamos a variável global do modelo como None
modelo_sensor_virtual = None


try:
    if TRACKING_URI != "local" and RUN_ID:
        logger.info(f"[MLflow] Conectando ao servidor de tracking externo: {TRACKING_URI}")
        mlflow.set_tracking_uri(TRACKING_URI)
        
        # Busca pelo caminho padrão do artefato registrado
        try:
            model_uri = f"runs:/{RUN_ID}/modelo_xgboost_silica"
            logger.info(f"[MLflow] Tentando baixar artefato pela URI: {model_uri}")
            modelo_sensor_virtual = mlflow.pyfunc.load_model(model_uri)
        except Exception as e_first:
            logger.warning(f"Falha na URI padrão ({e_first}). Tentando buscar direto na raiz dos artefatos da run...")
            # Caso o DagsHub tenha jogado o modelo direto na raiz da execução
            model_uri = f"runs:/{RUN_ID}/"
            modelo_sensor_virtual = mlflow.pyfunc.load_model(model_uri)

        if modelo_sensor_virtual is not None:
            logger.info("Sucesso! O Sensor Virtual Real (XGBoost) foi carregado do servidor externo e está ONLINE.")
    else:
        logger.critical("[Erro] Variáveis de ambiente MLFLOW_TRACKING_URI ou MLFLOW_RUN_ID não configuradas.")
        
except Exception as e:
    logger.critical(f"Erro crítico definitivo ao conectar ou baixar o modelo do servidor externo: {e}")


# Endpoints da API
@app.get("/")
def home():
    """Endpoint de checagem de saúde (Health Check) do sistema industrial."""
    status_modelo = "ONLINE" if modelo_sensor_virtual is not None else "OFFLINE/ERRO_CONEXAO"
    return {
        "status_sistema": "OPERACIONAL",
        "sensor_virtual_silica": status_modelo,
        "provedor_tracking": "Servidor MLflow Externo" if TRACKING_URI != "local" else "Configuração Local Incompleta"
    }

@app.post("/predict")
def predizer_silica(payload: PayloadSensoresPlanta):
    """Recebe os dados instantâneos das colunas e calcula o teor de sílica em tempo real."""
    if modelo_sensor_virtual is None:
        logger.error("Requisição rejeitada. O modelo preditivo está offline por falha no MLflow externo.")
        raise HTTPException(status_code=503, detail="O modelo de Machine Learning externo não foi inicializado.")
        
    try:
        dados_dict = payload.model_dump(by_alias=True)
        df_input = pd.DataFrame([dados_dict])
        predicao = modelo_sensor_virtual.predict(df_input)
        valor_predito = float(predicao[0]) if isinstance(predicao, (list, np.ndarray)) else float(predicao)
        
        logger.info(f"Inferência executada com sucesso. Valor Predito: {valor_predito:.4f}% de Sílica.")
        return {
            "silica_concentrate_predicted": round(valor_predito, 4),
            "unidade": "%"
        }
    except Exception as e:
        logger.error(f"Falha interna durante a execução do modelo: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno de processamento: {e}")
