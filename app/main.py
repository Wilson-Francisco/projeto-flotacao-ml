import os
import pandas as pd
import mlflow.pyfunc
from fastapi import FastAPI, HTTPException
from app.schemas import PayloadSensoresPlanta
from src.utils.logger import configurando_logger

logger = configurando_logger()

# Inicializa a aplicação FastAPI
app = FastAPI(
    title="Sensor Virtual de Flotação - ML",
    description="API de predição em tempo real para o teor de % Sílica no Concentrado Final.",
    version="1.0.0"
)

# Carrega o modelo de forma global na inicialização da API (Garante alta performance)
# Em ambiente local, vamos buscar o modelo da última execução salva na pasta mlruns
# Em produção na nuvem, passaremos o Run ID real via variável de ambiente
RUN_ID = os.getenv("MLFLOW_RUN_ID", "local")

if RUN_ID == "local":
    # Pega o caminho local padrão que o MLflow gerou no seu computador (ID do experimento é 1)
    # Buscar a pasta 'modelo_xgboost_silica' dentro de mlartifacts
    # Nota: Caso sua pasta tenha um hash diferente, o mlflow.pyfunc resolve pela URI oficial
    model_uri = "models:/Flotacao_Silica_Prediction/1" if not os.path.exists("mlruns") else "mlruns/1/"
    
    # Usar um atalho seguro para carregar o modelo local do MLflow
    # Procurar a última execução válida dentro do diretório local
    try:
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        runs = client.search_runs(experiment_ids=["1"])
        if runs:
            ultimo_run_id = runs[0].info.run_id
            model_uri = f"runs:/{ultimo_run_id}/modelo_xgboost_silica"
            logger.info(f"Carregando o melhor modelo local do MLflow. Run ID selecionado: {ultimo_run_id}")
        else:
            raise Exception("Nenhuma execução encontrada no MLflow local.")
    except Exception as e:
        logger.warning(f"Falha ao mapear MlflowClient local ({e}). Tentando fallback por arquivo.")
        # Fallback de segurança caso o banco do MLflow local esteja travado
        model_uri = "mlruns/1/" # Ajustaremos para o caminho correto se necessário
else:
    model_uri = f"runs:/{RUN_ID}/modelo_xgboost_silica"

try:
    logger.info(f"Conectando à URI do modelo no MLflow: {model_uri}")
    modelo_sensor_virtual = mlflow.pyfunc.load_model(model_uri)
    logger.info("Sucesso! O Sensor Virtual de Flotação (XGBoost) foi carregado e está ONLINE.")
except Exception as e:
    logger.critical(f"Erro crítico ao carregar o modelo de IA: {e}")
    modelo_sensor_virtual = None


# Endpoints da API
@app.get("/")
def home():
    """Endpoint de checagem de saúde (Health Check) do sistema industrial."""
    status_modelo = "ONLINE" if modelo_sensor_virtual is not None else "OFFLINE/ERRO_CARREGAMENTO"
    return {
        "status_sistema": "OPERACIONAL",
        "sensor_virtual_silica": status_modelo,
        "provedor_tracking": "MLflow Local" if RUN_ID == "local" else "Servidor Nuvem"
    }

@app.post("/predict")
def predizer_silica(payload: PayloadSensoresPlanta):
    """
    Recebe os dados instantâneos das 7 colunas e calcula o teor de sílica em tempo real.
    """
    if modelo_sensor_virtual is None:
        logger.error("Requisição rejeitada. O modelo preditivo está offline.")
        raise HTTPException(status_code=503, detail="O modelo de Machine Learning não foi inicializado.")
        
    try:
        # Converte o payload recebido do Pydantic em um dicionário do Python
        # O model_dump(by_alias=True) garante que os nomes reais com espaços  sejam mantidos
        dados_dict = payload.model_dump(by_alias=True)
        
        # Transforma o registro em um DataFrame do Pandas (formato que o XGBoost espera)
        df_input = pd.DataFrame([dados_dict])
        
        # Executa a inferência em milissegundos
        predicao = modelo_sensor_virtual.predict(df_input)
        
        # Retorna o resultado limpando o ponto flutuante do numpy para float nativo do JSON
        valor_predito = float(predicao[0])
        logger.info(f"Inferência executada com sucesso. Valor Predito: {valor_predito:.4f}% de Sílica.")
        
        return {
            "silica_concentrate_predicted": round(valor_predito, 4),
            "unidade": "%"
        }
        
    except Exception as e:
        logger.error(f"Falha interna durante a execução do modelo: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno de processamento: {e}")
