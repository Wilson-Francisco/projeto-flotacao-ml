import os
import mlflow
from src.utils.logger import configurando_logger

logger = configurando_logger()

def iniciar_tracking_mlflow(nome_experimento: str = "Flotacao_Silica_Prediction") -> str:
    """
    Configura centralizadamente a URI de tracking e define o experimento 
    ativo no MLflow, adaptando-se entre o ambiente local e produção na nuvem.
    
    Retorna o ID do experimento ativo.
    """
    # Se houver uma URL de servidor remoto nas variáveis de ambiente da nuvem, usa ela.
    # Caso contrário, utiliza o armazenamento local padrão.
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "local")
    
    if mlflow_tracking_uri != "local":
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        logger.info(f"[MLflow] Conectado ao servidor de tracking remoto: {mlflow_tracking_uri}")
    else:
        logger.info("[MLflow] Utilizando armazenamento de tracking local (pasta mlruns/).")
        
    # Define ou cria o experimento para agrupar as rodadas (runs) de treino do XGBoost
    mlflow.set_experiment(nome_experimento)
    experimento = mlflow.get_experiment_by_name(nome_experimento)
    
    logger.info(f"[MLflow] Experimento ativo definido com sucesso: '{nome_experimento}' (ID: {experimento.experiment_id})")
    return experimento.experiment_id
