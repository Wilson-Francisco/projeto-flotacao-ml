import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost
from xgboost import XGBRegressor

from src.data_processing import FlotationETL
from src.evaluate import calcular_metricas_regressao
from src.tracking.mlflow_config import iniciar_tracking_mlflow
from src.utils.logger import configurando_logger

logger = configurando_logger()

def executar_treinamento_modelo():
    """
    Orquestra o pipeline completo: executa o ETL, realiza a divisão temporal,
    treina o XGBoost mantendo o Target Lagging e registra tudo no MLflow.
    """
    # Inicializa a infraestrutura de MLOps do MLflow
    iniciar_tracking_mlflow(nome_experimento="Flotacao_Silica_Prediction")
    
    # Executa o pipeline de engenharia de dados (ETL)
    pipeline_dados = FlotationETL(
        file_path="data/MiningProcess_Flotation_Plant_Database.csv",
        output_path="data/dados_processados.csv"
    )
    df_processado = pipeline_dados.run()
    
    # Define as colunas de entrada (X) e o alvo (Y)
    # ATENÇÃO: Removemos apenas a Sílica atual (alvo), mantendo a '% Silica Concentrate_lag20' em X!
    colunas_descarte = ['date', '% Silica Concentrate', '% Iron Concentrate', '% Silica Feed', 'Ore Pulp pH']
    X_columns = [col for col in df_processado.columns if col not in colunas_descarte]
    y_column = '% Silica Concentrate'
    
    X = df_processado[X_columns]
    y = df_processado[y_column]
    
    logger.info(f"Quantidade de atributos selecionados para o modelo (X): {len(X_columns)}")
    
    # Divisão Temporal Rígida (80% treino cronológico, 20% teste futuro)
    split_index = int(len(df_processado) * 0.8)
    
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    
    logger.info(f"Dados divididos com sucesso: {len(X_train)} linhas para treino e {len(X_test)} linhas para teste.")
    
    # Configura e abre a execução (Run) ativa no MLflow
    with mlflow.start_run() as run:
        logger.info("Iniciando treinamento do XGBoost Regressor com parâmetros ampliados...")
        
        # Parâmetros otimizados para aprender com o volume real de dados
        params = {
            "n_estimators": 250,
            "max_depth": 10,
            "learning_rate": 0.05,
            "random_state": 42,
            "n_jobs": -1
        }
        
        # Loga os parâmetros automaticamente no painel do MLflow
        mlflow.log_params(params)
        
        # Treinamento do Modelo
        modelo_xgb = XGBRegressor(**params)
        modelo_xgb.fit(X_train, y_train)
        
        # Predições e Avaliação Estatística de Erro
        logger.info("Avaliando predições na base de teste temporal...")
        y_pred = modelo_xgb.predict(X_test)
        
        # Calculates metrics using src/evaluate.py
        metricas = calcular_metricas_regressao(y_test, y_pred)
        
        # Loga as métricas calculadas (R2, MAE, RMSE) no MLflow
        for nome_metrica, valor in metricas.items():
            mlflow.log_metric(nome_metrica, valor)
            
        # Salva o modelo treinado como artefato oficial no repositório do MLflow
        mlflow.xgboost.log_model(modelo_xgb, artifact_path="modelo_xgboost_silica")
        
        logger.info(f"Sucesso! Modelo registrado no MLflow. Run ID: {run.info.run_id}")

if __name__ == "__main__":
    executar_treinamento_modelo()
