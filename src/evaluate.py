import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.utils.logger import configurando_logger

logger = configurando_logger()

def calcular_metricas_regressao(y_real, y_predito) -> dict:
    """
    Calcula as métricas estatísticas de erro (R2, MAE, RMSE) 
    para avaliar a precisão do Sensor Virtual de sílica.
    """
    r2 = r2_score(y_real, y_predito)
    mae = mean_absolute_error(y_real, y_predito)
    rmse = np.sqrt(mean_squared_error(y_real, y_predito))
    
    logger.info("--- Métricas de Avaliação Calculadas ---")
    logger.info(f"R² (Explicação da variância): {r2:.4f}")
    logger.info(f"MAE (Erro Médio Absoluto): {mae:.4f} % de Sílica")
    logger.info(f"RMSE (Erro Quadrático Médio): {rmse:.4f} % de Sílica")
    
    return {
        "r2": float(r2),
        "mae": float(mae),
        "rmse": float(rmse)
    }
