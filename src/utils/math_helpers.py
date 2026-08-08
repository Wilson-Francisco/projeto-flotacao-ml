import pandas as pd
import numpy as np
from src.utils.logger import configurando_logger

logger = configurando_logger()

def remover_outliers_iqr(df: pd.DataFrame, coluna: str) -> pd.DataFrame:
    """
    Aplica o método estatístico do Intervalo Interquartil (IQR) para 
    identificar e remover outliers de uma coluna específica de sensores.
    """
    if coluna not in df.columns:
        logger.warning(f"A coluna {coluna} não existe no DataFrame para limpeza.")
        return df
        
    # Calcula os quartis 25% (Q1) e 75% (Q3)
    q1 = df[coluna].quantile(0.25)
    q3 = df[coluna].quantile(0.75)
    
    # Calcula a amplitude do intervalo (IQR)
    iqr = q3 - q1
    
    # Define os limites aceitáveis
    limite_inferior = q1 - 1.5 * iqr
    limite_superior = q3 + 1.5 * iqr
    
    # Conta quantas linhas serão removidas para registrar no log
    linhas_antes = len(df)
    df_filtrado = df[(df[coluna] >= limite_inferior) & (df[coluna] <= limite_superior)]
    linhas_depois = len(df_filtrado)
    
    linhas_removidas = linhas_antes - linhas_depois
    logger.info(f"Limpeza IQR na coluna [{coluna}]: {linhas_removidas} outliers removidos.")
    
    return df_filtrado.reset_index(drop=True)
