import pandas as pd
import numpy as np
from src.utils.logger import configurando_logger

logger = configurando_logger()

def remover_outliers_iqr(df: pd.DataFrame, coluna: str) -> pd.DataFrame:
    """
    Aplica o método do Intervalo Interquartil (IQR) em colunas já numéricas.
    """
    if coluna not in df.columns:
        logger.warning(f"A coluna {coluna} não existe no DataFrame para limpeza.")
        return df
        
    # Calcula os quartis estatísticos
    q1 = df[coluna].quantile(0.25)
    q3 = df[coluna].quantile(0.75)
    iqr = q3 - q1
    
    limite_inferior = q1 - 1.5 * iqr
    limite_superior = q3 + 1.5 * iqr
    
    linhas_antes = len(df)
    df_filtrado = df[(df[coluna] >= limite_inferior) & (df[coluna] <= limite_superior)]
    linhas_depois = len(df_filtrado)
    
    logger.info(f"Limpeza IQR na coluna [{coluna}]: {linhas_antes - linhas_depois} outliers removidos.")
    
    return df_filtrado.reset_index(drop=True)


