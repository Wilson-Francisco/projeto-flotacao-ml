import pandas as pd
import os
from src.utils.logger import configurando_logger

logger = configurando_logger()

# Lista de colunas para validação de integridade
COLUNAS_REQUISITADAS = [
    'date', '% Iron Feed', '% Silica Feed', 'Starch Flow', 'Amina Flow', 
    'Ore Pulp Flow', 'Ore Pulp pH', 'Ore Pulp Density',
    'Flotation Column 01 Air Flow', 'Flotation Column 02 Air Flow', 'Flotation Column 03 Air Flow',
    'Flotation Column 04 Air Flow', 'Flotation Column 05 Air Flow', 'Flotation Column 06 Air Flow',
    'Flotation Column 07 Air Flow', 'Flotation Column 01 Level', 'Flotation Column 02 Level',
    'Flotation Column 03 Level', 'Flotation Column 04 Level', 'Flotation Column 05 Level',
    'Flotation Column 06 Level', 'Flotation Column 07 Level', '% Iron Concentrate',
    '% Silica Concentrat'
]

def ler_csv_da_planta(caminho_arquivo: str) -> pd.DataFrame:
    """
    Carrega o arquivo CSV com os dados dos sensores da planta de flotação
    e valida se a estrutura de colunas está correta.
    """
    if not os.path.exists(caminho_arquivo):
        error_msg = f"Arquivo de dados não encontrado no caminho: {caminho_arquivo}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
        
    logger.info(f"Iniciando leitura do arquivo: {caminho_arquivo}")
    df = pd.read_csv(caminho_arquivo)
    
    # Validação do contrato de dados (Schema Check)
    colunas_faltantes = [col for col in COLUNAS_REQUISITADAS if col not in df.columns]
    
    if colunas_faltantes:
        error_msg = f"Contrato de dados violado! Colunas ausentes no CSV: {colunas_faltantes}"
        logger.error(error_msg)
        raise ValueError(error_msg)
        
    logger.info(f"Leitura concluída com sucesso. Linhas carregadas: {len(df)}. Estrutura validada.")
    return df[COLUNAS_REQUISITADAS]
