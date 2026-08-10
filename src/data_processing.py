import pandas as pd
from src.base_etl import BaseETL
from src.aquisicao.data_fetcher import ler_csv_da_planta
from src.utils.math_helpers import remover_outliers_iqr
from src.utils.logger import configurando_logger

logger = configurando_logger()

class FlotationETL(BaseETL):
    """
    Pipeline de ETL estruturado com base nas características reais dos dados.
    Aplica a padronização de nomes, limpeza de ruídos e criação de lags temporais,
    incluindo o lag da própria variável alvo (Target Lagging) como referência estável.
    """

    def extract(self) -> pd.DataFrame:
        """Busca e valida o arquivo CSV configurado usando o módulo de aquisição."""
        caminho_entrada = self.config.get("file_path", "data/MiningProcess_Flotation_Plant_Database.csv")
        return ler_csv_da_planta(caminho_entrada)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica limpeza de ruídos e gera os atrasos temporais (lags) provados."""
        logger.info("Iniciando transformações produtivas dos dados dos sensores...")
        df = df.copy()

        # Ajuste e padronização do nome da coluna alvo logo no início
        if '% Silica Concentrat' in df.columns:
            df = df.rename(columns={'% Silica Concentrat': '% Silica Concentrate'})
            
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)

        #  Conversão e Tratamento de Força Bruta para Tipos Numéricos (Evita NaNs)
        for col in df.columns:
            if col != 'date':
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str).str.strip().str.replace(',', '.', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Preenchimento de nulos pela Média da coluna
        for col in df.columns:
            if col != 'date':
                if df[col].isna().sum() > 0:
                    df[col] = df[col].fillna(df[col].mean())

        # Limpeza Estatística IQR na Densidade
        df = remover_outliers_iqr(df, coluna='Ore Pulp Density')

        # Engenharia de Recursos: Aplicação do Lag exato de 20 minutos (Provado no Notebook)
        # Incluímos a própria '% Silica Concentrate' para dar a referência do laboratório anterior ao modelo
        reagentes_e_processo = [
            'Starch Flow', 'Amina Flow', 'Ore Pulp Flow', 
            'Ore Pulp Density', '% Silica Concentrate'
        ]
        for col in reagentes_e_processo:
            df[f'{col}_lag20'] = df[col].shift(periods=20)

        # Remove linhas nulas geradas pelo efeito de shift/lag temporal
        df_final = df.dropna(subset=[f'{col}_lag20' for col in reagentes_e_processo]).reset_index(drop=True)
        
        logger.info(f"Dataset processado com sucesso com Target Lagging. Total de linhas: {len(df_final)}")
        return df_final

    def load(self, df: pd.DataFrame) -> None:
        """Salva os dados processados prontos para o Machine Learning."""
        caminho_saida = self.config.get("output_path", "data/dados_processados.csv")
        df.to_csv(caminho_saida, index=False)
        logger.info(f"Dados processados salvos com sucesso em: {caminho_saida}")
