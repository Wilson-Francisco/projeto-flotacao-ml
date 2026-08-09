import pandas as pd
from src.base_etl import BaseETL
from src.aquisicao.data_fetcher import ler_csv_da_planta
from src.utils.math_helpers import remover_outliers_iqr
from src.utils.logger import configurando_logger

logger = configurando_logger()

class FlotationETL(BaseETL):
    """
    Pipeline de ETL otimizado com base nos testes de hipóteses estatísticas.
    """

    def extract(self) -> pd.DataFrame:
        """Busca e valida o arquivo CSV configurado usando o módulo de aquisição."""
        caminho_entrada = self.config.get("file_path", "data/MiningProcess_Flotation_Plant_Database.csv")
        return ler_csv_da_planta(caminho_entrada)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica limpeza de ruídos, gera lags provados e balanceia o alvo."""
        logger.info("Iniciando transformações produtivas dos dados dos sensores...")
        df = df.copy()

        #  Ajuste de nome de coluna, data e ordenação cronológica
        if '% Silica Concentrat' in df.columns:
            df = df.rename(columns={'% Silica Concentrat': '% Silica Concentrate'})
            
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)

        # Conversão e Tratamento de Força Bruta para Tipos Numéricos (Evita NaNs)
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

        # Limpeza Estatística IQR (Apenas nas colunas que variam)
        df = remover_outliers_iqr(df, coluna='Ore Pulp Density')

        # Aplicação do Lag exato de 20 minutos (Provado no Notebook)
        reagentes_e_processo = ['Starch Flow', 'Amina Flow', 'Ore Pulp Flow', 'Ore Pulp Density']
        for col in reagentes_e_processo:
            df[f'{col}_lag20'] = df[col].shift(periods=20)

        # Remove linhas nulas geradas pelo shift
        df = df.dropna(subset=[f'{col}_lag20' for col in reagentes_e_processo]).reset_index(drop=True)

        # Estratégia de Subamostragem (Undersampling) para remover o vício do alvo (99% em 1.9)
        df_variacao = df[df['% Silica Concentrate'] != 1.9]
        df_travado = df[df['% Silica Concentrate'] == 1.9]

        # Mantém uma amostra de 10.000 registros do valor majoritário para equilíbrio
        tamanho_amostra = min(10000, len(df_travado))
        df_travado_amostrado = df_travado.sample(n=tamanho_amostra, random_state=42)

        # Une as bases novamente mantendo a ordenação temporal
        df_final = pd.concat([df_variacao, df_travado_amostrado]).sort_values('date').reset_index(drop=True)
        
        logger.info(f"Dataset reduzido de {len(df)} para {len(df_final)} linhas após balanceamento do alvo.")
        return df_final

    def load(self, df: pd.DataFrame) -> None:
        """Salva os dados processados e balanceados prontos para o MLflow."""
        caminho_saida = self.config.get("output_path", "data/dados_processados.csv")
        df.to_csv(caminho_saida, index=False)
        logger.info(f"Dados balanceados salvos com sucesso em: {caminho_saida}")







