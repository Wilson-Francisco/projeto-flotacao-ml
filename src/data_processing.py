import pandas as pd
from src.base_etl import BaseETL
from src.aquisicao.data_fetcher import ler_csv_da_planta
from src.utils.math_helpers import remover_outliers_iqr
from src.utils.logger import configurando_logger




logger = configurando_logger()


class FlotationETL(BaseETL):
    """
    Implementação do pipeline de ETL para o processo industrial
    da planta de flotação de minério de ferro
    """

    def extract(self) -> pd.DataFrame:
        """Busca e valida o arquivo CSV configurado usando o módulo de aquisição"""
        caminho_entrada = self.config.get("file_path", "data/dados_flotacao.csv")
        return ler_csv_da_planta(caminho_entrada)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica limpeza de ruídos nos sensores e gera os atrasos temporais (lags)."""
        logger.info("Iniciando transformações dos dados dos sensores...")
        
        # Cria uma cópia explícita para evitar o erro de referência do Pandas
        df = df.copy()

        # 1. Garantir formatação da data e ordenação cronológica
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)

        # Converte todas as colunas de sensores de texto para número real
        # Varre todas as colunas, exceto a coluna 'date'
        for col in df.columns:
            if col != 'date':
                if df[col].dtype == 'object':
                    logger.info(f"Convertendo coluna ruidosa [{col}] de texto para float...")
                    df[col] = df[col].astype(str).str.replace(',', '.')
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Remove linhas que eventualmente ficaram com valores nulos após a conversão forcada
        df = df.dropna(subset=['Ore Pulp pH', 'Ore Pulp Density']).reset_index(drop=True)

        # Limpeza Estatística de Sensores (Agora 100% numéricos)
        df = remover_outliers_iqr(df, coluna='Ore Pulp pH')
        df = remover_outliers_iqr(df, coluna='Ore Pulp Density')

        # Criação de Lags Temporais
        reagentes_criticos = ['Starch Flow', 'Amina Flow', 'Ore Pulp Flow']
        for col in reagentes_criticos:
            df[f'{col}_lag15'] = df[col].shift(periods=15)
            df[f'{col}_lag30'] = df[col].shift(periods=30)

        # Remover linhas com valores nulos resultantes do efeito do shift
        linhas_antes = len(df)
        df = df.dropna().reset_index(drop=True)
        linhas_depois = len(df)
        
        logger.info(f"Removidas {linhas_antes - linhas_depois} linhas devido aos lags temporais.")
        return df


    def load(self, df: pd.DataFrame) -> None:
        """Salva o DataFrame final, limpo e enriquecido, pronto para o Machine Learning."""
        caminho_saida = self.config.get("output_path", "data/dados_processados.csv")
        df.to_csv(caminho_saida, index=False)
        logger.info(f"Dados processados carregados e salvos com sucesso em: {caminho_saida}")



if __name__ == "__main__":
    # Testando o pipeline completo localmente
    pipeline = FlotationETL(
        file_path="data/MiningProcess_Flotation_Plant_Database.csv", 
        output_path="data/dados_processados.csv"
    )
    
    pipeline.run()
