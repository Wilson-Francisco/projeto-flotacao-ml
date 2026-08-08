from abc import ABC, abstractmethod
import pandas as pd
from src.utils.logger import configurando_logger


logger = configurando_logger()


class BaseETL(ABC):
    """
    Classe Abstrata Base que define o ciclo de vida rigoroso de qualquer 
    módulo de ETL (Extract, Transform, Load) no projeto.
    """

    def __init__(self, **kwargs):
        """
        Inicializa a classe permitindo a passagem de dicionários de configuração
        (caminhos de arquivos, parâmetros de lags, etc).
        """
        self.config = kwargs

    @abstractmethod
    def extract(self) -> pd.DataFrame:
        """Método obrigatório para extrair/ler os dados brutos de uma fonte."""
        pass

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Método obrigatório para limpar dados, tratar outliers e criar recursos."""
        pass

    @abstractmethod
    def load(self, df: pd.DataFrame) -> None:
        """Método obrigatório para salvar o resultado tratado no destino correto."""
        pass

    def run(self) -> pd.DataFrame:
        """
        Orquestrador padrão e centralizado do pipeline. Executa os métodos 
        na ordem lógica correta garantindo o fluxo contínuo dos dados.
        """
        nome_classe = self.__class__.__name__
        logger.info(f"[{nome_classe}] Iniciando execução do pipeline de ETL...")
        
        # 1. Extração
        dados_brutos = self.extract()
        
        # 2. Transformação
        dados_transformados = self.transform(dados_brutos)
        
        # 3. Carregamento
        self.load(dados_transformados)
        
        logger.info(f"[{nome_classe}] Pipeline de ETL concluído com sucesso de ponta a ponta!\n")
        return dados_transformados