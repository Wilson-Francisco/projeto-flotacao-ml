import pandas as pd
import numpy as np
import pytest
from src.data_processing import FlotationETL
from src.aquisicao.data_fetcher import COLUNAS_REQUISITADAS

@pytest.fixture
def base_dados_valida():
    """
    FIXTURE DO PYTEST: Cria uma base de dados mockada perfeitamente válida.
    Forçamos o tipo das colunas críticas para object para aceitarem a injeção 
    de textos de erro sem disparar travas do Pandas antes do ETL rodar.
    """
    dados = pd.DataFrame({
        'date': pd.date_range(start='2026-08-10', periods=25, freq='min'),
        'Ore Pulp pH': [10.0] * 25,
        'Ore Pulp Density': [1.68] * 25,
        'Starch Flow': [2000.0] * 25,
        'Amina Flow': [500.0] * 25,
        'Ore Pulp Flow': [400.0] * 25,
        '% Silica Concentrate': [1.9] * 25
    })
    
    # Preenche as demais colunas exigidas pelo contrato com zeros
    for col in COLUNAS_REQUISITADAS:
        if col not in dados.columns:
            dados[col] = 0.0
            
    # Converte para object para simular o comportamento de strings vindo do CSV bruto
    dados['Ore Pulp pH'] = dados['Ore Pulp pH'].astype(object)
    dados['Ore Pulp Density'] = dados['Ore Pulp Density'].astype(object)
            
    return dados


def test_robustez_dados_completamente_corrompidos(base_dados_valida):
    """
    TESTE DE ROBUSTEZ 1: Injeta textos aleatórios e corrompidos em colunas de sensores.
    O ETL deve converter os erros textuais para nulos (NaN) e tratá-los preenchendo 
    com a média de forma segura, convertendo a coluna final de volta para float numérico.
    """
    df_corrompido = base_dados_valida.copy()
    
    # alteração sem disparar erros precoces
    df_corrompido.loc[0, 'Ore Pulp pH'] = "FALHA_SENSOR_TEXTO"
    df_corrompido.loc[1, 'Ore Pulp Density'] = "CRITICAL_ERROR"
    
    etl = FlotationETL()
    df_resultado = etl.transform(df_corrompido)
    
    # os dados corrompidos devem ter virado números decimais puros
    assert df_resultado['Ore Pulp pH'].dtype == np.float64
    assert df_resultado['Ore Pulp Density'].dtype == np.float64
    assert not df_resultado['Ore Pulp pH'].isna().any()


def test_robustez_volume_insuficiente_de_dados():
    """
    TESTE DE ROBUSTEZ 2: O que acontece se a planta enviar menos de 20 linhas?
    Como o nosso código faz um .shift(periods=20), uma base com menos de 20 linhas 
    ficará 100% vazia após o dropna. O ETL deve lidar com isso de forma segura.
    """
    df_pequeno = pd.DataFrame({col: [0.0] * 5 for col in COLUNAS_REQUISITADAS})
    df_pequeno['date'] = pd.date_range(start='2026-08-10', periods=5, freq='min')
    
    etl = FlotationETL()
    df_resultado = etl.transform(df_pequeno)
    
    assert isinstance(df_resultado, pd.DataFrame)
    assert len(df_resultado) == 0

