from pydantic import BaseModel, Field

class PayloadSensoresPlanta(BaseModel):
    """
    Contrato de dados rígido via Pydantic para validar os dados instantâneos
    dos sensores da planta de flotação enviados para predição.
    """
    # Características da Carga e Variáveis de Controle
    Iron_Feed: float = Field(..., alias="% Iron Feed")
    Starch_Flow: float = Field(..., alias="Starch Flow")
    Amina_Flow: float = Field(..., alias="Amina Flow")
    Ore_Pulp_Flow: float = Field(..., alias="Ore Pulp Flow")
    Ore_Pulp_Density: float = Field(..., alias="Ore Pulp Density")
    
    # Sensores de Fluxo de Ar das 7 Colunas de Flotação
    Flotation_Column_01_Air_Flow: float = Field(..., alias="Flotation Column 01 Air Flow")
    Flotation_Column_02_Air_Flow: float = Field(..., alias="Flotation Column 02 Air Flow")
    Flotation_Column_03_Air_Flow: float = Field(..., alias="Flotation Column 03 Air Flow")
    Flotation_Column_04_Air_Flow: float = Field(..., alias="Flotation Column 04 Air Flow")
    Flotation_Column_05_Air_Flow: float = Field(..., alias="Flotation Column 05 Air Flow")
    Flotation_Column_06_Air_Flow: float = Field(..., alias="Flotation Column 06 Air Flow")
    Flotation_Column_07_Air_Flow: float = Field(..., alias="Flotation Column 07 Air Flow")
    
    # Sensores de Nível das 7 Colunas de Flotação
    Flotation_Column_01_Level: float = Field(..., alias="Flotation Column 01 Level")
    Flotation_Column_02_Level: float = Field(..., alias="Flotation Column 02 Level")
    Flotation_Column_03_Level: float = Field(..., alias="Flotation Column 03 Level")
    Flotation_Column_04_Level: float = Field(..., alias="Flotation Column 04 Level")
    Flotation_Column_05_Level: float = Field(..., alias="Flotation Column 05 Level")
    Flotation_Column_06_Level: float = Field(..., alias="Flotation Column 06 Level")
    Flotation_Column_07_Level: float = Field(..., alias="Flotation Column 07 Level")
    
    # Engenharia de Recursos (Lags Temporais de 20 minutos provados na EDA)
    Starch_Flow_lag20: float = Field(..., alias="Starch Flow_lag20")
    Amina_Flow_lag20: float = Field(..., alias="Amina Flow_lag20")
    Ore_Pulp_Flow_lag20: float = Field(..., alias="Ore Pulp Flow_lag20")
    Ore_Pulp_Density_lag20: float = Field(..., alias="Ore Pulp Density_lag20")
    
    # A nossa variável principal de Target Lagging (Análise anterior do laboratório)
    Silica_Concentrate_lag20: float = Field(..., alias="% Silica Concentrate_lag20")

    class Config:
        # Permite que a API receba os nomes reais com espaços e caracteres especiais (aliases) do JSON
        populate_by_name = True
