# 🏭 Sensor Virtual de Flotação - Inteligência Artificial Industrial

Este projeto implementa um ecossistema completo de **Machine Learning e MLOps de nível corporativo** (Maturidade Nível 2) para prever o teor de **% Sílica no Concentrado Final** de uma planta de flotação mineral de ferro. O modelo atingiu espetaculares **91.37% de R²** na base de testes temporal futura.

## 🚀 Links Públicos do Ecossistema em Produção

O projeto foi totalmente conteinerizado e implantado em arquitetura de nuvem descentralizada e gratuita:

*   **🖥️ Dashboard de Negócio & ROI (Streamlit)**: [https://streamlit.app](https://streamlit.app)
    *   *Painel visual com gráficos interativos de variabilidade, cálculo de impacto econômico anual e simulador operacional "What-If" para a diretoria.*
*   **🧠 API de Inferência (FastAPI + Docker no Render)**: [https://onrender.com](https://onrender.com)
    *   *Interface do Swagger UI rodando em container isolado que executa previsões em milissegundos a partir dos dados em tempo real dos sensores.*
*   **📊 Governança e Model Registry (MLflow no DagsHub)**: [https://dagshub.com](https://dagshub.com)
    *   *Servidor central remoto de tracking que gerencia o ciclo de vida do modelo XGBoost, parâmetros, métricas e o versionamento estável no catálogo oficial.*

---

## 🛠️ Arquitetura do Projeto e Ciclo MLOps

O sistema foi desenhado seguindo rigorosamente os padrões de engenharia de software e ciclo de vida de dados (SDLC):

1.  **Engenharia de Dados (`src/data_processing.py`)**: Pipeline orientado a objetos que trata ruídos e dízimas decimais de sensores brutos, remove outliers via IQR e gera o **Target Lagging de 20 minutos** provado cientificamente por testes de hipóteses estatísticas.
2.  **Treinamento Robustecido (`src/train.py`)**: Modelo baseado no algoritmo **XGBoost Regressor** expandido (250 árvores, profundidade 10) e treinado sob **Validação Temporal Rígida** (80% passado para treino, 20% futuro isolado para teste), eliminando qualquer risco de *Data Leakage* ou Overfitting.
3.  **Métricas Alcançadas**:
    *   **$R^2$ (Poder de explicação)**: **91.37%** de acerto no futuro da planta.
    *   **MAE (Erro Médio Absoluto)**: **0.1499%**. O sensor erra, em média, apenas 0.15% em relação às análises reais do laboratório químico.
4.  **Qualidade de Código (`src/tests/`)**: Suite de testes automatizados com **PyTest** cobrindo testes unitários de robustez do ETL e testes de integração com injeção de Mocks via `monkeypatch` na API, mantendo a esteira de **CI/CD (GitHub Actions) 100% verde ✅**.

---

## 💼 Impacto de Negócio e ROI Industrial

O Sensor Virtual atua como uma ferramenta de suporte à decisão em tempo real na sala de controle, permitindo ao operador agir com 20 minutos de antecedência. O impacto financeiro do sistema inclui:
*   **Redução de Penalizações Comerciais**: O modelo antecipa picos de impureza, permitindo corrigir o processo e evitar até 85% dos lotes fora da especificação contratual.
*   **Otimização de Reagentes**: Evita o superdosamento (*over-dosing*) de Amina e Amido na flotação, gerando uma economia estimada de **4.0% no consumo de reagentes**, reduzindo drasticamente o OPEX da planta.

---

## 📂 Como Executar o Projeto Localmente

### 1. Clonar o Repositório e Ativar o Ambiente Virtual
```bash
git clone https://github.com/Wilson-Francisco/projeto-flotacao-ml
cd projeto-flotacao-ml
python -m venv .venv
source .venv/bin/activate  # No Windows use: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Executar a Suite de Testes do CI/CD
```bash
python -m pytest -v
```

### 3. Subir o Servidor Local da API (FastAPI)
```bash
python -m uvicorn app.main:app --reload
```

### 4. Subir o Dashboard de ROI (Streamlit)
```bash
python -m streamlit run app_dashboard.py
```
