# Imagem base oficial do Python (versão slim para ser leve e segura)
FROM python:3.10-slim

# Define variáveis de ambiente para otimizar o Python dentro do container
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Define a pasta de trabalho dentro do container
WORKDIR /app

#  Instala dependências do sistema operacional necessárias para o XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copia primeiro apenas o arquivo de dependências (Aproveita o cache do Docker)
COPY requirements.txt .

#  Atualiza o pip e instala as dependências do Python de forma limpa
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copia APENAS as pastas produtivas de código (Sem mlruns local!)
COPY app/ ./app/
COPY src/ ./src/

# Cria um usuário comum para rodar a aplicação (Boa prática de segurança)
RUN useradd -m mineracao_user && chown -R mineracao_user:mineracao_user /app
USER mineracao_user

#  Expõe a porta em que o FastAPI vai rodar
EXPOSE 8000

# Comando definitivo para iniciar a API usando o Uvicorn quando o container ligar
CMD ["sh", "-c", "python -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
