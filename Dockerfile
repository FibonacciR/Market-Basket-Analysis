# Multi-stage build para producción con ML preinstalado
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime as base

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar requirements e instalar dependencias Python
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Instalar gunicorn para producción
RUN pip install gunicorn

# Copiar código fuente
COPY src/ ./src/
RUN mkdir -p ./data
RUN mkdir -p ./models

# Crear usuario no-root para seguridad
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Exponer puerto
EXPOSE 8000

# Arranque simple con uvicorn para debug
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]