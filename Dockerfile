FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt /app/

# Dependencias del sistema mínimas para OpenCV headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libgl1 && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto
COPY . /app

EXPOSE 5002

# Puedes ajustar workers según memoria disponible
CMD ["gunicorn", "--workers", "1", "--threads", "2", "--bind", "0.0.0.0:5002", "app:app"]
