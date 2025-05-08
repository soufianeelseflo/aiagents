# /Dockerfile: PRODUCTION VERSION
# --------------------------------------------------------------------------------
    FROM python:3.11-slim
    ENV PYTHONDONTWRITEBYTECODE=1
    ENV PYTHONUNBUFFERED=1
    ENV APP_HOME=/app
    WORKDIR ${APP_HOME}
    RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*
    COPY requirements.txt .
    RUN pip install --no-cache-dir --upgrade pip && \
        pip install --no-cache-dir -r requirements.txt --break-system-packages
    RUN playwright install --with-deps chromium
    COPY . .
    EXPOSE 8080
    CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
    # --------------------------------------------------------------------------------