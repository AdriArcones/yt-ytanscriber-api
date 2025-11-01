FROM python:3.11-slim

# ffmpeg es necesario para procesar audio; yt-dlp para descargar
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Carpeta de modelos en volumen para no re-descargar
RUN mkdir -p /models
ENV FASTER_WHISPER_CACHE_DIR=/models

COPY app.py /app/app.py

EXPOSE 8000
# Producción: workers 1-2 en CPU; ajusta según recursos.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
