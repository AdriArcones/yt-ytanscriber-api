import os, tempfile, subprocess, time, uuid
from urllib.parse import urlparse
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from faster_whisper import WhisperModel

MODEL_NAME = os.getenv("MODEL_NAME", "base")        # base|small|medium|large-v3 ...
DEVICE = os.getenv("DEVICE", "cpu")                 # cpu|cuda
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "int8")    # cpu: int8/int8_float32; gpu: float16
DOWNLOAD_ROOT = os.getenv("FASTER_WHISPER_CACHE_DIR", "/models")

# ---- helper ----
def normalize_yt_url(u: str) -> str:
    """Convierte youtu.be y shorts a la URL canónica watch?v=..."""
    p = urlparse(u)
    host = p.netloc.lower()
    if host in ("youtu.be", "www.youtu.be"):
        vid = p.path.lstrip("/")
        return f"https://www.youtube.com/watch?v={vid}"
    if "youtube.com" in host and p.path.startswith("/shorts/"):
        vid = p.path.rstrip("/").split("/")[-1]
        return f"https://www.youtube.com/watch?v={vid}"
    return u
# ---------------

# Carga perezosa del modelo (mejor arranque en VPS modestos)
model = None
def get_model():
    global model
    if model is None:
        t0 = time.time()
        model = WhisperModel(
            MODEL_NAME,
            device=DEVICE,
            compute_type=COMPUTE_TYPE,
            download_root=DOWNLOAD_ROOT
        )
        print(f"[whisper] Loaded {MODEL_NAME} on {DEVICE} ({COMPUTE_TYPE}) in {time.time()-t0:.2f}s")
    return model

app = FastAPI(title="YouTube Transcriber API")

class TranscribeReq(BaseModel):
    url: str
    lang: str | None = None      # "es", "en", ...
    translate: bool = False      # True => traduce a inglés
    word_timestamps: bool = False

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/transcribe")
def transcribe(req: TranscribeReq):
    """
    Descarga el audio con yt-dlp y transcribe con Faster-Whisper.
    """
    t0 = time.time()
    with tempfile.TemporaryDirectory() as td:
        base = os.path.join(td, str(uuid.uuid4()))
        url = normalize_yt_url(req.url)

        # --- opciones yt-dlp / UA / cookies / proxy ---
        UA_WEB = os.getenv(
            "YDL_UA_WEB",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
        )
        UA_IOS = os.getenv(
            "YDL_UA_IOS",
            "Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Mobile/15E148 Safari/604.1"
        )
        COOKIES = os.getenv("YDL_COOKIES")   # p.ej. /app/cookies.txt (formato Netscape)
        PROXY = os.getenv("YDL_PROXY")       # p.ej. http://user:pass@host:puerto

        use_cookies = bool(COOKIES and os.path.exists(COOKIES))

        common = [
            "yt-dlp",
            "-f", "bestaudio/best",
            "--extract-audio",
            "--audio-format", "m4a",
            "--audio-quality", "0",
            "--no-playlist",
            "--force-ipv4",
            "--retry-sleep", "2",
            "--retries", "10",
            "-o", f"{base}.%(ext)s",
        ]

        if use_cookies:
            # Cookies del navegador => usar cliente WEB de escritorio
            ytdlp_cmd = common + [
                "--user-agent", UA_WEB,
                "--extractor-args", "youtube:player_client=web",
                "--cookies", COOKIES,
                url,
            ]
        else:
            # Sin cookies => usar cliente iOS (evita PO Token de Android)
            ytdlp_cmd = common + [
                "--user-agent", UA_IOS,
                "--extractor-args", "youtube:player_client=ios",
                url,
            ]

        if PROXY:
            ytdlp_cmd += ["--proxy", PROXY]
        # ------------------------------------------------

        try:
            subprocess.run(ytdlp_cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            err = e.stderr.decode(errors="ignore")
            raise HTTPException(status_code=400, detail=f"yt-dlp error: {err[:500]}")

        # Buscar el archivo generado
        audio_path = None
        for ext in ("m4a", "mp3", "wav", "opus", "webm"):
            candidate = f"{base}.{ext}"
            if os.path.exists(candidate):
                audio_path = candidate
                break
        if not audio_path:
            raise HTTPException(status_code=500, detail="No se encontró el archivo de audio generado.")

        # Transcribir
        mdl = get_model()
        segments, info = mdl.transcribe(
            audio_path,
            language=req.lang,
            task="translate" if req.translate else "transcribe",
            vad_filter=True,
            word_timestamps=req.word_timestamps,
        )

        seg_list = []
        for s in segments:
            item = {"start": round(s.start, 3), "end": round(s.end, 3), "text": s.text.strip()}
            if req.word_timestamps and getattr(s, "words", None):
                item["words"] = [
                    {"start": round(w.start, 3), "end": round(w.end, 3), "word": w.word}
                    for w in s.words
                ]
            seg_list.append(item)

        return {
            "duration_s": getattr(info, "duration", None),
            "detected_language": getattr(info, "language", None),
            "segments": seg_list,
            "text": " ".join(s["text"] for s in seg_list).strip(),
            "elapsed_s": round(time.time() - t0, 2),
            "model": MODEL_NAME,
        }

@app.post("/transcribe_file")
async def transcribe_file(
    file: UploadFile = File(...),
    lang: str | None = Form(default=None),
    translate: bool = Form(default=False),
    word_timestamps: bool = Form(default=False),
):
    """
    Transcribe un archivo de audio subido (útil si n8n descarga previamente).
    """
    t0 = time.time()
    with tempfile.TemporaryDirectory() as td:
        audio_path = os.path.join(td, file.filename)
        with open(audio_path, "wb") as f:
            f.write(await file.read())

        mdl = get_model()
        segments, info = mdl.transcribe(
            audio_path,
            language=lang,
            task="translate" if translate else "transcribe",
            vad_filter=True,
            word_timestamps=word_timestamps,
        )

        seg_list = []
        for s in segments:
            item = {"start": round(s.start, 3), "end": round(s.end, 3), "text": s.text.strip()}
            if word_timestamps and getattr(s, "words", None):
                item["words"] = [
                    {"start": round(w.start, 3), "end": round(w.end, 3), "word": w.word}
                    for w in s.words
                ]
            seg_list.append(item)

        return {
            "duration_s": getattr(info, "duration", None),
            "detected_language": getattr(info, "language", None),
            "segments": seg_list,
            "text": " ".join(s["text"] for s in seg_list).strip(),
            "elapsed_s": round(time.time() - t0, 2),
            "model": MODEL_NAME,
        }
