import asyncio
import base64
import io
import logging
import subprocess
import time

import cv2
import httpx
import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import HTMLResponse, StreamingResponse
from model import get_default_model_name, load_model
from PIL import Image
from schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    Detection,
    HealthResponse,
    MetricsResponse,
    PredictRequest,
    PredictResponse,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="YOLO Inference API",
    description="API REST para inferência com YOLOv8 e Câmera no Raspberry Pi 5",
    version="1.1.0",
)

_metrics = {
    "total": 0,
    "success": 0,
    "total_ms": 0.0,
}
_streaming_lock = asyncio.Lock()


def _decode_image(image_base64: str) -> np.ndarray:
    raw = base64.b64decode(image_base64)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return np.array(img)


def _load_image_from_request(request: PredictRequest) -> np.ndarray:
    if not request.image_base64 and not request.image_url:
        raise HTTPException(
            status_code=422,
            detail="Forneça image_base64 ou image_url.",
        )

    if request.image_base64:
        return _decode_image(request.image_base64)

    resp = httpx.get(
        request.image_url,
        timeout=15.0,
        follow_redirects=True,
    )
    resp.raise_for_status()

    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    return np.array(img)


def _capture_frame_from_camera(device_id: int = 0) -> np.ndarray:
    """
    Captura frame via rpicam-still (Câmera CSI)
    ou OpenCV (Câmera USB).
    """

    # 1. Tenta captura nativa CSI da Raspberry Pi
    # (rpicam-still / libcamera-still)
    for cmd_tool in ["rpicam-still", "libcamera-still"]:
        try:
            cmd = [
                cmd_tool,
                "-t",
                "500",  # 500ms para ajuste de exposição e balanço de branco
                "-n",   # Sem janela de preview
                "-o",
                "-",    # Saída direta em memória (stdout)
                "--width",
                "640",
                "--height",
                "480",
                "-e",
                "jpg",
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=5,
                check=False,
            )

            if result.returncode == 0 and len(result.stdout) > 0:
                img = Image.open(
                    io.BytesIO(result.stdout)
                ).convert("RGB")

                return np.array(img)

        except (OSError, subprocess.SubprocessError) as e:
            logger.warning(
                "Falha ao capturar via %s: %s", cmd_tool, e
            )

    # 2. Fallback para câmeras USB padrão (V4L2)
    cap = cv2.VideoCapture(device_id)

    if cap.isOpened():
        try:
            for _ in range(3):
                cap.read()

            ret, frame_bgr = cap.read()

            if ret and frame_bgr is not None:
                return cv2.cvtColor(
                    frame_bgr,
                    cv2.COLOR_BGR2RGB,
                )
        finally:
            cap.release()

    raise HTTPException(
        status_code=500,
        detail=(
            "Falha ao capturar imagem da câmera. "
            "Verifique a conexão do cabo flat."
        ),
    )


def _run_inference(
    image_np: np.ndarray,
    model_name: str,
    confidence: float,
) -> PredictResponse:
    model = load_model(model_name)

    t0 = time.perf_counter()

    results = model(
        image_np,
        conf=confidence,
        verbose=False,
    )

    elapsed_ms = (time.perf_counter() - t0) * 1000

    detections = []

    for r in results:
        for box in r.boxes:
            coords = box.xyxy[0].tolist()
            cls_id = int(box.cls[0].item())
            conf_val = float(box.conf[0].item())

            detections.append(
                Detection(
                    label=model.names[cls_id],
                    confidence=round(conf_val, 4),
                    bbox=[
                        round(float(c), 2)
                        for c in coords
                    ],
                )
            )

    h, w = image_np.shape[:2]

    return PredictResponse(
        detections=detections,
        inference_ms=round(elapsed_ms, 2),
        model_used=model_name,
        image_width=w,
        image_height=h,
    )


# ── Endpoints Originais ─────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    model_name = get_default_model_name()

    try:
        load_model(model_name)
        loaded = True
    except (OSError, RuntimeError, ValueError) as e:
        logger.warning("Falha ao carregar modelo %s: %s", model_name, e)
        loaded = False

    return HealthResponse(
        status="ok",
        model_loaded=loaded,
        model_name=model_name,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    _metrics["total"] += 1

    try:
        img = _load_image_from_request(request)

        result = _run_inference(
            img,
            request.model_name,
            request.confidence,
        )

        _metrics["success"] += 1
        _metrics["total_ms"] += result.inference_ms

        return result

    except HTTPException:
        raise

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e),
        ) from e

    except Exception as e:
        logger.exception("Erro inesperado em /predict")
        raise HTTPException(
            status_code=500,
            detail=str(e),
        ) from e


@app.post(
    "/predict/image",
    responses={
        200: {
            "content": {
                "image/jpeg": {}
            }
        }
    },
)
def predict_image(request: PredictRequest):
    """
    Executa inferência em imagem enviada
    e retorna JPEG com caixas delimitadoras.
    """

    _metrics["total"] += 1

    try:
        img_rgb = _load_image_from_request(request)

        model = load_model(request.model_name)

        t0 = time.perf_counter()

        results = model(
            img_rgb,
            conf=request.confidence,
            verbose=False,
        )

        elapsed_ms = (
            time.perf_counter() - t0
        ) * 1000

        _metrics["success"] += 1
        _metrics["total_ms"] += elapsed_ms

        annotated_array = results[0].plot()
        annotated_pil = Image.fromarray(
            annotated_array
        )

        buffer = io.BytesIO()

        annotated_pil.save(
            buffer,
            format="JPEG",
            quality=95,
        )

        return Response(
            content=buffer.getvalue(),
            media_type="image/jpeg",
        )

    except HTTPException:
        raise

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e),
        ) from e

    except Exception as e:
        logger.exception("Erro inesperado em /predict/image")
        raise HTTPException(
            status_code=500,
            detail=str(e),
        ) from e


# ── Novos Endpoints: Integração com a Câmera ────────────────

@app.post(
    "/predict/camera",
    response_model=PredictResponse,
)
def predict_from_camera(
    device_id: int = Query(
        0,
        description="Índice do dispositivo (/dev/videoX)",
    ),
    confidence: float = Query(
        0.25,
        ge=0.0,
        le=1.0,
        description="Limiar de confiança",
    ),
    model_name: str = Query(
        "yolov8n.pt",
        description="Modelo YOLO a ser utilizado",
    ),
):
    """
    Dispara a captura de uma foto pela câmera,
    infere e retorna as detecções em JSON.
    """

    _metrics["total"] += 1

    try:
        img_rgb = _capture_frame_from_camera(
            device_id=device_id
        )

        result = _run_inference(
            img_rgb,
            model_name,
            confidence,
        )

        _metrics["success"] += 1
        _metrics["total_ms"] += result.inference_ms

        return result

    except HTTPException:
        raise

    except Exception as e:
        logger.exception("Erro inesperado em /predict/camera")
        raise HTTPException(
            status_code=500,
            detail=str(e),
        ) from e


@app.get(
    "/predict/camera/image",
    responses={
        200: {
            "content": {
                "image/jpeg": {}
            }
        }
    },
)
def predict_from_camera_image(
    device_id: int = Query(
        0,
        description="Índice do dispositivo (/dev/videoX)",
    ),
    confidence: float = Query(
        0.25,
        ge=0.0,
        le=1.0,
        description="Limiar de confiança",
    ),
    model_name: str = Query(
        "yolov8n.pt",
        description="Modelo YOLO a ser utilizado",
    ),
):
    """
    Dispara a câmera, infere e renderiza
    a foto real capturada com os bounding
    boxes na Swagger UI.
    """

    _metrics["total"] += 1

    try:
        img_rgb = _capture_frame_from_camera(
            device_id=device_id
        )

        model = load_model(model_name)

        t0 = time.perf_counter()

        results = model(
            img_rgb,
            conf=confidence,
            verbose=False,
        )

        elapsed_ms = (
            time.perf_counter() - t0
        ) * 1000

        _metrics["success"] += 1
        _metrics["total_ms"] += elapsed_ms

        annotated_array = results[0].plot()

        annotated_pil = Image.fromarray(
            annotated_array
        )

        buffer = io.BytesIO()

        annotated_pil.save(
            buffer,
            format="JPEG",
            quality=95,
        )

        return Response(
            content=buffer.getvalue(),
            media_type="image/jpeg",
        )

    except HTTPException:
        raise

    except Exception as e:
        logger.exception("Erro inesperado em /predict/camera/image")
        raise HTTPException(
            status_code=500,
            detail=str(e),
        ) from e


# ── Endpoints Batch e Métricas ──────────────────────────────

@app.post(
    "/predict/batch",
    response_model=BatchPredictResponse,
)
def predict_batch(
    request: BatchPredictRequest,
):
    t_total = time.perf_counter()

    results = []

    for img_b64 in request.images_base64:
        img = _decode_image(img_b64)

        results.append(
            _run_inference(
                img,
                request.model_name,
                request.confidence,
            )
        )

    total_ms = (
        time.perf_counter() - t_total
    ) * 1000

    return BatchPredictResponse(
        results=results,
        total_inference_ms=round(
            total_ms,
            2,
        ),
    )


@app.get(
    "/metrics",
    response_model=MetricsResponse,
)
async def get_metrics():
    avg = (
        _metrics["total_ms"]
        / _metrics["success"]
        if _metrics["success"] > 0
        else 0.0
    )

    return MetricsResponse(
        total_requests=_metrics["total"],
        successful_requests=_metrics["success"],
        avg_inference_ms=round(avg, 2),
    )

# ── Streaming de Vídeo (MJPEG) ──────────────────────────────

@app.get("/stream/camera")
async def stream_camera(
    request: Request,
    confidence: float = Query(
        0.25,
        ge=0.0,
        le=1.0,
        description="Limiar de confiança",
    ),
    model_name: str = Query(
        "yolov8n.pt",
        description="Modelo YOLO a ser utilizado",
    ),
    framerate: int = Query(
        15,
        ge=1,
        le=30,
        description="FPS de captura solicitados ao sensor",
    ),
):
    """
    Transmite vídeo contínuo da câmera com detecções YOLO
    sobrepostas em cada frame.
    """
    if _streaming_lock.locked():
        raise HTTPException(
            status_code=409,
            detail=(
                "Já existe um stream de câmera em andamento. "
                "Feche a aba atual antes de abrir outra."
            ),
        )

    model = load_model(model_name)

    async def frame_generator():
        cmd = [
            "rpicam-vid",
            "-t", "0",
            "-n",
            "--codec", "mjpeg",
            "--quality", "80",
            "--width", "640",
            "--height", "480",
            "--framerate", str(framerate),
            "-o", "-",
        ]

        loop = asyncio.get_event_loop()

        proc = await loop.run_in_executor(
            None,
            lambda: subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            ),
        )

        print(
            f"[stream/camera] rpicam-vid iniciado (pid={proc.pid})",
            flush=True,
        )

        async with _streaming_lock:
            try:
                buffer = b""

                while True:
                    if await request.is_disconnected():
                        break

                    try:
                        chunk = await asyncio.wait_for(
                            loop.run_in_executor(
                                None,
                                proc.stdout.read,
                                4096,
                            ),
                            timeout=5.0,
                        )
                    except asyncio.TimeoutError:
                        print(
                            "[stream/camera] Nenhum frame em 5s — encerrando "
                            "(câmera pode estar ocupada ou não detectada).",
                            flush=True,
                        )
                        break

                    if not chunk:
                        break

                    buffer += chunk

                    # Extrai todos os frames JPEG completos disponíveis.
                    while True:
                        start_idx = buffer.find(b"\xff\xd8")
                        end_idx = (
                            buffer.find(b"\xff\xd9", start_idx + 2)
                            if start_idx != -1
                            else -1
                        )

                        if start_idx == -1 or end_idx == -1:
                            break

                        raw_frame = buffer[start_idx:end_idx + 2]
                        buffer = buffer[end_idx + 2:]

                        img = Image.open(
                            io.BytesIO(raw_frame)
                        ).convert("RGB")

                        img_np = np.array(img)

                        results = model(
                            img_np,
                            conf=confidence,
                            verbose=False,
                        )

                        annotated = results[0].plot()
                        annotated_pil = Image.fromarray(annotated)

                        out_buffer = io.BytesIO()

                        annotated_pil.save(
                            out_buffer,
                            format="JPEG",
                            quality=85,
                        )

                        jpeg_bytes = out_buffer.getvalue()

                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n\r\n"
                            + jpeg_bytes
                            + b"\r\n"
                        )

            finally:
                proc.terminate()

                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2)

                if proc.stderr:
                    stderr_output = (
                        proc.stderr.read()
                        .decode(errors="ignore")
                        .strip()
                    )

                    if stderr_output:
                        print(
                            f"[stream/camera] rpicam-vid stderr:\n"
                            f"{stderr_output}",
                            flush=True,
                        )

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/stream/view", response_class=HTMLResponse)
async def stream_view():
    """Página simples para visualizar o stream anotado no navegador."""
    return """
<!DOCTYPE html>
<html>
<head>
    <title>YOLO Live Stream — Raspberry Pi 5</title>
</head>
<body style="margin:0; background:#111; display:flex; justify-content:center; align-items:center; height:100vh;">
    <img src="/stream/camera" style="max-width:100%; height:auto;" />
</body>
</html>
"""
