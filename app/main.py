import asyncio
import base64
import io
import json
import subprocess
import time
import uuid

import cv2
import httpx
import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import HTMLResponse, StreamingResponse
from PIL import Image

from preprocessing.preprocessor import CONFIG_DEFAULT, Preprocessor

from .model import get_default_model_name, load_model
from .schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    Detection,
    HealthResponse,
    MetricsResponse,
    PredictRequest,
    PredictResponse,
)


def log_event(event: str, level: str = "INFO", **kwargs):
    """Emite um evento estruturado em JSON para stdout."""
    import time
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "level":     level,
        "event":     event,
        **kwargs,
    }
    print(json.dumps(record, ensure_ascii=False), flush=True)
_streaming_lock = asyncio.Lock()

app = FastAPI(
    title="YOLO Inference API",
    description="API REST para inferencia com YOLOv8 e Camera no Raspberry Pi 5",
    version="1.1.0",
)

_metrics = {"total": 0, "success": 0, "total_ms": 0.0}

_preprocessor = Preprocessor(CONFIG_DEFAULT)

def _decode_image(image_base64: str) -> np.ndarray:
    raw = base64.b64decode(image_base64)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return np.array(img)


def _load_image_from_request(request: PredictRequest) -> np.ndarray:
    if not request.image_base64 and not request.image_url:
        raise HTTPException(
            status_code=422, detail="Forneca image_base64 ou image_url."
        )
    if request.image_base64:
        return _decode_image(request.image_base64)
    else:
        resp = httpx.get(
            request.image_url, timeout=15.0, follow_redirects=True
        )
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        return np.array(img)


def _capture_frame_from_camera(device_id: int = 0) -> np.ndarray:
    """Captura frame via rpicam-still ou OpenCV."""
    for cmd_tool in ["rpicam-still", "libcamera-still"]:
        try:
            cmd = [
                cmd_tool,
                "-t",
                "500",
                "-n",
                "-o",
                "-",
                "--width",
                "640",
                "--height",
                "480",
                "-e",
                "jpg",
            ]
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=5,
            )
            if result.returncode == 0 and len(result.stdout) > 0:
                img = Image.open(io.BytesIO(result.stdout)).convert("RGB")
                return np.array(img)
        except Exception:
            pass
    cap = cv2.VideoCapture(device_id)
    if cap.isOpened():
        try:
            for _ in range(3):
                cap.read()
            ret, frame_bgr = cap.read()
            if ret and frame_bgr is not None:
                return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        finally:
            cap.release()
    raise HTTPException(
        status_code=500,
        detail="Falha ao capturar imagem da camera. Verifique a conexao do cabo flat.",
    )


def _run_inference(
    image_np: np.ndarray, model_name: str, confidence: float) -> PredictResponse:
    model = load_model(model_name)

    # Pré-processamento explícito
    # image_np chega em RGB (já convertido em _decode_image) --
    # o Preprocessor espera BGR, então converte temporariamente
    frame_bgr = image_np[:, :, ::-1]
    preproc_res = _preprocessor.process(frame_bgr)
    frame_ready = preproc_res.frame  # RGB, letterboxed

    t0 = time.perf_counter()
    results = model(frame_ready, conf=confidence, verbose=False)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    detections = []
    for r in results:
        for box in r.boxes:
            # Ajusta as coordenadas do espaço letterboxed de volta ao
            # espaço da imagem original -- sem isso, os bboxes retornados
            # pela API ficam deslocados sempre que houver padding
            bbox_lb = box.xyxy[0].numpy().reshape(1, 4)
            bbox_orig = _preprocessor.adjust_boxes(bbox_lb, preproc_res)[0]
            cls_id = int(box.cls[0].item())
            conf_val = float(box.conf[0].item())

            detections.append(
                Detection(
                    label=model.names[cls_id],
                    confidence=round(conf_val, 4),
                    bbox=[round(float(c), 2) for c in bbox_orig],
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

@app.get("/health", response_model=HealthResponse)
async def health_check():
    model_name = get_default_model_name()
    try:
        load_model(model_name)
        loaded = True
    except Exception:
        loaded = False
    return HealthResponse(
        status="ok", model_loaded=loaded, model_name=model_name
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    request_id = str(uuid.uuid4())[:8]
    _metrics["total"] += 1

    log_event("predict_start",
              request_id=request_id,
              model=request.model_name,
              confidence=request.confidence)

    if not request.image_base64 and not request.image_url:
        log_event("predict_error", level="WARN",
                  request_id=request_id, reason="missing_input")
        raise HTTPException(status_code=422,
            detail="Forneça image_base64 ou image_url.")
    try:
        if request.image_base64:
            img = _decode_image(request.image_base64)
        else:
            import httpx
            resp = httpx.get(request.image_url, timeout=10)
            resp.raise_for_status()
            img = _decode_image(base64.b64encode(resp.content).decode())

        result = _run_inference(img, request.model_name, request.confidence)
        _metrics["success"] += 1
        _metrics["total_ms"] += result.inference_ms

        log_event("predict_complete",
                  request_id=request_id,
                  model=result.model_used,
                  detections=len(result.detections),
                  inference_ms=result.inference_ms,
                  image_size=f"{result.image_width}x{result.image_height}")
        return result

    except FileNotFoundError as e:
        log_event("predict_error", level="ERROR",
                  request_id=request_id, reason=str(e))
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        log_event("predict_error", level="ERROR",
                  request_id=request_id, reason=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post(
    "/predict/image",
    responses={200: {"content": {"image/jpeg": {}}}},
)
def predict_image(request: PredictRequest):
    _metrics["total"] += 1
    try:
        img_rgb = _load_image_from_request(request)
        model = load_model(request.model_name)
        t0 = time.perf_counter()
        results = model(
            img_rgb, conf=request.confidence, verbose=False
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        _metrics["success"] += 1
        _metrics["total_ms"] += elapsed_ms
        annotated_array = results[0].plot()
        annotated_pil = Image.fromarray(annotated_array)
        buffer = io.BytesIO()
        annotated_pil.save(buffer, format="JPEG", quality=95)
        return Response(
            content=buffer.getvalue(), media_type="image/jpeg"
        )
    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/camera", response_model=PredictResponse)
def predict_from_camera(
    device_id: int = Query(
        0, description="Indice do dispositivo (/dev/videoX)"
    ),
    confidence: float = Query(
        0.25, ge=0.0, le=1.0, description="Limiar de confianca"
    ),
    model_name: str = Query(
        "yolov8n.pt", description="Modelo YOLO a ser utilizado"
    ),
):
    _metrics["total"] += 1
    try:
        img_rgb = _capture_frame_from_camera(device_id=device_id)
        result = _run_inference(img_rgb, model_name, confidence)
        _metrics["success"] += 1
        _metrics["total_ms"] += result.inference_ms
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/predict/camera/image",
    responses={200: {"content": {"image/jpeg": {}}}},
)
def predict_from_camera_image(
    device_id: int = Query(
        0, description="Indice do dispositivo (/dev/videoX)"
    ),
    confidence: float = Query(
        0.25, ge=0.0, le=1.0, description="Limiar de confianca"
    ),
    model_name: str = Query(
        "yolov8n.pt", description="Modelo YOLO a ser utilizado"
    ),
):
    _metrics["total"] += 1
    try:
        img_rgb = _capture_frame_from_camera(device_id=device_id)
        model = load_model(model_name)
        t0 = time.perf_counter()
        results = model(img_rgb, conf=confidence, verbose=False)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        _metrics["success"] += 1
        _metrics["total_ms"] += elapsed_ms
        annotated_array = results[0].plot()
        annotated_pil = Image.fromarray(annotated_array)
        buffer = io.BytesIO()
        annotated_pil.save(buffer, format="JPEG", quality=95)
        return Response(
            content=buffer.getvalue(), media_type="image/jpeg"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(request: BatchPredictRequest):
    t_total = time.perf_counter()
    results = []
    for img_b64 in request.images_base64:
        img = _decode_image(img_b64)
        results.append(
            _run_inference(img, request.model_name, request.confidence)
        )
    total_ms = (time.perf_counter() - t_total) * 1000
    return BatchPredictResponse(
        results=results, total_inference_ms=round(total_ms, 2)
    )


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    avg = (
        _metrics["total_ms"] / _metrics["success"]
        if _metrics["success"] > 0
        else 0.0
    )
    return MetricsResponse(
        total_requests=_metrics["total"],
        successful_requests=_metrics["success"],
        avg_inference_ms=round(avg, 2),
    )


@app.get("/stream/camera")
async def stream_camera(
    request: Request,
    confidence: float = Query(
        0.25, ge=0.0, le=1.0, description="Limiar de confianca"
    ),
    model_name: str = Query(
        "yolov8n.pt", description="Modelo YOLO a ser utilizado"
    ),
    framerate: int = Query(
        15, ge=1, le=30, description="FPS de captura solicitados ao sensor"
    ),
):
    if _streaming_lock.locked():
        raise HTTPException(
            status_code=409,
            detail="Ja existe um stream de camera em andamento.",
        )
    model = load_model(model_name)

    async def frame_generator():
        cmd = [
            "rpicam-vid",
            "-t",
            "0",
            "-n",
            "--codec",
            "mjpeg",
            "--quality",
            "80",
            "--width",
            "640",
            "--height",
            "480",
            "--framerate",
            str(framerate),
            "-o",
            "-",
        ]
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        loop = asyncio.get_event_loop()
        async with _streaming_lock:
            try:
                buffer = b""
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        chunk = await asyncio.wait_for(
                            loop.run_in_executor(
                                None, proc.stdout.read, 4096
                            ),
                            timeout=5.0,
                        )
                    except asyncio.TimeoutError:
                        break
                    if not chunk:
                        break
                    buffer += chunk
                    while True:
                        start = buffer.find(b"\xff\xd8")
                        end = (
                            buffer.find(b"\xff\xd9", start + 2)
                            if start != -1
                            else -1
                        )
                        if start == -1 or end == -1:
                            break
                        raw_frame = buffer[start : end + 2]
                        buffer = buffer[end + 2 :]
                        img = Image.open(io.BytesIO(raw_frame)).convert(
                            "RGB"
                        )
                        img_np = np.array(img)
                        results = model(
                            img_np, conf=confidence, verbose=False
                        )
                        annotated = results[0].plot()
                        annotated_pil = Image.fromarray(annotated)
                        out_buffer = io.BytesIO()
                        annotated_pil.save(
                            out_buffer, format="JPEG", quality=85
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

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/stream/view", response_class=HTMLResponse)
async def stream_view():
    return """
    <!DOCTYPE html>
    <html> <head><title>YOLO Live Stream</title></head>
    <body style="margin:0; background:#111; display:flex; justify-content:center; align-items:center; height:100vh;">
    <img src="/stream/camera" style="max-width:100%; height:auto;" />
    </body>
    </html>
    """
