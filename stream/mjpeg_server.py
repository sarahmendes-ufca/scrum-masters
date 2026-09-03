#!/usr/bin/env python3
"""
stream/mjpeg_server.py — Serve o stream YOLO anotado como MJPEG via HTTP.
Acesse no navegador: http://<IP_DO_RASPBERRY>:5000/stream
Execução: python3 stream/mjpeg_server.py --device 0 --port 5000
"""

import argparse
import threading
import time
import sys
from pathlib import Path

import cv2
from flask import Flask, Response

sys.path.insert(0, str(Path(__file__).parent.parent))
from stream.v3_optimized import OptimizedCamera, RealtimeDetector


# ── Estado global do servidor ────────────────────────────────
app = Flask(__name__)
_camera = None
_detector = None
_lock = threading.Lock()
_latest_jpg: bytes = b""  # último frame JPEG comprimido


def _frame_producer():
    """Thread que captura, processa e comprime frames continuamente."""
    global _latest_jpg
    while True:
        frame = _camera.read(timeout=2.0)
        if frame is None:
            continue

        annotated = _detector.process(frame)

        # Comprime para JPEG (qualidade 80 = bom equilíbrio tamanho/qualidade)
        ok, jpg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with _lock:
                _latest_jpg = jpg.tobytes()


def _generate_mjpeg():
    """Generator que produz o stream multipart para o cliente HTTP."""
    while True:
        with _lock:
            jpg = _latest_jpg

        if not jpg:
            time.sleep(0.01)
            continue

        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")

        # Limita a taxa de envio (desnecessário aumentar além do FPS real)
        time.sleep(0.033)  # ~30 FPS máximo de envio


@app.route("/")
def index():
    """Página HTML simples com o stream embutido."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset='UTF-8'>
        <title>YOLO — Stream em Tempo Real</title>
        <style>
            body { background:#111; color:#eee; font-family:sans-serif;
                   display:flex; flex-direction:column; align-items:center;
                   padding-top:30px; }
            h1 { margin-bottom: 12px; font-size: 1.4rem; }
            img { border: 2px solid #444; border-radius: 4px; max-width:100%; }
            p  { color: #888; font-size: 0.85rem; margin-top: 10px; }
        </style>
    </head>
    <body>
        <h1>YOLOv8 — Raspberry Pi 5 — Tempo Real</h1>
        <img src='/stream' />
        <p>Stream MJPEG com inferência YOLO e anotações em tempo real.</p>
    </body>
    </html>
    """


@app.route("/stream")
def stream():
    """Endpoint MJPEG: resposta multipart contínua."""
    return Response(
        _generate_mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/health")
def health():
    """Health check compatível com o padrão da API de inferência."""
    import json

    status = {
        "status": "ok",
        "stream": "active",
        "frame_count": _detector._frame_idx if _detector else 0,
    }
    return Response(json.dumps(status), mimetype="application/json")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--model", type=str, default="models/yolov8n.pt")
    p.add_argument("--conf", type=float, default=0.4)
    p.add_argument("--infer-every", type=int, default=3)
    p.add_argument("--infer-size", type=int, default=320)
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--host", type=str, default="0.0.0.0")
    return p.parse_args()


def main():
    global _camera, _detector
    args = parse_args()

    _camera = OptimizedCamera(args.device, args.width, args.height, args.fps).start()

    _detector = RealtimeDetector(
        args.model, args.conf, args.infer_every, args.infer_size
    )

    # Thread de processamento de frames
    producer = threading.Thread(
        target=_frame_producer, daemon=True, name="FrameProducer"
    )
    producer.start()
    time.sleep(1.0)  # aguarda primeiro frame

    print(f"[INFO] Servidor MJPEG iniciado.")
    print(f"[INFO] Acesse no navegador: http://<IP_DO_RASPBERRY>:{args.port}/")
    print(f"[INFO] Stream direto:        http://<IP_DO_RASPBERRY>:{args.port}/stream")
    print(f"[INFO] Health check:         http://<IP_DO_RASPBERRY>:{args.port}/health")

    app.run(
        host=args.host, port=args.port, debug=False, threaded=True, use_reloader=False
    )


if __name__ == "__main__":
    main()
