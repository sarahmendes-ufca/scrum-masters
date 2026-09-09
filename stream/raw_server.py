#!/usr/bin/env python3
"""
stream/raw_server.py — Preview MJPEG bruto, sem inferência YOLO.

Dedicado à captura de dataset: serve o feed cru da câmera (sem
bounding boxes nem OSD) para você acompanhar ao vivo no navegador
enquanto o capture_frames.py salva os frames em disco.

Expõe dois endpoints:
  /stream    → MJPEG contínuo, para visualização no navegador
  /snapshot  → um único frame JPEG por requisição, sem buffer
               acumulado (usado pelo capture_frames.py --manual)

Uso: python3 stream/raw_server.py --device 0 --port 5001
"""
import argparse
import subprocess
import threading
import time

from flask import Flask, Response

app = Flask(__name__)
_lock = threading.Lock()
_latest_jpg: bytes = b""


def _capture_loop(device: int, width: int, height: int, fps: int):
    """Lê frames JPEG direto do rpicam-vid e mantém só o mais recente."""
    global _latest_jpg
    cmd = [
        "rpicam-vid", "-t", "0", "-n", "--codec", "mjpeg",
        "--camera", str(device),
        "--width", str(width), "--height", str(height),
        "--framerate", str(fps),
        "-o", "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    buf = b""
    while True:
        chunk = proc.stdout.read(4096)
        if not chunk:
            break
        buf += chunk
        end = buf.find(b"\xff\xd9")
        if end == -1:
            continue
        start = buf.rfind(b"\xff\xd8", 0, end)
        if start == -1:
            continue
        with _lock:
            _latest_jpg = buf[start:end + 2]
        buf = buf[end + 2:]


def _generate_mjpeg():
    while True:
        with _lock:
            jpg = _latest_jpg
        if not jpg:
            time.sleep(0.01)
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")


@app.route('/')
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head><title>Preview bruto — captura de dataset</title></head>
    <style>
        body { background:#111; color:#eee; font-family:sans-serif;
               display:flex; flex-direction:column; align-items:center;
               padding-top:30px; }
        img { border: 2px solid #444; border-radius: 4px; max-width:100%; }
    </style>
    </head>
    <body>
    <h1>Preview bruto (sem YOLO)</h1>
    <img src='/stream' />
    <p>Rode o capture_frames.py em outro terminal para salvar os frames.</p>
    </body>
    </html>
    """


@app.route('/stream')
def stream():
    return Response(_generate_mjpeg(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/snapshot')
def snapshot():
    """Devolve um único frame JPEG — sem stream, sem buffer acumulado."""
    with _lock:
        jpg = _latest_jpg
    if not jpg:
        return Response(status=503)
    return Response(jpg, mimetype='image/jpeg')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--port", type=int, default=5001)
    p.add_argument("--host", type=str, default="0.0.0.0")
    return p.parse_args()


def main():
    args = parse_args()
    t = threading.Thread(
        target=_capture_loop,
        args=(args.device, args.width, args.height, args.fps),
        daemon=True,
    )
    t.start()
    time.sleep(1.0)
    print("[INFO] Preview bruto iniciado (sem YOLO).")
    print(f"[INFO] Acesse no navegador: http://<IP_DO_RASPBERRY>:{args.port}/")
    print(f"[INFO] Snapshot (usado pelo --manual): http://<IP_DO_RASPBERRY>:{args.port}/snapshot")
    app.run(host=args.host, port=args.port, debug=False, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
