#!/usr/bin/env python3
"""
stream/capture_frames.py
Captura frames do stream de câmera e os salva em dataset/raw/.
Reutiliza a lógica de captura do pipeline de streaming em tempo real..
Uso:
  python stream/capture_frames.py --source 0 --total 200 --interval 1.5
  python stream/capture_frames.py --source mjpeg --url http://localhost:5001/stream
"""
import argparse
import time
import urllib.request
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
import subprocess

OUTPUT_DIR = Path("dataset/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def fetch_snapshot(url: str):
    """Busca um único frame JPEG via HTTP — sem streaming, sem buffer acumulado."""
    with urllib.request.urlopen(url, timeout=5) as resp:
        data = resp.read()
    frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    return frame is not None, frame

def parse_args():
    p = argparse.ArgumentParser(description="Captura de frames para dataset")
    p.add_argument("--source",   default="0",
                   help="Índice da câmera (0, 1) ou 'mjpeg' para stream HTTP")
    p.add_argument("--url",      default="http://localhost:5001/stream",
                   help="URL do stream MJPEG (usado quando --source mjpeg)")
    p.add_argument("--total",    type=int, default=200,
                   help="Total de frames a capturar")
    p.add_argument("--interval", type=float, default=1.5,
                   help="Intervalo entre capturas em segundos")
    p.add_argument("--width",    type=int, default=640)
    p.add_argument("--height",   type=int, default=480)
    p.add_argument("--manual", action="store_true",
                   help="Captura manual: pressione ENTER a cada frame, em vez de intervalo automático")
    p.add_argument("--snapshot-url", default="http://localhost:5001/snapshot",
                   help="URL do endpoint /snapshot (usado apenas com --source mjpeg --manual)")
    return p.parse_args()

class RpicamCapture:
    """
    Captura frames da câmera CSI via rpicam-vid (MJPEG), com interface
    compatível com cv2.VideoCapture (.read(), .isOpened(), .release()).
    cv2.VideoCapture() não suporta libcamera (câmeras CSI no Raspberry Pi
    OS Bookworm/Trixie) — só funciona com V4L2 nativo (webcams USB).
    """
    def __init__(self, device: int, width: int, height: int, fps: int = 15):
        cmd = [
            "rpicam-vid", "-t", "0", "-n", "--codec", "mjpeg",
            "--camera", str(device),
            "--width", str(width), "--height", str(height),
            "--framerate", str(fps),
            "-o", "-",
        ]
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        self._buf = b""

    def read(self):
        while True:
            start = self._buf.find(b"\xff\xd8")
            end = self._buf.find(b"\xff\xd9", start + 2) if start != -1 else -1
            if start != -1 and end != -1:
                jpg = self._buf[start:end + 2]
                self._buf = self._buf[end + 2:]
                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                return frame is not None, frame
            chunk = self._proc.stdout.read(4096)
            if not chunk:
                return False, None
            self._buf += chunk

    def isOpened(self):
        return self._proc.poll() is None

    def release(self):
        self._proc.terminate()
        try:
            self._proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            self._proc.kill()

def open_capture(args):
    """Abre a câmera local (via rpicam-vid) ou o stream MJPEG configurado no pipeline de streaming."""
    if args.source == "mjpeg":
        cap = cv2.VideoCapture(args.url)
    else:
        cap = RpicamCapture(int(args.source), args.width, args.height)
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir a fonte: {args.source}")
    return cap

def is_sharp_enough(frame: np.ndarray, threshold: float = 80.0) -> bool:
    """
    Descarta frames borrados usando a variância do Laplaciano.
    Um frame nítido tem alta variância nas bordas detectadas.
    Threshold calibrado para câmeras CSI em 640x480.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score >= threshold

def main():
    args = parse_args()
    cap  = open_capture(args)
    saved   = 0
    skipped = 0
    last_saved = 0.0

    print(f"[INFO] Iniciando captura: {args.total} frames | intervalo: {args.interval}s")
    print(f"[INFO] Salvando em: {OUTPUT_DIR.resolve()}")
    print("[INFO] Pressione Ctrl+C para encerrar antecipadamente.")

    try:
        while saved < args.total:
            if args.manual:
                input(f"  [{saved:>3}/{args.total}] Pressione ENTER para capturar...")
                if args.source == "mjpeg":
                    ret, frame = fetch_snapshot(args.snapshot_url)
                else:
                    flush_until = time.time() + 0.5
                    while time.time() < flush_until:
                        cap.read()
                    ret, frame = cap.read()
            else:
                ret, frame = cap.read()

            if not ret:
                print("[AVISO] Frame inválido, tentando novamente...")
                time.sleep(0.1)
                continue

            if not args.manual:
                now = time.time()
                if now - last_saved < args.interval:
                    continue

            if not is_sharp_enough(frame):
                skipped += 1
                if args.manual:
                    print("  [AVISO] Frame borrado, descartado — tente de novo.")
                continue

            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            path = OUTPUT_DIR / f"frame_{ts}.jpg"
            cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            saved += 1
            last_saved = time.time()
            end_char = "\n" if args.manual else "\r"
            print(f"  [{saved:>3}/{args.total}] Salvo: {path.name} ",
                  f"(descartados: {skipped})", end=end_char)
    except KeyboardInterrupt:
        print("\n[INFO] Captura interrompida pelo usuário.")
    finally:
        cap.release()
        print(f"\n[OK] {saved} frames salvos em {OUTPUT_DIR}")
        print(f"[OK] {skipped} frames borrados descartados automaticamente")

if __name__ == "__main__":
    main()

