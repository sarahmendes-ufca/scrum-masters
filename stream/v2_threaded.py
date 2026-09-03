#!/usr/bin/env python3
"""
stream/v2_threaded.py — Captura e inferência em threads separadas.
Buffer de 1 frame elimina o acúmulo e garante processamento do frame atual.
Execução: python3 stream/v2_threaded.py --device 0
"""

import argparse
import json
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

_orig_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)


torch.load = _patched_torch_load
sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Classe de captura em thread dedicada ────────────────────
class CameraCapture:
    """
    Captura frames em thread separada.
    Mantém sempre o frame mais recente disponível.
    O buffer de tamanho 1 descarta frames antigos automaticamente.
    """

    def __init__(self, device: int, width: int, height: int, fps: int = 30):
        self._cmd = [
            "rpicam-vid",
            "-t",
            "0",
            "-n",
            "--codec",
            "mjpeg",
            "--camera",
            str(device),
            "--width",
            str(width),
            "--height",
            str(height),
            "--framerate",
            str(fps),
            "-o",
            "-",
        ]
        self._proc = None
        # Buffer de 1 frame: producer sobrescreve, consumer lê o mais recente
        self._buffer = queue.Queue(maxsize=1)
        self._running = threading.Event()
        self._running.set()
        self._thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="CaptureThread"
        )
        # Métricas de captura
        self.frames_captured = 0
        self.frames_dropped = 0

    def start(self):
        self._proc = subprocess.Popen(
            self._cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        self._thread.start()
        print(
            f"[CameraCapture] rpicam-vid iniciado (pid={self._proc.pid}) — buffer maxsize=1"
        )
        return self

    def _capture_loop(self):
        """Loop interno do thread de captura."""
        raw = b""
        while self._running.is_set():
            chunk = self._proc.stdout.read(4096)
            if not chunk:
                break
            raw += chunk
            # Mantém só o último frame JPEG completo; descarta os anteriores
            end = raw.rfind(b"\xff\xd9")
            if end == -1:
                continue
            start = raw.rfind(b"\xff\xd8", 0, end)
            if start == -1:
                continue
            jpg = raw[start : end + 2]
            raw = raw[end + 2 :]
            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            # Se o buffer está cheio, descarta o frame antigo
            if self._buffer.full():
                try:
                    self._buffer.get_nowait()
                    self.frames_dropped += 1
                except queue.Empty:
                    pass
            self._buffer.put(frame)
            self.frames_captured += 1

    def read(self, timeout: float = 1.0):
        """Retorna o frame mais recente. Bloqueia até timeout se nenhum disponível."""
        try:
            return self._buffer.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        self._running.clear()
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        print(
            f"[CameraCapture] Encerrada — ",
            f"capturados: {self.frames_captured}, ",
            f"descartados: {self.frames_dropped}",
        )


# ── Classe de inferência com métricas ───────────────────────
class YOLOInference:
    """Wrapper do YOLO com métricas de latência acumuladas."""

    def __init__(self, model_path: str, conf: float = 0.4):
        print(f"[YOLOInference] Carregando: {model_path}")
        self.model = YOLO(model_path)
        self.conf = conf
        self.count = 0
        self.total_ms = 0.0

    def run(self, frame):
        """Executa inferência e retorna (frame_anotado, n_deteccoes, ms)."""
        t0 = time.perf_counter()
        results = self.model(frame, conf=self.conf, verbose=False)
        elapsed = (time.perf_counter() - t0) * 1000

        self.count += 1
        self.total_ms += elapsed

        # Anota bounding boxes diretamente no frame
        annotated = results[0].plot()
        n_det = len(results[0].boxes)
        return annotated, n_det, elapsed

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.count if self.count > 0 else 0.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--model", type=str, default="models/yolov8n.pt")
    p.add_argument("--conf", type=float, default=0.4)
    p.add_argument("--frames", type=int, default=100)
    return p.parse_args()


def main():
    args = parse_args()

    camera = CameraCapture(args.device, args.width, args.height, args.fps)
    yolo = YOLOInference(args.model, args.conf)

    camera.start()
    time.sleep(0.5)  # aguarda câmera estabilizar

    print(f"[INFO] Processando {args.frames} frames com threading...")
    print(f"{'Frame':>6} | {'Inferência':>10} | {'FPS inst.':>9} | {'Detecções':>9}")
    print("-" * 48)

    t_start = time.perf_counter()
    frame_count = 0

    while frame_count < args.frames:
        frame = camera.read(timeout=2.0)
        if frame is None:
            print("[AVISO] Timeout na leitura do frame.")
            continue

        annotated, n_det, infer_ms = yolo.run(frame)
        frame_count += 1

        elapsed_total = time.perf_counter() - t_start
        fps_avg = frame_count / elapsed_total if elapsed_total > 0 else 0

        if frame_count % 10 == 0:
            print(
                f"{frame_count:>6} | {infer_ms:>9.1f}ms | ",
                f"{fps_avg:>8.1f} | {n_det:>9}",
            )

    camera.stop()

    total_time = time.perf_counter() - t_start
    print("\n" + "=" * 58)
    print("RELATÓRIO — Threading com buffer de 1 frame")
    print("=" * 58)
    print(f"  Frames processados   : {frame_count}")
    print(f"  Tempo total          : {total_time:.1f} s")
    print(f"  FPS médio sustentado : {frame_count / total_time:.1f} FPS")
    print(f"  Inferência média     : {yolo.avg_ms:.1f} ms")
    print(f"  Frames capturados    : {camera.frames_captured}")
    print(
        f"  Frames descartados   : {camera.frames_dropped} ",
        f"({100 * camera.frames_dropped / max(camera.frames_captured, 1):.0f}%)",
    )
    print("=" * 58)


if __name__ == "__main__":
    main()
