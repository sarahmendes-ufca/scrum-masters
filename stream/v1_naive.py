#!/usr/bin/env python3
"""
stream/v1_naive.py — Implementação ingênua: diagnóstico de FPS e latência.
Execução: python3 stream/v1_naive.py --device 0 --width 640 --height 480
"""

import argparse
import subprocess
import sys
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


def _read_next_frame(proc, leftover: bytes):
    """
    Le o proximo frame JPEG completo disponivel no stdout do rpicam-vid.
    Nao pula para o mais recente de proposito -- reproduz o mesmo
    comportamento de acumulo de buffer que cv2.VideoCapture() tinha,
    para que o diagnostico continue medindo o problema real.
    """
    buf = leftover
    while True:
        start = buf.find(b"\xff\xd8")
        end = buf.find(b"\xff\xd9", start + 2) if start != -1 else -1
        if start != -1 and end != -1:
            jpg = buf[start : end + 2]
            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            return buf[end + 2 :], frame
        chunk = proc.stdout.read(4096)
        if not chunk:
            return buf, None
        buf += chunk


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--device", type=int, default=0, help="Índice do dispositivo de câmera"
    )
    p.add_argument("--width", type=int, default=640, help="Largura de captura")
    p.add_argument("--height", type=int, default=480, help="Altura de captura")
    p.add_argument("--model", type=str, default="models/yolov8n.pt")
    p.add_argument("--conf", type=float, default=0.4)
    p.add_argument(
        "--frames", type=int, default=100, help="Frames para medir antes de encerrar"
    )
    return p.parse_args()


def main():
    args = parse_args()

    print(f"[INFO] Carregando modelo: {args.model}")
    model = YOLO(args.model)

    print(
        f"[INFO] Abrindo câmera CSI via rpicam-vid (câmera {args.device}) @ {args.width}x{args.height}"
    )
    cmd = [
        "rpicam-vid",
        "-t",
        "0",
        "-n",
        "--codec",
        "mjpeg",
        "--camera",
        str(args.device),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "-o",
        "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    leftover = b""

    # ── Métricas de diagnóstico ─────────────────────────────
    frame_count = 0
    total_capture = 0.0
    total_infer = 0.0
    total_cycle = 0.0

    print(f"[INFO] Medindo {args.frames} frames...")
    print(
        f"{'Frame':>6} | {'Captura':>8} | {'Inferência':>10} | {'Ciclo':>8} | {'FPS inst.':>9}"
    )
    print("-" * 58)

    while frame_count < args.frames:
        t0 = time.perf_counter()

        # ── Etapa 1: captura ─────────────────────────────────
        leftover, frame = _read_next_frame(proc, leftover)
        t1 = time.perf_counter()

        if frame is None:
            print("[AVISO] Frame inválido, pulando.")
            continue

        # ── Etapa 2: inferência ──────────────────────────────
        results = model(frame, conf=args.conf, verbose=False)
        t2 = time.perf_counter()

        cap_ms = (t1 - t0) * 1000
        infer_ms = (t2 - t1) * 1000
        cycle_ms = (t2 - t0) * 1000
        fps_inst = 1000 / cycle_ms if cycle_ms > 0 else 0

        total_capture += cap_ms
        total_infer += infer_ms
        total_cycle += cycle_ms
        frame_count += 1

        if frame_count % 10 == 0:
            print(
                f"{frame_count:>6} | {cap_ms:>7.1f}ms | {infer_ms:>9.1f}ms | {cycle_ms:>7.1f}ms | {fps_inst:>8.1f}"
            )

    proc.terminate()
    try:
        proc.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        proc.kill()

    # ── Relatório final ─────────────────────────────────────
    n = frame_count
    print("\n" + "=" * 58)
    print("RELATÓRIO DE DIAGNÓSTICO — Abordagem Ingênua")
    print("=" * 58)
    print(f"  Frames medidos    : {n}")
    print(f"  Captura média     : {total_capture / n:>7.1f} ms")
    print(f"  Inferência média  : {total_infer / n:>7.1f} ms")
    print(f"  Ciclo médio       : {total_cycle / n:>7.1f} ms")
    print(f"  FPS sustentado    : {1000 / (total_cycle / n):>7.1f} FPS")
    print("=" * 58)
    print("")
    print("DIAGNÓSTICO:")
    avg_cap = total_capture / n
    if avg_cap > 50:
        print(f"  ⚠️  Captura alta ({avg_cap:.0f}ms): buffer acumulado detectado.")
        print("      Solução: use threading + buffer de 1 frame (v2_threaded.py)")
    else:
        print(f"  ✓  Captura OK ({avg_cap:.0f}ms)")


if __name__ == "__main__":
    main()
