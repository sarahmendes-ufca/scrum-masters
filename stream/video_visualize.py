#!/usr/bin/env python3
"""
stream/run_on_video.py — Roda o modelo de classificação usando um arquivo de vídeo.
Execução: python3 stream/run_on_video.py --input artefatos/videos/meu_video.mp4 --output resultado.mp4
"""

import argparse

import cv2
import torch
from ultralytics import YOLO

_orig_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)


torch.load = _patched_torch_load


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input", type=str, required=True, help="Caminho do vídeo de entrada"
    )
    p.add_argument(
        "--output",
        type=str,
        default="resultado.mp4",
        help="Caminho do vídeo anotado de saída",
    )
    p.add_argument("--model", type=str, default="models/best.pt")
    p.add_argument("--conf", type=float, default=0.4)
    p.add_argument(
        "--infer-every",
        type=int,
        default=1,
        help="Roda inferência a cada N frames (1 = todo frame)",
    )
    p.add_argument(
        "--no-display", action="store_true", help="Não abre janela (modo headless/SSH)"
    )
    return p.parse_args()


def main():
    args = parse_args()

    print(f"[INFO] Carregando modelo: {args.model}")
    model = YOLO(args.model)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"[ERRO] Não foi possível abrir o vídeo: {args.input}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Vídeo: {width}x{height} @ {fps:.1f}fps, {total_frames} frames")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    frame_idx = 0
    last_cls = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if frame_idx % args.infer_every == 0:
            results = model(frame, conf=args.conf, verbose=False)
            for r in results:
                if r.probs is not None:
                    top1 = int(r.probs.top1)
                    label = model.names[top1]
                    conf = float(r.probs.top1conf)
                    last_cls = (label, conf)
                    print(f"[Frame {frame_idx}/{total_frames}] {label} ({conf:.0%})")

        output = frame.copy()
        if last_cls is not None:
            label, conf = last_cls
            cor = (0, 255, 0) if "integro" in label.lower() else (0, 0, 255)  # RGB
            caption = f"{label} {conf:.0%}"
            cv2.putText(
                output,
                caption,
                (10, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                cor,
                2,
            )

        writer.write(output)

        if not args.no_display:
            cv2.imshow("Resultado — pressione q para sair", output)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    writer.release()
    if not args.no_display:
        cv2.destroyAllWindows()
    print(f"[INFO] Concluído. Vídeo salvo em: {args.output}")


if __name__ == "__main__":
    main()
