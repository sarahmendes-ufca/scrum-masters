#!/usr/bin/env python3
"""
stream/video_visualize.py — Roda o modelo de detecção (object detection) sobre um arquivo de vídeo.

Execução recomendada:
    python3 stream/video_visualize.py --model runs/epi-v1/weights/best.pt --input artefatos/videos/17.mp4 --output resultado_deteccao.mp4 --no-display
Parâmetros:
    --input         (obrigatório) Caminho do vídeo de entrada.
    --output        Caminho do vídeo anotado de saída. Padrão: "resultado.mp4".
    --model         Caminho do arquivo de pesos (.pt) do modelo de detecção. Padrão: "models/best.pt".
    --conf          Limiar mínimo de confiança para considerar uma detecção (0.0–1.0). Padrão: 0.4.
    --iou           Limiar de IoU usado no NMS para suprimir caixas sobrepostas. Padrão: 0.5.
    --infer-every   Roda a inferência a cada N frames; reaproveita o último resultado
                     nos frames intermediários (1 = infere em todo frame). Padrão: 1.
    --single-box    Mantém apenas a detecção de maior confiança por frame
    --multi-box     Desativa o modo de caixa única e desenha todas as detecções do frame
                     (sobrepõe --single-box).
    --no-display    Não abre janela de preview (cv2.imshow) — necessário em sessões
                     SSH/headless sem display gráfico.
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
        "--iou",
        type=float,
        default=0.5,
        help="Limiar de IoU para o NMS (padrão: 0.5)",
    )
    p.add_argument(
        "--infer-every",
        type=int,
        default=1,
        help="Roda inferência a cada N frames (1 = todo frame)",
    )
    p.add_argument(
        "--single-box",
        action="store_true",
        default=True,
        help="Mantém apenas a detecção de maior confiança por frame (padrão: ativado)",
    )
    p.add_argument(
        "--multi-box",
        dest="single_box",
        action="store_false",
        help="Desativa o modo de caixa única e desenha todas as detecções",
    )
    p.add_argument(
        "--no-display", action="store_true", help="Não abre janela (modo headless/SSH)"
    )
    return p.parse_args()


def draw_boxes(frame, boxes):
    """Desenha bounding boxes + labels no frame. boxes: [(label, conf, x1,y1,x2,y2), ...]"""
    output = frame.copy()
    for label, conf, x1, y1, x2, y2 in boxes:
        cor = (0, 255, 0) if "integro" in label.lower() else (0, 0, 255)
        cv2.rectangle(output, (x1, y1), (x2, y2), cor, 2)

        caption = f"{label} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(output, (x1, y1 - th - 10), (x1 + tw + 4, y1), cor, -1)
        cv2.putText(
            output,
            caption,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
    return output


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
    print(f"[INFO] Modo caixa única: {args.single_box}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    frame_idx = 0
    last_boxes = []  # [(label, conf, x1,y1,x2,y2), ...]

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if frame_idx % args.infer_every == 0:
            # agnostic_nms=True: suprime caixas sobrepostas mesmo entre classes
            # diferentes (evita "rotulo ausente" e "rotulo integro" desenhados
            # ao mesmo tempo sobre a mesma região do rótulo)
            results = model(
                frame,
                conf=args.conf,
                iou=args.iou,
                agnostic_nms=True,
                verbose=False,
            )

            candidatas = []
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    label = model.names[int(box.cls[0])]
                    conf = float(box.conf[0])
                    candidatas.append((label, conf, int(x1), int(y1), int(x2), int(y2)))

            if args.single_box:
                # Mantém apenas a detecção de maior confiança (1 garrafa por vez na esteira)
                last_boxes = [max(candidatas, key=lambda c: c[1])] if candidatas else []
            else:
                last_boxes = candidatas

            print(
                f"[Frame {frame_idx}/{total_frames}] "
                f"{len(last_boxes)} detecção(ões): "
                f"{[f'{l} {c:.0%}' for l, c, *_ in last_boxes]}"
            )

        output = draw_boxes(frame, last_boxes)
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
