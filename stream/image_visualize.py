#!/usr/bin/env python3
"""
stream/image_visualize.py — Roda o modelo de detecção (object detection) sobre uma ou mais imagens.

Execução:
    python3 stream/image_visualize.py --input teste_garrafa.jpg --output resultado.jpg
    python3 stream/image_visualize.py --input imagens/ --output-dir resultados/

Parâmetros:
    --input         (obrigatório) Caminho de uma imagem OU de uma pasta contendo várias imagens
                     (.jpg, .jpeg, .png).
    --output        Caminho do arquivo de saída quando --input é uma única imagem.
                     Padrão: "resultado.jpg".
    --output-dir     Pasta de saída quando --input é uma pasta com várias imagens.
                     Padrão: "resultados/".
    --model         Caminho do arquivo de pesos (.pt) do modelo de detecção. Padrão: "models/best.pt".
    --conf          Limiar mínimo de confiança para considerar uma detecção (0.0–1.0). Padrão: 0.4.
    --iou           Limiar de IoU usado no NMS para suprimir caixas sobrepostas. Padrão: 0.5.
    --single-box    Mantém apenas a detecção de maior confiança por imagem — útil quando
                     só um objeto por vez aparece na cena (ex.: 1 garrafa por foto).
                     Ativado por padrão.
    --multi-box     Desativa o modo de caixa única e desenha todas as detecções da imagem
                     (sobrepõe --single-box).
    --show          Abre uma janela exibindo cada imagem anotada (precisa de display gráfico;
                     não usar em sessões SSH/headless).
"""

import argparse
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

_orig_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

EXTENSOES_VALIDAS = {".jpg", ".jpeg", ".png"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        type=str,
        required=True,
        help="Caminho de uma imagem ou de uma pasta com imagens",
    )
    p.add_argument(
        "--output",
        type=str,
        default="resultado.jpg",
        help="Caminho de saída (usado quando --input é um único arquivo)",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="resultados",
        help="Pasta de saída (usada quando --input é uma pasta)",
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
        "--single-box",
        action="store_true",
        default=True,
        help="Mantém apenas a detecção de maior confiança por imagem (padrão: ativado)",
    )
    p.add_argument(
        "--multi-box",
        dest="single_box",
        action="store_false",
        help="Desativa o modo de caixa única e desenha todas as detecções",
    )
    p.add_argument(
        "--show",
        action="store_true",
        help="Exibe cada imagem anotada em uma janela (precisa de display gráfico)",
    )
    return p.parse_args()


def draw_boxes(frame, boxes):
    """Desenha bounding boxes + labels na imagem. boxes: [(label, conf, x1,y1,x2,y2), ...]"""
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


def processar_imagem(model, image_path: Path, output_path: Path, args):
    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"[ERRO] Não foi possível abrir: {image_path}")
        return

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
        boxes = [max(candidatas, key=lambda c: c[1])] if candidatas else []
    else:
        boxes = candidatas

    print(
        f"[{image_path.name}] {len(boxes)} detecção(ões): "
        f"{[f'{l} {c:.0%}' for l, c, *_ in boxes]}"
    )

    output = draw_boxes(frame, boxes)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), output)

    if args.show:
        cv2.imshow(f"Resultado — {image_path.name} (pressione qualquer tecla)", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    args = parse_args()

    print(f"[INFO] Carregando modelo: {args.model}")
    model = YOLO(args.model)
    print(f"[INFO] Modo caixa única: {args.single_box}")

    input_path = Path(args.input)

    if input_path.is_dir():
        imagens = sorted(
            p for p in input_path.iterdir() if p.suffix.lower() in EXTENSOES_VALIDAS
        )
        if not imagens:
            print(f"[ERRO] Nenhuma imagem encontrada em: {input_path}")
            return

        output_dir = Path(args.output_dir)
        print(
            f"[INFO] {len(imagens)} imagem(ns) encontrada(s). Salvando em: {output_dir}"
        )
        for img_path in imagens:
            processar_imagem(model, img_path, output_dir / img_path.name, args)

    elif input_path.is_file():
        processar_imagem(model, input_path, Path(args.output), args)

    else:
        print(f"[ERRO] Caminho não encontrado: {input_path}")
        return

    print("[INFO] Concluído.")


if __name__ == "__main__":
    main()
