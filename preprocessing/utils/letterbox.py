"""
preprocessing/utils/letterbox.py
Implementação manual do letterbox — a mesma lógica usada internamente
pelo YOLOv8 (ultralytics/data/augment.py).
"""
from typing import Tuple
import cv2
import numpy as np


def letterbox(
    frame: np.ndarray,
    target_size: int = 640,
    pad_color: int = 114,
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Redimensiona preservando proporção e adiciona padding cinza.

    Returns:
        frame_lb:   imagem redimensionada + padded (target_size × target_size)
        scale:      fator de escala aplicado (usado para ajustar bboxes)
        (pad_w, pad_h): padding adicionado em cada lado (horizontal, vertical)
    """
    h, w = frame.shape[:2]

    # Fator de escala: limitar pela dimensão maior
    scale = min(target_size / h, target_size / w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    # Redimensiona mantendo proporção
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Calcula padding simétrico
    pad_w = (target_size - new_w) // 2
    pad_h = (target_size - new_h) // 2

    # Aplica padding com cor cinza (valor 114 ≈ média ImageNet em uint8)
    frame_lb = np.full((target_size, target_size, 3), pad_color, dtype=np.uint8)
    frame_lb[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

    return frame_lb, scale, (pad_w, pad_h)


def adjust_bboxes(
    boxes_xyxy: np.ndarray,
    scale: float,
    pad_w: int,
    pad_h: int,
) -> np.ndarray:
    """
    Corrige coordenadas de bounding boxes produzidas após inferência
    em uma imagem letterboxed, mapeando-as de volta ao espaço original.

    Args:
        boxes_xyxy: array (N, 4) com coords [x1, y1, x2, y2] no espaço letterboxed
        scale:      fator de escala retornado por letterbox()
        pad_w/h:    padding retornado por letterbox()

    Returns:
        boxes_orig: coords ajustadas ao espaço da imagem original
    """
    boxes = boxes_xyxy.copy().astype(float)
    # Remove o padding
    boxes[:, [0, 2]] -= pad_w   # eixo x
    boxes[:, [1, 3]] -= pad_h   # eixo y
    # Desfaz a escala
    boxes /= scale
    return boxes
