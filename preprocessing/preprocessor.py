"""
preprocessing/preprocessor.py
Módulo central de pré-processamento de imagens para o projeto yolo-edge-api.
Integra-se ao pipeline de tempo real e à API REST.
"""
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from preprocessing.utils.letterbox import letterbox


@dataclass
class PreprocessConfig:
    """
    Configuração imutável do pipeline de pré-processamento.
    Instanciar uma vez e reutilizar para todos os frames.
    """
    infer_size:    int   = 320
    convert_rgb:   bool  = True
    use_letterbox: bool  = True
    gaussian_blur: bool  = False
    gaussian_ksize: int  = 3
    gaussian_sigma: float = 0.8
    median_blur:   bool  = False
    median_ksize:  int   = 3
    clahe:         bool  = False
    clahe_clip:    float = 2.0
    clahe_tile:    int   = 8
    clahe_space:   str   = "lab"
    normalize:     bool  = False


@dataclass
class PreprocessResult:
    """Resultado do pré-processamento — imagem + metadados."""
    frame:     np.ndarray
    scale:     float = 1.0   # escala uniforme (válida quando use_letterbox=True)
    scale_x:   float = 1.0   # escala real no eixo horizontal
    scale_y:   float = 1.0   # escala real no eixo vertical
    pad_w:     int   = 0
    pad_h:     int   = 0
    orig_size: Tuple[int, int] = (0, 0)


class Preprocessor:
    """
    Encapsula o pipeline de pré-processamento configurável.
    Thread-safe: não mantém estado mutável entre chamadas.
    """

    def __init__(self, config: Optional[PreprocessConfig] = None):
        self.cfg = config or PreprocessConfig()
        if self.cfg.clahe:
            self._clahe = cv2.createCLAHE(
                clipLimit=self.cfg.clahe_clip,
                tileGridSize=(self.cfg.clahe_tile, self.cfg.clahe_tile),
            )

    def process(self, frame: np.ndarray) -> PreprocessResult:
        """
        Aplica o pipeline completo a um único frame BGR NumPy.
        """
        orig_h, orig_w = frame.shape[:2]
        out = frame.copy()

        if self.cfg.clahe:
            out = self._apply_clahe(out)

        if self.cfg.convert_rgb:
            out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

        if self.cfg.gaussian_blur:
            k = self.cfg.gaussian_ksize
            out = cv2.GaussianBlur(out, (k, k), sigmaX=self.cfg.gaussian_sigma)
        elif self.cfg.median_blur:
            out = cv2.medianBlur(out, self.cfg.median_ksize)

        if self.cfg.use_letterbox:
            out, scale, (pad_w, pad_h) = letterbox(out, self.cfg.infer_size)
            scale_x = scale_y = scale
        else:
            out = cv2.resize(out, (self.cfg.infer_size, self.cfg.infer_size))
            # Sem letterbox, a proporção é distorcida -- x e y escalam de
            # forma diferente sempre que a imagem de entrada não for quadrada.
            scale_x = self.cfg.infer_size / orig_w
            scale_y = self.cfg.infer_size / orig_h
            scale   = min(scale_x, scale_y)  # mantido só por compatibilidade
            pad_w = pad_h = 0

        if self.cfg.normalize:
            out = out.astype(np.float32) / 255.0

        return PreprocessResult(
            frame=out, scale=scale, scale_x=scale_x, scale_y=scale_y,
            pad_w=pad_w, pad_h=pad_h, orig_size=(orig_h, orig_w),
        )

    def adjust_boxes(self, boxes_xyxy: np.ndarray, result: PreprocessResult) -> np.ndarray:
        """
        Ajusta coordenadas de bboxes do espaço processado para o original.
        Usa scale_x/scale_y separadamente -- correto tanto com letterbox
        (onde os dois são iguais) quanto sem (onde podem ser diferentes).
        """
        boxes = boxes_xyxy.copy().astype(float)
        boxes[:, [0, 2]] -= result.pad_w
        boxes[:, [1, 3]] -= result.pad_h
        boxes[:, [0, 2]] /= result.scale_x
        boxes[:, [1, 3]] /= result.scale_y
        return boxes

    def _apply_clahe(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Aplica CLAHE no canal de luminância do espaço escolhido."""
        if self.cfg.clahe_space == "lab":
            lab     = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l_cl    = self._clahe.apply(l)
            return cv2.cvtColor(cv2.merge([l_cl, a, b]), cv2.COLOR_LAB2BGR)
        else:  # hsv
            hsv     = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            v_cl    = self._clahe.apply(v)
            return cv2.cvtColor(cv2.merge([h, s, v_cl]), cv2.COLOR_HSV2BGR)


# ── Configurações pré-definidas ───────────────────────────────────

CONFIG_DEFAULT = PreprocessConfig(
    infer_size=320,
    convert_rgb=True,
    use_letterbox=True,
    gaussian_blur=False,
    clahe=False,
)

CONFIG_LOW_LIGHT = PreprocessConfig(
    infer_size=320,
    convert_rgb=True,
    use_letterbox=True,
    clahe=True,
    clahe_clip=2.0,
    clahe_tile=8,
    clahe_space="lab",
)

CONFIG_HIGH_QUALITY = PreprocessConfig(
    infer_size=640,
    convert_rgb=True,
    use_letterbox=True,
    gaussian_blur=False,
    clahe=False,
)
