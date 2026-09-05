"""
preprocessing/utils/evaluate.py
Avalia o mAP@0.5 de um pipeline de pré-processamento no dataset epi-v1.
Recebe uma função de pré-processamento e retorna as métricas.
"""
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import cv2
from ultralytics import YOLO
import shutil
import yaml
import torch

_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

DATASET_YAML = "dataset/exports/epi-v1/data.yaml"
MODEL_PATH   = "models/yolov8n.pt"


def evaluate_pipeline(
    preprocess_fn: Optional[Callable] = None,
    label: str = "baseline",
    split: str = "val",
    verbose: bool = False,
) -> dict:
    """
    Avalia o mAP@0.5 do modelo com uma função de pré-processamento opcional.

    Args:
        preprocess_fn: função que recebe um frame BGR NumPy e retorna
                       um frame transformado (qualquer formato aceito pelo YOLO).
                       Se None, usa o comportamento padrão da Ultralytics.
        label:         nome para identificar este experimento no log.
        split:         split do dataset a avaliar ('val' ou 'test').

    Returns:
        dict com map50, map50_95, tempo médio de pré-processamento em ms.

    Nota sobre E1 (espaço de cor): esta função salva os frames pré-processados
    em disco e aponta o model.val() para eles. O model.val() sempre relê o
    arquivo e aplica sua própria conversão BGR->RGB internamente -- por isso
    ela mede corretamente transformações espaciais/de intensidade (resize,
    blur, CLAHE), mas não consegue medir o efeito de inverter a ordem dos
    canais (E1): o resultado do E1 aparece com delta ~0 aqui, de propósito.
    A verificação do E1 é visual, via e1_visualize.py.
    """
    model = YOLO(MODEL_PATH)

    if preprocess_fn is None:
        # Avaliação padrão — sem pré-processamento customizado
        metrics = model.val(
            data=DATASET_YAML,
            split=split,
            verbose=verbose,
        )
        preproc_ms = 0.0
    else:
        # Roboflow exporta a pasta de validação como "valid/", mas a chave
        # do data.yaml (e o split= do Ultralytics) usa "val" -- traduz aqui
        split_dirname = {"val": "valid", "test": "test", "train": "train"}.get(split, split)
        dataset_dir = Path(DATASET_YAML).parent
        src_images_dir = dataset_dir / split_dirname / "images"
        src_labels_dir = dataset_dir / split_dirname / "labels"
        images = sorted(src_images_dir.glob("*.jpg")) + sorted(src_images_dir.glob("*.png"))

        # Diretório temporário com as imagens JÁ pré-processadas + os mesmos
        # labels -- o model.val() só sabe ler do disco, então o resultado de
        # preprocess_fn precisa ser gravado antes de avaliar (esse era o bug:
        # antes, os frames processados eram calculados e descartados)
        safe_label = "".join(c if c.isalnum() else "_" for c in label)
        tmp_root = Path("preprocessing/outputs/_tmp_eval") / safe_label
        if tmp_root.exists():
            shutil.rmtree(tmp_root)
        tmp_images_dir = tmp_root / "images"
        tmp_labels_dir = tmp_root / "labels"
        tmp_images_dir.mkdir(parents=True, exist_ok=True)
        tmp_labels_dir.mkdir(parents=True, exist_ok=True)

        preproc_times = []
        for img_path in images:
            frame = cv2.imread(str(img_path))
            t0 = time.perf_counter()
            frame_proc = preprocess_fn(frame)
            preproc_times.append((time.perf_counter() - t0) * 1000)
            cv2.imwrite(str(tmp_images_dir / img_path.name), frame_proc)
            label_src = src_labels_dir / f"{img_path.stem}.txt"
            if label_src.exists():
                shutil.copy(label_src, tmp_labels_dir / label_src.name)

        # data.yaml temporário apontando para as imagens já processadas
        with open(DATASET_YAML) as f:
            base_cfg = yaml.safe_load(f)
        tmp_yaml_cfg = {
            "path": str(tmp_root.resolve()),
            "train": "images",
            "val": "images",
            "test": "images",
            "names": base_cfg["names"],
        }
        tmp_yaml = tmp_root / "data.yaml"
        with open(tmp_yaml, "w") as f:
            yaml.safe_dump(tmp_yaml_cfg, f)

        # Roda inferência nas imagens JÁ processadas (não mais nas originais)
        metrics = model.val(data=str(tmp_yaml), split="val", verbose=verbose)
        preproc_ms = float(np.mean(preproc_times)) if preproc_times else 0.0

    map50    = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)
    print(f"[{label:30s}]  mAP@0.5={map50:.4f}  mAP@0.5:0.95={map50_95:.4f}  ",
          f"preproc={preproc_ms:.1f}ms")
    return {"label": label, "map50": map50, "map50_95": map50_95,
            "preproc_ms": preproc_ms}
