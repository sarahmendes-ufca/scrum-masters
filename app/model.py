import os
from pathlib import Path

import torch
from ultralytics import YOLO

# Ajuste para PyTorch 2.6+: desativa weights_only temporariamente
# no carregamento de pesos do YOLO
_orig_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False

    return _orig_torch_load(*args, **kwargs)


torch.load = _patched_torch_load


MODELS_DIR = Path("/app/models")
_cache: dict = {}


def load_model(model_name: str) -> YOLO:
    """Carrega o modelo da primeira vez e mantém em cache."""
    if model_name not in _cache:
        model_path = MODELS_DIR / model_name

        if not model_path.exists():
            raise FileNotFoundError(
                f"Modelo '{model_name}' não encontrado em "
                f"{MODELS_DIR}. "
                f"Arquivos disponíveis: "
                f"{list(MODELS_DIR.glob('*.pt'))}"
            )

        _cache[model_name] = YOLO(str(model_path))

    return _cache[model_name]


def get_default_model_name() -> str:
    return os.getenv("MODEL_NAME", "yolov8n.pt")
