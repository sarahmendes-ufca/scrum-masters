import os

import torch
import ultralytics
from ultralytics import YOLO

# Permite o carregamento completo do modelo em versões do PyTorch >= 2.6
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

# Adiciona a classe Sequential aos globais seguros do PyTorch
try:
    torch.serialization.add_safe_globals([torch.nn.modules.container.Sequential])
except AttributeError:
    pass

# Permite que o PyTorch 2.6+ carregue as classes do Ultralytics com seguranca
try:
    torch.serialization.add_safe_globals([ultralytics.nn.tasks.DetectionModel])
except AttributeError:
    pass

_models = {}

def get_default_model_name() -> str:
    return "yolov8n.pt"

def load_model(model_name: str = "yolov8n.pt"):
    if model_name not in _models:
        _models[model_name] = YOLO(model_name)
    return _models[model_name]
