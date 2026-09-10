# %%
# Célula 1 --- Patch do torch.load e confirmação da GPU
import torch

_orig_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

print("CUDA disponível:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))

# %%
# Célula 2 --- Treinamento com GPU
from ultralytics import YOLO

if __name__ == "__main__":
    # model = YOLO("yolov8n.pt")
    model = YOLO("yolov8n-cls.pt")
    results = model.train(
        data="dataset",
        epochs=100,
        imgsz=224,  # Como classificacao nao precisa de tanto detalhe, posso dimnuir a resolucao
        device=0,
        patience=20,
        project="runs",
        name="epi-v1-cls",  # cls = classification
    )
    print("Pesos salvos em:", results.save_dir)

    print("top1_acc:", results.results_dict.get("metrics/accuracy_top1"))
    print("top5_acc:", results.results_dict.get("metrics/accuracy_top5"))
