"""
Gera versões escuras das imagens de validação para testar equalização.
Salva em dataset/exports/epi-v1-dark/ mantendo os labels originais.
"""
import cv2
import numpy as np
import shutil
from pathlib import Path

SRC  = Path("dataset/exports/epi-v1/valid")
DEST = Path("dataset/exports/epi-v1-dark/valid")

(DEST / "images").mkdir(parents=True, exist_ok=True)
(DEST / "labels").mkdir(parents=True, exist_ok=True)

# Copia labels sem alteração — a posição dos objetos não muda
for lbl in (SRC / "labels").glob("*.txt"):
    shutil.copy(lbl, DEST / "labels" / lbl.name)

# Gera imagens escurecidas (gamma > 1 escurece, < 1 clareia)
gamma  = 2.2   # simula subexposição severa
table  = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype(np.uint8)

for img_path in (SRC / "images").glob("*.jpg"):
    img  = cv2.imread(str(img_path))
    dark = cv2.LUT(img, table)   # aplica a curva de gamma
    cv2.imwrite(str(DEST / "images" / img_path.name), dark)

print(f"Geradas {len(list((DEST/'images').glob('*.jpg')))} imagens escurecidas")
print("Para criar o data.yaml de teste:")
print("  cp dataset/exports/epi-v1/data.yaml dataset/exports/epi-v1-dark/")
print("  (Edite os caminhos train/val/test no arquivo copiado)")
