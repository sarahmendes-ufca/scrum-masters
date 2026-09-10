#!/usr/bin/env python3
"""
preprocessing/experiments/e4_generate_dark_yaml.py

Gera o data.yaml do dataset escurecido (epi-v1-dark), usado no
Experimento E4 (equalização de histograma vs CLAHE).

O epi-v1-dark só tem a pasta valid/ (as imagens escurecidas geradas
por e4_generate_dark.py) -- train/val/test apontam todos para ela,
o que é suficiente já que esse dataset é usado apenas para avaliação
(evaluate_pipeline / model.val()), nunca para treino.

Pré-requisito: rodar antes o e4_generate_dark.py, que cria as
imagens escurecidas em dataset/exports/epi-v1-dark/valid/.

Uso:
  python3 preprocessing/experiments/e4_generate_dark_yaml.py
"""

# TODO: Fazer um script generico para apontar para qualquer pasta dentro de dataset/exports/

import yaml
from pathlib import Path

SRC_YAML = Path("dataset/exports/epi-v1/data.yaml")
DEST_DIR = Path("dataset/exports/epi-v1-dark")
DEST_YAML = DEST_DIR / "data.yaml"


def main():
    if not SRC_YAML.exists():
        raise FileNotFoundError(
            f"Não encontrado: {SRC_YAML} "
            "(rode 'dvc pull dataset/exports/epi-v1' primeiro)"
        )
    if not (DEST_DIR / "valid").exists():
        raise FileNotFoundError(
            f"Não encontrado: {DEST_DIR / 'valid'} "
            "(rode e4_generate_dark.py primeiro para gerar as imagens escurecidas)"
        )

    with open(SRC_YAML) as f:
        base = yaml.safe_load(f)

    dark_cfg = {
        "path": str(DEST_DIR.resolve()),
        "train": "valid/images",
        "val": "valid/images",
        "test": "valid/images",
        "names": base["names"],
    }

    with open(DEST_YAML, "w") as f:
        yaml.safe_dump(dark_cfg, f)

    print(f"data.yaml criado em {DEST_DIR}/")
    print(f"  path: {dark_cfg['path']}")
    print(f"  names: {dark_cfg['names']}")


if __name__ == "__main__":
    main()
