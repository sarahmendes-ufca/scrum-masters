"""Inspeciona e valida um dataset de classificacao no formato de pastas
(train/valid/test

Uso:
    python scripts/inspect_dataset.py --dataset dataset --min-per-class 30
"""

import argparse
from pathlib import Path
# import yaml

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "valid", "test")


def count_images(class_dir: Path) -> int:
    return sum(1 for f in class_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspeciona e valida dataset de classificação (formato pastas)"
    )
    parser.add_argument(
        "--dataset", required=True, help="Caminho para a pasta raiz do dataset"
    )
    parser.add_argument(
        "--min-per-class",
        type=int,
        default=30,
        help="Mínimo de amostras por classe, por split",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Pasta do dataset não encontrada: {dataset_root}")

    print(f"[INFO] Inspecionando dataset em: {dataset_root}")

    problems: list[str] = []
    all_classes: set[str] = set()

    for split in SPLITS:
        split_dir = dataset_root / split
        if not split_dir.exists():
            problems.append(f"Split '{split}' não encontrado em {split_dir}")
            continue

        class_dirs = sorted(d for d in split_dir.iterdir() if d.is_dir())
        if not class_dirs:
            problems.append(f"Nenhuma subpasta de classe encontrada em {split_dir}")
            continue

        print(f"\n[{split.upper()}]")
        for class_dir in class_dirs:
            all_classes.add(class_dir.name)
            n = count_images(class_dir)
            status = "OK" if n >= args.min_per_class else "ABAIXO DO MÍNIMO"
            print(f"  {class_dir.name:20s} {n:4d} imagens  [{status}]")
            if n < args.min_per_class:
                problems.append(
                    f"Classe '{class_dir.name}' no split '{split}' tem {n} imagens "
                    f"(mínimo exigido: {args.min_per_class})"
                )

    # Confere se as mesmas classes existem em todos os splits — um erro comum
    # é uma classe existir em train/ mas não em valid/ ou test/.
    per_split_classes = {
        split: {d.name for d in (dataset_root / split).iterdir() if d.is_dir()}
        for split in SPLITS
        if (dataset_root / split).exists()
    }
    for split, classes in per_split_classes.items():
        missing = all_classes - classes
        if missing:
            problems.append(f"Split '{split}' está sem as classes: {sorted(missing)}")

    print(
        f"\n[INFO] Classes encontradas no total ({len(all_classes)}): {sorted(all_classes)}"
    )

    if problems:
        print("\n[REPROVADO] Problemas encontrados:")
        for p in problems:
            print(f"  - {p}")
        raise SystemExit(1)

    print(
        "\n[APROVADO] Dataset validado — todas as classes presentes em todos os splits, "
        f"com pelo menos {args.min_per_class} imagens cada."
    )


if __name__ == "__main__":
    main()
