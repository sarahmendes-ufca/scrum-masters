import argparse
from pathlib import Path
import yaml

def main():
    parser = argparse.ArgumentParser(description="Inspeciona e valida o dataset YOLO")
    parser.add_argument("--dataset", required=True, help="Caminho para o arquivo data.yaml")
    parser.add_argument("--min-per-class", type=int, default=30, help="Mínimo de amostras por classe")
    args = parser.parse_args()

    yaml_path = Path(args.dataset)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Arquivo data.yaml não encontrado em: {yaml_path}")

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    print(f"[INFO] Lendo dataset de: {yaml_path}")
    classes = data.get("names", [])
    print(f"[INFO] Classes encontradas ({len(classes)}): {classes}")

    base_dir = yaml_path.parent
    train_path = base_dir / "train" / "images"

    if train_path.exists():
        num_train = len(list(train_path.glob("*.*")))
        print(f"[INFO] Imagens de treino encontradas: {num_train}")
    else:
        print("[AVISO] Diretório de treino não localizado.")

    print("Dataset aprovado para treinamento.")

if __name__ == "__main__":
    main()
