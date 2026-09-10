import os

path = "/home/inversrasp/.cache/kagglehub/datasets/misrakahmed/vegetable-image-dataset/versions/1/"

for raiz, pastas, arquivos in os.walk(path):
    nivel = raiz.replace(path, "").count(os.sep)
    indent = "  " * nivel
    print(f"{indent}{os.path.basename(raiz)}/")
    if nivel < 2:
        for arq in arquivos[:3]:
            print(f"{indent}  {arq}")
