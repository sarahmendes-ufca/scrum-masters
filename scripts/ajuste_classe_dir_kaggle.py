# %%
# Ajusta o caminho para a pasta de treino

import os

path = "/home/inversrasp/.cache/kagglehub/datasets/misrakahmed/vegetable-image-dataset/versions/1/Vegetable Images/"

CLASSES_DIR = os.path.join(
    path, "train"
)  # ajuste "train" se o nome exato for diferente
print(os.listdir(CLASSES_DIR))  # deve mostrar as classes: apple, banana, beetroot, ...
