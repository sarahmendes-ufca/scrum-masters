import kagglehub

# Baixa a versão mais recente do dataset
path = kagglehub.dataset_download(
    "misrakahmed/vegetable-image-dataset"
)  # Dataset de tipos de vegetais
print("Path to dataset files:", path)
