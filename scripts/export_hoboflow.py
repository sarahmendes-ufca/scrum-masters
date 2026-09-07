# Exporta o dataset via API do Roboflow para a pasta dataset/exports/

import os
from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv()

api_key = os.getenv("ROBOFLOW_API_KEY")
project_name = os.getenv("PROJECT_NAME")
version_number = int(os.getenv("ROBOFLOW_VERSION", "1"))

if not api_key:
    raise RuntimeError("ROBOFLOW_API_KEY não encontrada nas variáveis de ambiente")
if not project_name:
    raise RuntimeError("ROBOFLOW_PROJECT não encontrada nas variáveis de ambiente")

print("loading Roboflow workspace...")
rf = Roboflow(api_key=api_key)

print("loading Roboflow project...")
project = rf.workspace().project(project_name)
version = project.version(version_number)

dataset = version.download(
    model_format="yolov8",
    location=f"dataset/exports/{project_name}-v{version_number}",
    overwrite=True,
)

print("Export concluído:", dataset.location)
