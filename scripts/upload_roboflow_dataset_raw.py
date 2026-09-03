import os

from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv()  # lê o .env da pasta atual
API_KEY = os.getenv("ROBOFLOW_API_KEY")

rf = Roboflow(api_key=API_KEY)
# Se a api_key falhar, não vai encontar, então não tem problema ser hardcoded
project = rf.workspace().project("epi-detection-rpi5-oenw9")
project.upload("dataset/raw/", num_retry_uploads=3)
