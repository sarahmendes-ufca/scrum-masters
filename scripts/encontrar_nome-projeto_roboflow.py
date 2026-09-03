import os

from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv()  # lê o .env da pasta atual
API_KEY = os.getenv("ROBOFLOW_API_KEY")

rf = Roboflow(api_key=API_KEY)
ws = rf.workspace()
for p in ws.projects():
    print(p)
