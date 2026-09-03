python3 -c "
from roboflow import Roboflow
rf = Roboflow(api_key='<SUA_API_KEY>')  # Settings > API Keys
project = rf.workspace().project('epi-detection-rpi5')
project.upload('dataset/raw/', num_retry_uploads=3)
"
