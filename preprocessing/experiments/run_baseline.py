import sys
sys.path.insert(0, '.')
from preprocessing.utils.evaluate import evaluate_pipeline

baseline = evaluate_pipeline(preprocess_fn=None, label="baseline (sem preproc)")
print(f"\nBaseline mAP@0.5 = {baseline['map50']:.4f}")
print("Anote este valor — ele é a referência de todos os experimentos.")

cd ~/yolo-edge-api
python preprocessing/experiments/run_baseline.py

# Saída esperada (valores aproximados para yolov8n + epi-v1):
# [baseline (sem preproc)         ]  mAP@0.5=0.6120  mAP@0.5:0.95=0.3890
# Baseline mAP@0.5 = 0.6120
