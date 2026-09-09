import sys
sys.path.insert(0, '.')
from preprocessing.utils.evaluate import evaluate_pipeline

baseline = evaluate_pipeline(preprocess_fn=None, label="baseline (sem preproc)")
print(f"\nBaseline mAP@0.5 = {baseline['map50']:.4f}")
print("Anote este valor — ele é a referência de todos os experimentos.")
