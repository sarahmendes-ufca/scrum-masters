"""
Experimento E1: impacto da conversão de espaço de cor BGR → RGB.
Compara três variantes de pré-processamento de cor.
"""
import sys, cv2, numpy as np
sys.path.insert(0, '.')
from preprocessing.utils.evaluate import evaluate_pipeline


# ── Variante A: sem conversão (passa BGR puro ao modelo) ─────────
def preproc_bgr_raw(frame: np.ndarray) -> np.ndarray:
    """Não faz nenhuma conversão — intencionalmente incorreto."""
    return frame   # BGR — o modelo espera RGB


# ── Variante B: conversão BGR→RGB (correto) ──────────────────────
def preproc_rgb_correct(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


# ── Variante C: inversão manual de canais (equivalente ao B) ─────
def preproc_rgb_flip(frame: np.ndarray) -> np.ndarray:
    """Equivalente a cvtColor, mas usando indexação NumPy."""
    return frame[:, :, ::-1]   # inverte os canais: B,G,R → R,G,B


if __name__ == "__main__":
    print("=" * 65)
    print(" E1 — Impacto da Conversão de Espaço de Cor")
    print("=" * 65)

    results = []
    results.append(evaluate_pipeline(None,               "E1-baseline (Ultralytics padrão)"))
    results.append(evaluate_pipeline(preproc_bgr_raw,    "E1-A: BGR sem conversão"))
    results.append(evaluate_pipeline(preproc_rgb_correct,"E1-B: BGR→RGB (cvtColor)"))
    results.append(evaluate_pipeline(preproc_rgb_flip,   "E1-C: BGR→RGB (NumPy flip)"))

    print("\n--- Resumo E1 ---")
    baseline_map = results[0]['map50']
    for r in results[1:]:
        delta = r['map50'] - baseline_map
        sinal = '+' if delta >= 0 else ''
        print(f"  {r['label']:35s}  mAP@0.5={r['map50']:.4f}  delta={sinal}{delta:.4f}")
