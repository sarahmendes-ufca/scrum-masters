"""
Experimento E3: impacto de filtros de suavização na detecção de EPIs.
Testa quatro configurações: sem filtro, Gaussiano 3×3, Gaussiano 5×5,
Mediana 3 e Bilateral (bônus de análise de custo).
"""
import sys, cv2, numpy as np, time
sys.path.insert(0, '.')
from preprocessing.utils.evaluate import evaluate_pipeline


def preproc_gauss_33(frame: np.ndarray) -> np.ndarray:
    f = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return cv2.GaussianBlur(f, (3, 3), sigmaX=0.8)


def preproc_gauss_55(frame: np.ndarray) -> np.ndarray:
    f = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return cv2.GaussianBlur(f, (5, 5), sigmaX=1.5)


def preproc_gauss_77(frame: np.ndarray) -> np.ndarray:
    f = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return cv2.GaussianBlur(f, (7, 7), sigmaX=2.0)


def preproc_median_3(frame: np.ndarray) -> np.ndarray:
    f = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return cv2.medianBlur(f, 3)


def preproc_rgb_only(frame: np.ndarray) -> np.ndarray:
    """Apenas BGR→RGB, sem filtro — para comparação direta."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def benchmark_filter_cost(n_frames: int = 200):
    """Mede o custo em ms de cada filtro em frames 640×480."""
    import numpy as np
    test = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    fns  = [
        ("cvtColor apenas",  lambda f: cv2.cvtColor(f, cv2.COLOR_BGR2RGB)),
        ("GaussianBlur 3×3", lambda f: cv2.GaussianBlur(f, (3,3), 0.8)),
        ("GaussianBlur 5×5", lambda f: cv2.GaussianBlur(f, (5,5), 1.5)),
        ("GaussianBlur 7×7", lambda f: cv2.GaussianBlur(f, (7,7), 2.0)),
        ("medianBlur k=3",   lambda f: cv2.medianBlur(f, 3)),
        ("bilateralFilter",  lambda f: cv2.bilateralFilter(f, 9, 75, 75)),
    ]
    print("\n--- Custo por filtro (média de 200 frames 640×480) ---")
    for nome, fn in fns:
        t0 = time.perf_counter()
        for _ in range(n_frames): fn(test)
        ms = (time.perf_counter() - t0) / n_frames * 1000
        print(f"  {nome:22s}: {ms:.2f} ms/frame")


if __name__ == "__main__":
    print("=" * 65)
    print(" E3 — Filtragem: Gaussiano vs Mediana")
    print("=" * 65)

    results = []
    results.append(evaluate_pipeline(None,            "E3-baseline"))
    results.append(evaluate_pipeline(preproc_rgb_only,"E3-A: RGB apenas (sem filtro)"))
    results.append(evaluate_pipeline(preproc_gauss_33,"E3-B: GaussianBlur 3×3"))
    results.append(evaluate_pipeline(preproc_gauss_55,"E3-C: GaussianBlur 5×5"))
    results.append(evaluate_pipeline(preproc_gauss_77,"E3-D: GaussianBlur 7×7"))
    results.append(evaluate_pipeline(preproc_median_3,"E3-E: medianBlur k=3"))

    benchmark_filter_cost()

    print("\n--- Resumo E3 ---")
    b = results[0]['map50']
    for r in results[1:]:
        print(f"  {r['label']:35s}  mAP={r['map50']:.4f}  delta={r['map50']-b:+.4f}")
