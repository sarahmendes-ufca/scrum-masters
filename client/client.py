import base64
import json
import os
import time
from pathlib import Path

import httpx

API_URL = os.getenv("API_URL", "http://localhost:8000")
IMAGES_DIR = Path("/client/images")
OUTPUT_DIR = Path("/client/output")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def encode_image(path: Path) -> str:
    """Lê um arquivo de imagem e retorna a string base64."""
    return base64.b64encode(path.read_bytes()).decode()

def wait_for_api(max_retries: int = 10, delay: float = 3.0):
    """Aguarda a API ficar disponível antes de enviar requisições."""

    for attempt in range(max_retries):
        try:
            r = httpx.get(
                f"{API_URL}/health",
                timeout=5,
            )

            if r.status_code == 200:
                data = r.json()

                print(
                    f"[OK] API disponível | modelo: "
                    f"{data['model_name']}"
                )

                return True

        except httpx.ConnectError:
            pass

        print(
            f"[...] Aguardando API "
            f"({attempt + 1}/{max_retries})..."
        )

        time.sleep(delay)

    raise RuntimeError(
        "API não ficou disponível a tempo."
    )


def run_single_inference(
    image_path: Path,
    confidence: float = 0.25,
):
    """Envia uma imagem e imprime as detecções recebidas."""

    print(
        f"\n─── Inferência: {image_path.name} ───"
    )

    payload = {
        "image_base64": encode_image(image_path),
        "confidence": confidence,
        "model_name": "yolov8n.pt",
    }

    response = httpx.post(
        f"{API_URL}/predict",
        json=payload,
        timeout=60,
    )

    response.raise_for_status()

    data = response.json()

    print(
        f" Tempo de inferência : "
        f"{data['inference_ms']} ms"
    )

    print(
        f" Resolução : "
        f"{data['image_width']}x"
        f"{data['image_height']} px"
    )

    print(
        f" Detecções "
        f"({len(data['detections'])}):"
    )

    for det in data["detections"]:
        print(
            f" • {det['label']:20s} "
            f"conf={det['confidence']:.2f}"
        )

    # Salva o JSON de resultado
    out_file = OUTPUT_DIR / f"{image_path.stem}_result.json"

    out_file.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
        )
    )

    print(
        f" Resultado salvo em : {out_file}"
    )


def run_batch_inference(
    image_paths: list,
    confidence: float = 0.25,
):
    """Envia múltiplas imagens em uma única requisição batch."""

    print(
        f"\n─── Batch: {len(image_paths)} imagens ───"
    )

    payload = {
        "images_base64": [
            encode_image(p)
            for p in image_paths
        ],
        "confidence": confidence,
    }

    response = httpx.post(
        f"{API_URL}/predict/batch",
        json=payload,
        timeout=120,
    )

    response.raise_for_status()

    data = response.json()

    print(
        f" Total batch : "
        f"{data['total_inference_ms']} ms"
    )

    for i, r in enumerate(data["results"]):
        print(
            f" Imagem {i + 1}: "
            f"{len(r['detections'])} detecções em "
            f"{r['inference_ms']} ms"
        )


if __name__ == "__main__":
    wait_for_api()

    images = (
        sorted(IMAGES_DIR.glob("*.jpg"))
        + sorted(IMAGES_DIR.glob("*.png"))
    )

    if not images:
        print(
            "[AVISO] Nenhuma imagem encontrada em "
            "/client/images/"
        )

    else:
        # Inferência individual na primeira imagem
        run_single_inference(images[0])

        # Batch com todas as imagens disponíveis
        if len(images) > 1:
            run_batch_inference(images)

        # Consulta métricas ao final
        metrics = httpx.get(
            f"{API_URL}/metrics"
        ).json()

        print("\n─── Métricas da API ───")

        print(
            f" Total de requisições : "
            f"{metrics['total_requests']}"
        )

        print(
            f" Latência média : "
            f"{metrics['avg_inference_ms']} ms"
        )

