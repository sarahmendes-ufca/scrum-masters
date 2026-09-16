import time
import cv2
from flask import Flask, Response
from picamera2 import Picamera2
from edge_impulse_linux.image import ImageImpulseRunner

app = Flask(__name__)
caminho_modelo = "./modelfile.eim"

# Inicializa a câmera e o modelo globalmente para o servidor web
picam2 = Picamera2()
config = picam2.create_video_configuration(
    main={"size": (640, 480), "format": "RGB888"}
)
picam2.configure(config)
picam2.start()

runner = ImageImpulseRunner(caminho_modelo)
model_info = runner.init()
print(f"Modelo carregado: {model_info['project']['name']}")


def gerar_frames():
    while True:
        inicio_ciclo = time.time()

        frame_rgb = picam2.capture_array()
        if frame_rgb is None:
            break

        features, cropped = runner.get_features_from_image(frame_rgb)
        res = runner.classify(features)

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        if "result" in res and "bounding_boxes" in res["result"]:
            for box in res["result"]["bounding_boxes"]:
                x, y, w, h = box["x"], box["y"], box["width"], box["height"]
                label = box["label"]
                score = box["value"]
                cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    frame_bgr,
                    f"{label} ({score:.2f})",
                    (x, max(y - 10, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

        # Codifica o frame em JPEG para transmissão web
        _, buffer = cv2.imencode(".jpg", frame_bgr)
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

        # Intervalo fixo de 100ms
        tempo_gasto = time.time() - inicio_ciclo
        tempo_espera = 0.1 - tempo_gasto
        if tempo_espera > 0:
            time.sleep(tempo_espera)


@app.route("/")
def index():
    return "<html><body><h2>Edge Impulse - Tempo Real (100ms)</h2><img src='/video_feed'></body></html>"


@app.route("/video_feed")
def video_feed():
    return Response(
        gerar_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    # Roda o servidor acessível na rede local na porta 1337
    app.run(host="0.0.0.0", port=1337, debug=False)
