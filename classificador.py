import time
import cv2
from picamera2 import Picamera2
from edge_impulse_linux.image import ImageImpulseRunner
from prometheus_client import start_http_server, Counter

# Inicia o servidor HTTP de métricas na porta 8000 (para o Alloy ler)
start_http_server(8000)

# Cria as métricas para o Grafana
PRODUTOS_AVALIADOS = Counter(
    "classificador_produtos_avaliados_total", "Total de produtos avaliados"
)
CLASSIFICACAO_LOGS = Counter(
    "classificador_logs_por_classificacao_total",
    "Logs por tipo de classificação",
    ["classificacao"],
)

caminho_modelo = "./modelfile.eim"


def main():
    # Inicializa a câmera nativa da Raspberry Pi 5
    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    print("Câmera inicializada e servidor de métricas rodando na porta 8000.")

    # Lista de rótulos que acionam o desvio automático
    rotulos_alerta = ["rotulo ausente", "rotulo mal posicionado", "rotulo danificado"]

    with ImageImpulseRunner(caminho_modelo) as runner:
        try:
            model_info = runner.init()
            print(f"Modelo carregado: {model_info['project']['name']}")
            print("Iniciando inferência em tempo real e exibindo logs no terminal...")

            while True:
                inicio_ciclo = time.time()

                # Captura o frame diretamente em RGB
                frame_rgb = picam2.capture_array()

                if frame_rgb is None:
                    print("Falha ao capturar imagem da câmera.")
                    break

                # Extrai características e executa a classificação
                features, cropped = runner.get_features_from_image(frame_rgb)
                res = runner.classify(features)

                if "result" in res:
                    # Tratamento para detecção de objetos (bounding boxes)
                    if "bounding_boxes" in res["result"]:
                        caixas = res["result"]["bounding_boxes"]
                        print(f"Detectado: {caixas}")

                        for obj in caixas:
                            label = obj.get("label", "").lower()
                            confianca = obj.get("value", 0)

                            # Incrementa as métricas do Grafana
                            PRODUTOS_AVALIADOS.inc()
                            CLASSIFICACAO_LOGS.labels(classificacao=label).inc()

                            # Verifica se o rótulo está na lista de alertas
                            if label in rotulos_alerta:
                                print(
                                    f"⚠️ ALERTA: Desvio automático! (Motivo: {label} com {confianca:.2f} de confiança)"
                                )

                    # Tratamento para classificação geral de imagem
                    elif "classification" in res["result"]:
                        classificacoes = res["result"]["classification"]
                        print(f"Resultado: {classificacoes}")

                        for label, score in classificacoes.items():
                            PRODUTOS_AVALIADOS.inc()
                            CLASSIFICACAO_LOGS.labels(classificacao=label).inc()

                            if label.lower() in rotulos_alerta and score > 0.5:
                                print(
                                    f"⚠️ ALERTA: Desvio automático! (Motivo: {label} - {score:.2f})"
                                )

                # Pausa exata de 100 milissegundos
                tempo_gasto = time.time() - inicio_ciclo
                tempo_espera = 0.1 - tempo_gasto
                if tempo_espera > 0:
                    time.sleep(tempo_espera)

        except KeyboardInterrupt:
            print("\nEncerrado pelo usuário.")
        finally:
            picam2.stop()


if __name__ == "__main__":
    main()
