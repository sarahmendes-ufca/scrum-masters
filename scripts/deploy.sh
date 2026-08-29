#!/bin/bash
# scripts/deploy.sh
# Executa no Raspberry Pi via SSH pelo pipeline de CI/CD.
# Faz pull da nova imagem, reinicia o serviço e valida o health check.
# Em caso de falha, reverte para a imagem anterior automaticamente.


set -euo pipefail


DEPLOY_PATH="${DEPLOY_PATH:-~/yolo-edge-api}"
HEALTH_URL="http://localhost:8000/health"
HEALTH_RETRIES=6
HEALTH_WAIT=10


echo "========================================"
echo " Deploy — $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"


cd "$DEPLOY_PATH"


# ── Salva a imagem atual para possível rollback ──────────────
PREVIOUS=$(docker inspect yolo-api \
    --format '{{.Config.Image}}' 2>/dev/null || echo "none")
echo "[INFO] Imagem atual: $PREVIOUS"


# ── Baixa a nova imagem ──────────────────────────────────────
echo "[1/4] Baixando nova imagem..."
docker compose pull


# ── Sobe a nova versão ───────────────────────────────────────
echo "[2/4] Iniciando nova versão..."
docker compose up -d


# ── Aguarda o serviço estabilizar ────────────────────────────
echo "[3/4] Aguardando health check ($((HEALTH_RETRIES * HEALTH_WAIT))s max)..."
SUCCESS=false
for i in $(seq 1 $HEALTH_RETRIES); do
    sleep $HEALTH_WAIT
    if curl -sf "$HEALTH_URL" > /dev/null 2>&1; then
        SUCCESS=true
        break
    fi
    echo "  Tentativa $i/$HEALTH_RETRIES falhou, aguardando..."
done


# ── Avalia o resultado ───────────────────────────────────────
if [ "$SUCCESS" = true ]; then
    echo "[4/4] Health check OK"
    NEW=$(docker inspect yolo-api --format '{{.Config.Image}}' 2>/dev/null)
    echo ""
    echo "[OK] Deploy bem-sucedido: $NEW"
    exit 0
else
    echo "[ERRO] Health check falhou após $((HEALTH_RETRIES * HEALTH_WAIT))s"
    if [ "$PREVIOUS" != "none" ]; then
        echo "[ROLLBACK] Revertendo para: $PREVIOUS"
        docker compose down
        IMAGE=$PREVIOUS docker compose up -d
        echo "[ROLLBACK] Concluído. Serviço restaurado."
    else
        echo "[AVISO] Sem imagem anterior para rollback."
    fi
    exit 1
fi
