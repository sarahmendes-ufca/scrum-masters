Teste do yolo edge api em um raspberry pi5
=======
# 📦 TCC CENÁRIO 3: Visão Computacional para Detecção de Defeitos em Embalagens de Produtos

### 👤 Identificação da Equipe
- **scrum-masters**
- **GitHub dos Membros:** 
  - José Dhonatan Fernandes de Almeida — [`@sudo-invers`](https://github.com/sudo-invers)
  - Letícia Maria dos Santos Dias — [`@leticia-software-engineer`](https://github.com/leticia-software-engineer)
  - Sarah Mendes Teles — [`@sarahmendes-ufca`](https://github.com/sarahmendes-ufca)

---

## 📌 Sumário
1. [Visão Geral da Solução](#1-visão-geral-da-solução)
2. [Arquitetura do Sistema](#2-arquitetura-do-sistema)
3. [Componentes Utilizados na Simulação](#3-componentes-utilizados-na-simulação)
4. [Pré-requisitos e Recursos](#4-pré-requisitos-e-recursos)
5. [Dependências e Procedimento de Instalação](#5-dependências-e-procedimento-de-instalação)
6. [Comandos e Procedimento para Execução](#7-comandos-e-procedimento-para-execução)
7. [Confirmação do Resultado](#8-confirmação-do-resultado)
8. [Diagrama de blocos](#9-diagrama-de-blocos)
9. [Comentários Adicionais](#10-comentários-adicionais)

---

## 1. Visão Geral da Solução

No ambiente fabril, a etapa de rotulagem na fase final da confecção de produtos frequentemente apresenta gargalos operacionais. Embora seja um processo automatizado, os equipamentos estão sujeitos a falhas. A eventual transferência de produtos defeituosos da esteira de produção para a distribuidora e, consequentemente, para o consumidor final, acarreta insatisfação, prejuízos à credibilidade da marca e perda da confiança na linha de produtos. Adicionalmente, o rótulo constitui a principal fonte de dados para o consumidor, contendo informações essenciais como instruções de uso e prazo de validade; a ausência ou baixa legibilidade desses itens representa um problema crítico de qualidade e conformidade.

A inserção de inspeção humana 100% manual após a rotulagem, para checar a presença e o posicionamento das etiquetas, seria uma alternativa direta. No entanto, essa abordagem mostra-se financeiramente inviável e pouco operacional, dada a alta exigência de tempo, mão de obra e custos associados.

Diante desse cenário, este trabalho propõe uma solução prática, segura e de menor custo por meio do uso de visão computacional. Trata-se do desenvolvimento de um sistema baseado em inteligência artificial para a verificação e classificação de rótulos em tempo real. A solução permite a emissão de alertas inteligentes e a integração com mecanismos automatizados de triagem — como braços robóticos —, garantindo a imediata remoção dos itens com defeitos identificados.

---

## 2️. Arquitetura do Sistema

```text
tcc-scrum-masters/
├───.dvc
|  ├───.gitignore
│  └─── config
├───.github
│   └───workflows
│       └───edge-deploy.yml
├───app
|   ├───__init__.py
|   ├───main.py
|   ├───model.py
|   ├───requirements.txt
│   └─── schemas.py
├───client
|   ├─── client.py
│   └─── requirements.txt
├───files
│   └───md5
│       ├───95
│       └───a9
├───models
├───scripts
├───stream
├───tests
    └───assets
├─── .docker-compose.yaml.swp
├─── Dockerfile.api
├─── Dockerfile.client
├─── README.md
├─── dataset.dvc
├─── docker-compose.yaml
├─── modelo_backup_yolov8n.pt
├─── ruff.toml
├─── teste_gpu.py
├─── train_cepi.py
├─── yolov8n-cls.pt
└─── yolov8n.py
```
## 3. Componentes utilizados 

- Raspberry pi 5 ou microcomputador similar
- Fonte de alimentação USB-C 27 W para Raspberry pi 5
- Módulo câmera Raspberry pi V1.3, 5 MP, interface CSI
- Cabo adaptador CSI para câmera do Raspberry pi
- Cartão de Memória MicroSDXC 128 GB, classificação A2/V30
---
## 4. Pré-requisitos e Recursos

### Hardware, Dispositivos e Infraestrutura

* **Placa Principal:** Raspberry Pi 5 (8 GB de RAM)  (ARM64) com conexão à internet.. (Plataformas alternativas compatíveis/projetadas: Gigabyte GA-SBCAP3350, Nvidia Jetson Nano, Banana Pi e ESP-32 com módulo câmera).

* **Câmera:** Módulo Câmera Raspberry Pi V1.3 (5 MP) com interface e cabo adaptador CSI.


* **Alimentação:** Fonte oficial USB-C 27 W para Raspberry Pi 5.


* **Armazenamento:** Cartão microSD (com sistema de arquivos Overlay FS para restrição de escrita).


### Software de Sistema e Drivers de Captura

* **Sistema Operacional:** Raspberry Pi OS 64-bit (Debian Bookworm/Trixie).

* **Drivers & Utilitários de Vídeo:** V4L2 (Video4Linux2), libcamera e pacote rpicam-apps (rpicam-vid, rpicam-still).

---
## 5. Dependências e procedimento de instalação


### Linguagem e Frameworks Web / Servidores

* **Linguagem Base:** Python 3.11.


* **Servidor e Framework REST:** FastAPI (API REST de inferência) e Uvicorn (servidor ASGI).


* **Streaming de Vídeo:** Flask (micro-framework utilitário para stream MJPEG).


* **Validação de Dados:** Pydantic.



### Visão Computacional, IA e Processamento Matemático

* **Detecção/IA:** Ultralytics YOLOv8 (modelo yolov8n.pt).


* **Framework DL:** PyTorch e Torchvision.


* **Processamento de Imagem:** OpenCV (opencv-python-headless), Pillow (PIL) e NumPy.



### MLOps, Engenharia de Dados e Observabilidade

* **Gestão de Dataset:** Roboflow (SaaS para anotação/augmentation).


* **Versionamento:** DVC (Data Version Control para modelos/dados), Git e GitHub.


* **Visualização/Dashboards:** Grafana.



### Conteinerização, Rede e CI/CD

* **Contêineres:** Docker e Docker Compose.


* **Compilação Cruzada:** Docker Buildx e QEMU.


* **Registro de Imagens:** GitHub Container Registry (GHCR).


* **Rede / VPN Mesh:** Tailscale (acesso SSH e integração CI/CD).


* **CI/CD:** GitHub Actions.


* **Testes e Qualidade:** Pytest, TestClient (FastAPI), Ruff (linter) e Logs Estruturados em JSON.

### Procedimento de Instalação e Configuração

* **Instale Pacotes do Sistema Host:**
```bash
sudo apt update && sudo apt upgrade -y[cite: 6]
sudo apt install -y curl wget git jq tree python3-pip python3-venv libcamera-tools[cite: 1, 2, 3]

```

---

#### 2. Configuração do Docker e Docker Compose

Para isolar a aplicação em containers leves e evitar desgaste do cartão SD faça as seguintes instalações:

1. **Instalação do Docker:**
```bash
curl -fsSL https://get.docker.com -o get-docker.sh[cite: 2]
sudo sh get-docker.sh[cite: 2]

```


2. **Permissão de Usuário sem `sudo`:**
```bash
sudo usermod -aG docker $USER[cite: 2]

```


(Efetue logout e login novamente para aplicar a alteração).


3. **Validação do Docker:**
```bash
docker version[cite: 2]
docker run --rm hello-world[cite: 2]

```

---

#### Instalação das Dependências Python e MLOps (Ambiente de Desenvolvimento)

Caso precise rodar testes ou validações diretamente no host ou em um ambiente virtual Python (`venv`):

**Crie e Ative um Ambiente Virtual:**
```bash
python3 -m venv venv[cite: 6]
source venv/bin/activate[cite: 6]
pip install --upgrade pip[cite: 6]

```


**Instalação das Bibliotecas de Aprendizado e IA:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu[cite: 1]
pip install ultralytics opencv-python-headless pillow numpy[cite: 1, 2]

```


**Instalação de Frameworks Web e Utilitários:**
```bash
pip install fastapi "uvicorn[standard]" httpx pydantic flask[cite: 1, 2, 3]

```

**Instalação de Ferramentas de MLOps, Testes e Qualidade:**
```bash
pip install dvc "dvc[ssh]" pytest ruff prometheus-client roboflow pyyaml[cite: 1, 3, 6]

```

#### Procedimento de Instalação e Execução via Docker Compose (Stack Completa)

**Clone o Repositório do Projeto:**
```bash
git clone https://github.com/<seu-usuario>/yolo-edge-api.git[cite: 1]
cd yolo-edge-api[cite: 1]

```

**Recupere Pesos e Datasets Versionados com DVC:**
```bash
dvc pull[cite: 1]

```


**Construa as Imagens Multi-Arquitetura (ARM64) e Inicialização dos Serviços:**
```bash
docker compose build[cite: 2]
docker compose up -d[cite: 2]

```

**Por fim, Faça a Verificação do Status dos Serviços:**
```bash
docker compose ps[cite: 1, 2]
curl -f http://localhost:8000/health[cite: 1, 2]

```
---
## 6. Comandos e Procedimento para Execução
---
## 7. Cofirmação do resultado
---
## 8. Diagrama de blocos

O diagrama de blocos desenvolvido ilustra as entradas, processamento e saídas do nosso sistema, considerando aspectos de hardware e software. A plataforma utilizada para desenvolvê-lo foi o Miro.

Como informações de entrada haverão apenas as capturas de imagem realizadas pela câmera do Raspberry pi.
Já no processamento são consideradas todas as operações realizadas após a captura até a formulação de dados de saída, sendo a inferência, a classificação e o cálculo das métricas, as principais operações dessa etapa.
Por fim, como saída temos as informações expressas no dashboard, a notificação de email em caso de alto índice de passagem de itens defeituosos, a documentação do FastAPI e o sinal de alerta para desvio automático de itens a ser processado por outro dispositivo embarcado. 

<img width="1029" height="1518" alt="Meu primeiro board" src="https://github.com/user-attachments/assets/a050302a-7d85-4d7e-8f2d-2a44df469123" />

---
## 9. Comentários Adicionais
---
