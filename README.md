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
tcc-scrum-masters/                                  # Diretório raiz do projeto (TCC)
├───.dvc/                                           # Configurações do Data Version Control (DVC)
|  ├───.gitignore                                   # Ignora arquivos de cache locais do DVC
│  └─── config                                      # Configuração do repositório remoto de dados
├───.github/
│   └───workflows/
│       └───edge-deploy.yml                         # Pipeline CI/CD (GitHub Actions) para deploy no Edge (ex: Raspberry Pi)
├───app/                                            # Aplicação backend (FastAPI)
|  ├───__init__.py                                  # Inicializador do módulo Python
|  ├───main.py                                      # Ponto de entrada da API, rotas e inicialização
|  ├───model.py                                     # Lógica de carregamento e inferência do modelo YOLOv8
|  ├───requirements.txt                             # Dependências específicas da API
│  └─── schemas.py                                  # Modelos de validação de dados usando Pydantic
├───client/                                         # Aplicação cliente para interagir com a API
|  ├─── client.py                                   # Script principal que consome os endpoints de detecção
│  └─── requirements.txt                            # Dependências específicas do cliente
├───dataset/                                        # Diretório de dados (gerenciado via DVC)
|  ├─── raw/                                        # Imagens e anotações brutas originais
│  └───exports/                                     # Datasets pré-processados/formatados exportados
├───models/                                         # Pesos do modelo treinado
│    ├─── best.pt                                   # Melhores pesos do YOLOv8 (deploy principal)
│    └─── last.pt                                   # Últimos pesos salvos do treinamento
├───preprocessing/                                  # Pipeline de visão computacional (OpenCV)
│    ├─── experiments/                              # Scripts para testes de pré-processamento
│    │  ├───e1_color_space.py                       # Experimentos de conversão de espaço de cores
│    │  ├───e1_visualize.py                         # Ferramenta para visualização das transformações
│    │  ├───e2_resize.py                            # Redimensionamento de imagens para otimização
│    │  ├───e3_filters.py                           # Aplicação de filtros de suavização/ruído
│    │  ├───e4_contrast.py                          # Ajuste de contraste das imagens
│    │  ├───e4_generate_dark.py                     # Aumento de dados (data augmentation) para baixa luz
│    │  └───run_baseline.py                         # Avaliação do modelo base (baseline)
│    └─── utils/                                    # Funções utilitárias de pré-processamento
│       ├───__init__.py                             # Inicializador do módulo de utilitários
│       ├───evaluate.py                             # Cálculo de métricas de qualidade de imagem
│       └───letterbox.py                            # Algoritmo de letterbox (mantém proporção adicionando bordas)
├───scripts/                                        # Scripts de automação, MLOps e manipulação de datasets
|  ├─── ajuste_classe_dir_kaggle.py                 # Corrige estrutura de classes baixadas do Kaggle
|  ├─── baixa_dataset_kaggle.py                     # Script para download automatizado via API do Kaggle
|  ├─── deploy.sh                                   # Shell script para facilitar o deploy no ambiente Edge
|  ├─── encontrar_nome-projeto_roboflow.py          # Busca IDs/nomes na API do Roboflow
|  ├─── export_hoboflow.py                          # Exporta dataset formatado a partir do Roboflow
|  ├─── generate_dark_dataset_epi-v1.py             # Script final para geração de dataset escuro (foco em EPI)
|  ├─── inspect_dataset.py                          # Ferramenta de auditoria/verificação do dataset
|  ├─── upload_roboflow_dataset_raw.py              # Script para envio automatizado de dados ao Roboflow
|  ├─── validate_model.py                           # Executa testes de validação pós-treinamento
│  └───verificar_estrutura_dataset_kaggl            # Checa integridade de pastas vindas do Kaggle
├───stream/                                         # Módulo de captura e transmissão de vídeo (Edge)
|  ├───__init__.py                                  # Inicializador do módulo de streaming
|  ├───capture_frames.py                            # Script base para capturar frames da câmera (OpenCV)
|  ├───mjpeg_server.py                              # Servidor leve para stream de vídeo em MJPEG
|  ├───raw_server.py                                # Servidor para envio de frames não comprimidos
|  ├───v1_naive.py                                  # Captura síncrona padrão (versão não otimizada)
|  ├───v2_threaded.py                               # Captura assíncrona (usa threads p/ destravar I/O)
|  ├───v3_optimized.py                              # Captura de alta performance (para Raspberry/ESP32)
│  └───video_visualize.py                           # Exibe o stream processado na tela local
├───tests/                                          # Testes unitários e de integração (pytest)
│   └───assets/                                     # Arquivos estáticos usados nos testes
│   │   └───zidane.jpg                              # Imagem padrão de teste do ecossistema YOLO
|   ├───test_api.py                                 # Testes dos endpoints da FastAPI
│   └───test_preprocessor.py                        # Testes unitários das funções em preprocessing/
├─── Dockerfile.api                                 # Instruções para conteinerizar a FastAPI
├─── Dockerfile.client                              # Instruções para conteinerizar a aplicação cliente
├─── README.md                                      # Documentação oficial do projeto
├─── dataset.dvc                                    # Arquivo de metadados do DVC apontando para os dados reais
├─── docker-compose.yaml                            # Orquestração do Docker para subir API e Cliente juntos
├─── modelo_backup_yolov8n.pt                       # Backup dos pesos iniciais (nano) do YOLOv8
├─── ruff.toml                                      # Configurações do Ruff (linter/formatador de código Python)
├─── teste_gpu.py                                   # Script rápido para checar acesso à GPU/CUDA ou NPU
├─── train_cepi.py                                  # Script principal de treinamento customizado (EPIs)
├─── yolov8n-cls.pt                                 # Pesos pré-treinados para Classificação (YOLOv8 Nano)
└─── yolov8n.py                                     # Script rápido de teste ou download do YOLOv8
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
sudo apt update && sudo apt upgrade -y
sudo apt install -y curl wget git jq tree python3-pip python3-venv libcamera-tools

```

---

#### Configuração do Docker e Docker Compose

Para isolar a aplicação em containers leves e evitar desgaste do cartão SD faça as seguintes instalações:

**Instalação do Docker:**
```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

```


**Permissão de Usuário sem `sudo`:**
```bash
sudo usermod -aG docker $USER

```


(Efetue logout e login novamente para aplicar a alteração).


**Validação do Docker:**
```bash
docker version
docker run --rm hello-world

```

---

#### Instalação das Dependências Python e MLOps (Ambiente de Desenvolvimento)

Caso precise rodar testes ou validações diretamente no host ou em um ambiente virtual Python (`venv`):

**Crie e Ative um Ambiente Virtual:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

```


**Instalação das Bibliotecas de Aprendizado e IA:**

Para instalar as bibliotecas utilizadas basta rodar o comando

```bash
pip install -r app/requirements.txt

```
**Construa as Imagens Multi-Arquitetura (ARM64) e Inicialização dos Serviços:**
```bash
docker compose build
docker compose up -d

```

**Por fim, Faça a Verificação do Status dos Serviços:**
```bash
docker compose ps
curl -f http://localhost:8000/health

```
---
## 6. Comandos e Procedimento para Execução

### Testando a PoC:

Para testar a nossa PoC, com o repositório e todas as dependências instaladas, execute:

Para conectar ao seu dispositivo:
```bash
ssh <ip do seu raspberry pi 5>

```
Para acessar o projeto:

```bash
cd <caminho da pasta do projeto>
```
Para realizar a inferência da imagem de uma garrafa a partir de um vídeo:
```bash
python3 stream/video_visualize.py --input <caminho do vídeo no seu dispositivo> --output <nome do vídeo após a inferência> --model models/best.pt --no-display
```

Depois rode novamente cd com o caminho da pasta do projeto caso esteja em outra pasta e ls para verificar se o arquivo foi criado. Após isso ao assistir o vídeo você verá a inferência e sua classificação sendo exibidas no momento da execução.

Para fazer download do vídeo para teste, acesse: https://drive.google.com/drive/folders/1UVhUWeMSqkuLZVQD1KXtWIFyr2M3RvUF?usp=sharing

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
