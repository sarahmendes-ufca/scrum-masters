# 📦 TCC CENÁRIO 3: Visão Computacional para Detecção de Defeitos em Embalagens de Produtos

### 👤 Identificação da Equipe
- **scrum-masters**
- **repositório: https://github.com/sarahmendes-ufca/scrum-masters**
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

---

> [!WARNING]
> Sempre que em algum comando estiver escrito os simbolo "<" ou ">", não inclua eles no comando.
> Exemplo: http://<ip do raspberry pi>:8000
> Deve ser escrito quando usado assim (usado um ip aleatorio como exemplo): http://100.95.153.33:8000
> Substituido <ip do raspeberry pi> por 100.95.153.33
---

## 1. Visão Geral da Solução

No ambiente fabril, a etapa de rotulagem na fase final da confecção de produtos frequentemente apresenta gargalos operacionais. Embora seja um processo automatizado, os equipamentos estão sujeitos a falhas. A eventual transferência de produtos defeituosos da esteira de produção para a distribuidora e, consequentemente, para o consumidor final, acarreta insatisfação, prejuízos à credibilidade da marca e perda da confiança na linha de produtos. Adicionalmente, o rótulo constitui a principal fonte de dados para o consumidor, contendo informações essenciais como instruções de uso e prazo de validade; a ausência ou baixa legibilidade desses itens representa um problema crítico de qualidade e conformidade.

A inserção de inspeção humana 100% manual após a rotulagem, para checar a presença e o posicionamento das etiquetas, seria uma alternativa direta. No entanto, essa abordagem mostra-se financeiramente inviável e pouco operacional, dada a alta exigência de tempo, mão de obra e custos associados.

Diante desse cenário, este trabalho propõe uma solução prática, segura e de menor custo por meio do uso de visão computacional. Trata-se do desenvolvimento de um sistema baseado em inteligência artificial para a verificação e classificação de rótulos em tempo real. A solução permite a emissão de alertas inteligentes e a integração com mecanismos automatizados de triagem — como braços robóticos —, garantindo a imediata remoção dos itens com defeitos identificados.

---

## 2️. Arquitetura do Sistema

{Adicionar nova arquitetura (TODO: tree --gitignore)}

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

> Para saber porque cada dependencia é necessária, acesse {TODO: Adicionar pagina da wiki com dependencias e explicações depois}

#### Instalando as dependencias do sistema
No raspberry pi, em um terminal:
```bash
sudo apt update
sudo apt install -y python3-opencv python3-picamera2 portaudio19-dev
```

#### Instalando as dependencias do projeto
No raspberry pi, crie um ambiente virtual:
```bash
python3 -m venv .venv
```
e depois ative ele:
```bash
source .venv/bin/activate
```

## 6. Comandos e Procedimento para Execução

### Executando:
#### Visualização pelo terminal:

com o repositório, com todas as dependências instaladas e dentro do ambiente virtual (venv), execute:

```bash
python3 classification.py
```
o que irá iniciar o programa, e o log poderá ser visto no terminal.

#### Visualizando pelo navegador:

Para conectar ao seu dispositivo:

```bash
python3 classification_interface.py
```
depois, no seu navegador web de preferencia, na barra de url, digite:

```
http://<ip do raspberry pi>:8000
```
- Substitua <ip do seu raspberry pi>, pelo ip real do seu raspberry pi

#### Utilizar um vídeo pré-gravado para visualizar
Se preferir utilizar um vídeo pré-feito para testar, no terminal:
```bash
cd <caminho da pasta do projeto>
```
Depois:

```bash
python3 stream/video_visualize.py --input <caminho do vídeo no seu dispositivo> --output <nome do vídeo após a inferência> --model models/best.pt --no-display
```
- substitua o "<caminho do vídeo no seu dispositivo>", pelo lugar aonde está o video que será analisado;
- substitua o " <nome do vídeo após a inferência>", para indicar o nome e local aonde será salvo o vídeo;

##### Vendo o resultado:
Caso não possa ver o video pelo seu raspberry (Por exemplo, está no modo somente terminal).
Para visualizar o video, precisamos transferir do seu raspberry pi para sua máquina local.
Na sua maquina, abra o terminal, e digite:
```bash
scp <username do raspberry>@<ip do raspberry>:~/<local aonde foi salvo o video> <Aonde será enviado o video no seu computador>
```

#### Utilizar uma imagemm pré-feita para visualizar: 

Também é possível fazer o teste com imagens pré-feitas, tendo todas as dependências instaladas, basta usar o comando:
```bash
python3 stream/image_visualize model=runs/<Nome do modelo>/weights/best.pt source=<caminho da imagem>
```
- substitua <Nome do modelo>, pelo nome do modelo que queira utilizar para a analise;
- substitua <caminho da imagem>, pelo nome e caminho da imagem a ser utilizado;
---

##### Vendo o resultado:
Caso não possa ver a imagem pelo seu raspberry (Por exemplo, está no modo somente terminal).
Para visualizar a imagem, precisamos transferir do seu raspberry pi para sua máquina local.
Na sua maquina, abra o terminal, e digite:
```bash
scp <username do raspberry>@<ip do raspberry>:~/<local aonde foi salvo a imagem> <Aonde será enviado a imagem no seu computador>
```

## 7. Integrando os dados com o grafena

> [!WARNING]
> Estamos considerando que está a fazer essa configuração no raspberry pi

A integração com o grafena é bastante simples.
Para integrar com o grafena, iremos utilizar o cliente prometheus:
Para instalar, va para https://prometheus.io/download/
selecione em `Operating System` como `Linux` 
Selecione em `Architecture` como `arm64`
Faça o download da versão que contenha `LTS`
depois, no terminal, extraia o arquivo instalado:
```bash
tar -xvzf <arquivo a ser extraido>
```
TODO: Estudando conteinerização com container prometheus



## 8. Cofirmação do resultado

A confirmação do resultado na PoC pode ser visualizada no terminal durante o treinamento e nas imagens ou vídeos pós treinamento com a porcentagem da inferência e a respectiva classificação. 

Os resultados do projeto final serão informados posteriormente após a conclusão do projeto.

---
## 9. Diagrama de blocos

O diagrama de blocos desenvolvido ilustra as entradas, processamento e saídas do nosso sistema, considerando aspectos de hardware e software. A plataforma utilizada para desenvolvê-lo foi o Miro.

Como informações de entrada haverão apenas as capturas de imagem realizadas pela câmera do Raspberry pi.
Já no processamento são consideradas todas as operações realizadas após a captura até a formulação de dados de saída, sendo a inferência, a classificação e o cálculo das métricas, as principais operações dessa etapa.
Por fim, como saída temos as informações expressas no dashboard, a notificação de email em caso de alto índice de passagem de itens defeituosos, a documentação do FastAPI e o sinal de alerta para desvio automático de itens a ser processado por outro dispositivo embarcado. 

<img width="1029" height="1518" alt="Meu primeiro board" src="https://github.com/user-attachments/assets/a050302a-7d85-4d7e-8f2d-2a44df469123" />

---
