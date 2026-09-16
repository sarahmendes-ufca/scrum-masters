# scrum-masters

# 📦 TCC CENÁRIO 3: Visão Computacional para Detecção de Defeitos em Embalagens de Produtos

### 👤 Identificação da Equipe
- **scrum-masters**
- **repositório: https://github.com/sarahmendes-ufca/scrum-masters**
- **GitHub dos Membros:*- 
  - José Dhonatan Fernandes de Almeida — [`@sudo-invers`](https://github.com/sudo-invers)
  - Letícia Maria dos Santos Dias — [`@leticia-software-engineer`](https://github.com/leticia-software-engineer)
  - Sarah Mendes Teles — [`@sarahmendes-ufca`](https://github.com/sarahmendes-ufca)

---

## 📌 Sumário

- [1. Visão Geral da Solução](#1-visão-geral-da-solução)
- [2. Arquitetura do Sistema](#2-arquitetura-do-sistema)
- [3. Componentes utilizados](#3-componentes-utilizados)
- [4. Pré-requisitos e Recursos](#4-pré-requisitos-e-recursos)

  - [4.1 Hardware, Dispositivos e Infraestrutura](#41-hardware-dispositivos-e-infraestrutura)
  - [4.2 Software de Sistema e Drivers de Captura](#42-software-de-sistema-e-drivers-de-captura)
- [5. Dependências e procedimento de instalação](#5-dependências-e-procedimento-de-instalação)

  - [5.1 Linguagem e Frameworks Web / Servidores](#51-linguagem-e-frameworks-web--servidores)
  - [5.2 Visão Computacional, IA e Processamento Matemático](#52-visão-computacional-ia-e-processamento-matemático)
  - [5.3 MLOps, Engenharia de Dados e Observabilidade](#53-mlops-engenharia-de-dados-e-observabilidade)
  - [5.4 Conteinerização, Rede e CI/CD](#54-conteinerização-rede-e-cicd)
  - [5.5 Procedimento de Instalação e Configuração](#55-procedimento-de-instalação-e-configuração)
- [6. Comandos e Procedimento para Execução](#6-comandos-e-procedimento-para-execução)

  - [6.1 Visualização pelo terminal](#61-visualização-pelo-terminal)
  - [6.2 Visualização pelo navegador](#62-visualização-pelo-navegador)
  - [6.3 Utilização de vídeo pré-gravado](#63-utilização-de-vídeo-pré-gravado)
  - [6.4 Utilização de imagem pré-gravada](#64-utilização-de-imagem-pré-gravada)
- [7. Integrando os dados com o Grafana](#7-integrando-os-dados-com-o-grafana)

  - [7.1 Instalação](#71-instalação)
  - [7.2 Configuração](#72-configuração)
- [8. Confirmação do resultado](#8-confirmação-do-resultado)
- [9. Diagrama de blocos](#9-diagrama-de-blocos)

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

```bash
.
├── models
│   └── modelfile.eim
├── README.md
└── scripts
    ├── classificador_interface.py
    ├── classificador.py
    └── requirements.txt
```
---
## 3. Componentes utilizados

- Raspberry pi 5 ou microcomputador similar
- Fonte de alimentação USB-C 27 W para Raspberry pi 5
- Módulo câmera Raspberry pi V1.3, 5 MP, interface CSI
- Cabo adaptador CSI para câmera do Raspberry pi
- Cartão de Memória MicroSDXC 128 GB, classificação A2/V30
---

## 4. Pré-requisitos e Recursos

### Hardware, Dispositivos e Infraestrutura

- **Placa Principal:*- Raspberry Pi 5 (8 GB de RAM)  (ARM64) com conexão à internet.. (Plataformas alternativas compatíveis/projetadas: Gigabyte GA-SBCAP3350, Nvidia Jetson Nano, Banana Pi e ESP-32 com módulo câmera).

- **Câmera:*- Módulo Câmera Raspberry Pi V1.3 (5 MP) com interface e cabo adaptador CSI.

- **Alimentação:*- Fonte oficial USB-C 27 W para Raspberry Pi 5.

- **Armazenamento:*- Cartão microSD (com sistema de arquivos Overlay FS para restrição de escrita).


### Software de Sistema e Drivers de Captura

- **Sistema Operacional:*- Raspberry Pi OS 64-bit (Debian Trixie).

- **Drivers & Utilitários de Vídeo:*- V4L2 (Video4Linux2), libcamera e pacote rpicam-apps (rpicam-vid, rpicam-still).

---
## 5. Dependências

### Linguagem e Frameworks Web / Servidores

- **Linguagem Base:** Python 3.11.

- **Servidor Web:** Flask, utilizado para disponibilizar a interface web e o streaming MJPEG da câmera.

- **Servidor de Métricas:** `prometheus-client`, utilizado para expor as métricas do classificador em um endpoint HTTP para coleta pelo Grafana Alloy.

### Visão Computacional, IA e Processamento de Imagem

- **Inferência de IA:** Edge Impulse Linux SDK (`edge_impulse_linux`), utilizado para carregar e executar o modelo `.eim` diretamente na Raspberry Pi 5.

- **Controle da Câmera:** Picamera2, utilizado para configurar e capturar imagens da câmera da Raspberry Pi.

- **Processamento de Imagem:** OpenCV (`python3-opencv`), utilizado para conversão de formatos de imagem, desenho das caixas delimitadoras e codificação dos frames em JPEG para o streaming.

- **Processamento Numérico:** NumPy, utilizado no processamento dos frames capturados pela câmera.

### MLOps, Engenharia de Dados e Observabilidade

- **Desenvolvimento e Treinamento do Modelo:** Edge Impulse, utilizado para preparação do dataset, treinamento, validação e exportação do modelo de classificação/detecção.

- **Gestão e Anotação do Dataset:** Roboflow, utilizado na preparação e anotação das imagens utilizadas no desenvolvimento do modelo.

- **Versionamento de Código:** Git e GitHub.

- **Coleta de Métricas:** Prometheus Client, responsável por disponibilizar as métricas produzidas pelo classificador.

- **Agente de Coleta:** Grafana Alloy, responsável por coletar as métricas expostas pelo classificador e encaminhá-las ao Grafana Cloud.

- **Visualização e Monitoramento:** Grafana, utilizado para criação dos dashboards e acompanhamento das métricas do sistema.

### Conteinerização, Rede e CI/CD

- **Contêineres:** Docker e Docker Compose, utilizados para execução e organização dos serviços que fazem parte da aplicação.

- **Compilação Multiplataforma:** Docker Buildx e QEMU, utilizados quando necessário para construção de imagens compatíveis com a arquitetura ARM64 da Raspberry Pi 5.

- **Registro de Imagens:** GitHub Container Registry (GHCR).

- **Rede / VPN Mesh:** Tailscale, utilizado para acesso remoto à Raspberry Pi e comunicação entre os dispositivos da infraestrutura.

- **CI/CD:** GitHub Actions, utilizado para automação dos processos de versionamento e integração/entrega do projeto.

### Testes e Qualidade

- **Testes:** Pytest, quando utilizado para testes automatizados dos componentes Python do projeto.

- **Linting:** Ruff, quando utilizado para análise estática e padronização do código Python.

- **Logs:** Logs utilizando a saída padrão da aplicação para acompanhamento da execução, inferências e alertas do classificador.

## Procedimento de Instalação e Configuração

### Instalando as dependencias do sistema
No raspberry pi, em um terminal:
```bash
sudo apt update
sudo apt install -y python3-opencv python3-picamera2 portaudio19-dev
```

#### motivo de cada dependência
- python3-opencv: 
  - Capturar/processar frames
  - Desenhar informações sobre os frames
  - Fazer pré-processamento antes da inferência

- python3-picamera2:
  - Biblioteca usada para controlar a câmera do Raspberry Pi

- portaudio19-dev:
  - Não utilizado, mas é dependência de python3-picamera2

### Instalando as dependencias do projeto
No raspberry pi, crie um ambiente virtual:
> [!WARNING]
> Daqui em diante, sempre execute os comando estando dentro do ambiente virtual
> Instalar depedencias com pip fora de um, pode acarretar a problemas sérios no sistema operacional

```bash
python3 -m venv .venv
```
e depois ative ele:
```bash
source .venv/bin/activate
```
> [!WARNING]
> Apenas instale os comando estando dentro do ambiente virtual
> Instalar depedencias com pip fora de um, pode acarretar a problemas sérios no sistema operacional

após isso, instalamos as dependências do projeto:

```bash
pip install -r requirements.txt
```

> Se não possuir pip instalado no seu raspberry, instale com:
> ```bash
>  sudo apt install python3-pip
> ```

## 6. Procedimento para Execução

### Visualização pelo terminal:

com o repositório, com todas as dependências instaladas e dentro do ambiente virtual (venv), execute:

```bash
python3 classification.py
```
o que irá iniciar o programa, e o log do que a camerâ está vendo, poderá ser visto no terminal através de um log.

### Visualizando pelo navegador:

Para ver no navegador, no raspberry pi, dentro do ambiente virtual:

```bash
python3 classification_interface.py
```
depois, no seu navegador web de preferencia, na barra de url, digite:

```
http://<ip do raspberry pi>:8000
```
- Substitua <ip do seu raspberry pi>, pelo ip real do seu raspberry pi

### Utilizar um vídeo pré-gravado para visualizar
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

#### Vendo o resultado:
Caso não possa ver o video pelo seu raspberry (Por exemplo, está no modo somente terminal).
Para visualizar o video, precisamos transferir do seu raspberry pi para sua máquina local.
Na sua maquina, abra o terminal, e digite:
```bash
scp <username do raspberry>@<ip do raspberry>:~/<local aonde foi salvo o video> <Aonde será enviado o video no seu computador>
```

### Utilizar uma imagem pré-feita para visualizar: 

Também é possível fazer o teste com imagens pré-feitas, basta usar o comando:
```bash
python3 stream/image_visualize model=runs/<Nome do modelo>/weights/best.pt source=<caminho da imagem>
```
- substitua <Nome do modelo>, pelo nome do modelo que queira utilizar para a analise;
- substitua <caminho da imagem>, pelo nome e caminho da imagem a ser utilizado;
---

#### Vendo o resultado:
Caso não possa ver a imagem pelo seu raspberry (Por exemplo, está no modo somente terminal).
Para visualizar a imagem, precisamos transferir do seu raspberry pi para sua máquina local.
Na sua maquina, abra o terminal, e digite:
```bash
scp <username do raspberry>@<ip do raspberry>:~/<local aonde foi salvo a imagem> <Aonde será enviado a imagem no seu computador>
```

## 7. Integrando os dados com o grafena

> [!WARNING]
> Estamos considerando que está a fazer essa configuração no raspberry pi

### Instalação
A integração com o grafena é bastante simples.
Para integrar com o grafena, iremos utilizar o cliente prometheus, com o grafana alloy. para isso, vamos instalar as dependências:
```bash
sudo apt update
sudo apt install -y gpg wget
```

agora vamos adicionar o repositorio do grafana ao apt do rasp:
```bash
echo "deb [signed-by=/usr/share/keyrings/grafana.gpg] https://apt.grafana.com stable main" | sudo tee /etc/apt/sources.list.d/grafana.list
```

Finalmente, vamos instalar o grafana alloy:
```bash
sudo apt update
sudo apt install -y alloy
```

Para ter certeza que está funcionando, veja se esta habilitado (deve mostrar `running`)

```bash
sudo systemctl enable alloy.service # garante que vai ficar habilitado mesmo que o sistema reinicie
sudo systemctl start alloy.service # Inicia o serviço caso não tenha iniciado ainda
sudo systemctl status alloy.service # Verifica o status
```

### Configuração
Para começar, iremos criar variaveis de ambiente para o alloy, isso deixa a configuração mais organizada e legível.
Acesse como super usuario:

```bash
sudo nano /etc/alloy/config.alloy
```

e adicione no final do arquivo (se já ouver alguma das linhas, ignore e não copie a linha que já tiver no arquivo, o resto copie), o seguinte:
```text
CONFIG_FILE="/etc/alloy/config.alloy"
CUSTOM_ARGS="--disable-reporting"

GRAFANA_CLOUD_URL="<sua_url_do_prometheus>"
GRAFANA_CLOUD_USERNAME="<seu_id_de_usuario_do_prometheus>"
GRAFANA_CLOUD_TOKEN="<seu_token_do_prometheus>"
```

> Substitua oque esta entre "<" ">", pelos dados pedidos. Se não souber como fazer, pode ver nosso tutorial
> em TODO:wiki_page ou na página oficial do grafana labs prometheus: [Documentação oficial](https://grafana.com/docs/grafana/latest/datasources/prometheus/configure/)
> aviso dado, porquê configurar errado os dados do prometheus é algo comum.

Restrinja a leitura e edição para apenas super-usuários(`sudo`):

```bash
sudo chmod 600 /etc/default/alloy
```

Agora, vamos referenciar oque colocamos no arquivo anterior, para efetivamente conectar ao grafana.
como super  usuário, abra:

```bash
sudo nano /etc/alloy/config.alloy
```

e substitua oque tiver no arquivo por:

```text
prometheus.scrape "node_exporter" {
  targets = [{ "__address__" = "localhost:9100" }]
  forward_to = [prometheus.remote_write.grafana_cloud.receiver]
}

prometheus.scrape "yolo_metrics" {
  targets = [{ "__address__" = "localhost:8000" }]
  forward_to = [prometheus.remote_write.grafana_cloud.receiver]
}

prometheus.remote_write "grafana_cloud" {
  endpoint {
    url = sys.env("GRAFANA_CLOUD_URL")

    basic_auth {
      username = sys.env("GRAFANA_CLOUD_USERNAME")
      password = sys.env("GRAFANA_CLOUD_TOKEN")
    }
  }
}
```

Feito isso, reinicie o serviço do alloy:
```bash
sudo systemctl restart alloy.service
```
Confirme que esta carregado (`running`):
```bash
sudo systemctl status alloy.service
```
e verifique o log do alloy, para ver se tudo está indo certo:
```bash
sudo journalctl -u alloy -n 30 --no-pager
```

## Visualizando os dados no grafana

Com os passos anteriores feitos, podemos começar a fazer o seu dashboard no grafana.
Escolha se prefere rodar `classificador.py` ou `classificador_interface`
O projeto inclui por padrão, as seguintes métricas para o grafana:

> Mais métricas podem ser adicionadas manualmente modificando os arquvivos `classificador.py` ou `classificador_interface.py` (oque estiver usando)

- `classificador_produtos_avaliados_total`
- `classificador_logs_por_classificacao_total`

Serão as métricas acima que iramos visualiar, para poder fazer isso acesse:
{TODO: grafana_explore1}
{TODO: grafana_explore2}

e coloque uma métrica a ser avaliada:
{TODO: grafana_explore3}

e para ver o grafico, selecion 'Run Query':

{TODO: grafana_explore4}

Para criar seus próprios paineis (dashboards), pode-se ler a documentação oficial do grafana para isso:
[Documentação oficial do grafana](https://grafana.com/docs/grafana/latest/visualizations/dashboards/build-dashboards/create-dashboard/)

## 8. Cofirmação do resultado


---
## 9. Diagrama de blocos

O diagrama de blocos desenvolvido ilustra as entradas, processamento e saídas do nosso sistema, considerando aspectos de hardware e software. A plataforma utilizada para desenvolvê-lo foi o Miro.

Como informações de entrada haverão apenas as capturas de imagem realizadas pela câmera do Raspberry pi.
Já no processamento são consideradas todas as operações realizadas após a captura até a formulação de dados de saída, sendo a inferência, a classificação e o cálculo das métricas, as principais operações dessa etapa.
Por fim, como saída temos as informações expressas no dashboard, a notificação de email em caso de alto índice de passagem de itens defeituosos, a documentação do FastAPI e o sinal de alerta para desvio automático de itens a ser processado por outro dispositivo embarcado. 

<img width="1029" height="1518" alt="diagrama de blocos represenando as entradas, processamento e saídas do nosso sistema" src="https://github.com/user-attachments/assets/a050302a-7d85-4d7e-8f2d-2a44df469123" />

---

