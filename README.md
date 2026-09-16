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

- [1. Visão Geral da Solução](#1-visão-geral-da-solução)
- [2. Arquitetura do Sistema](#2-arquitetura-do-sistema)
  - [2.1. O que os scripts fazem](#21-o-que-os-scripts-fazem)
- [3. Pré-requisitos e Recursos](#3-pré-requisitos-e-recursos)
  - [3.1. Componentes Utilizados](#31-componentes-utilizados)
  - [3.2. Hardware, Dispositivos e Infraestrutura](#32-hardware-dispositivos-e-infraestrutura)
  - [3.3. Software de Sistema e Drivers de Captura](#33-software-de-sistema-e-drivers-de-captura)
- [4. Dependências](#4-dependências)
  - [4.1. Linguagem e Frameworks Web / Servidores](#41-linguagem-e-frameworks-web--servidores)
  - [4.2. Visão Computacional, IA e Processamento de Imagem](#42-visão-computacional-ia-e-processamento-de-imagem)
  - [4.3. MLOps, Engenharia de Dados e Observabilidade](#43-mlops-engenharia-de-dados-e-observabilidade)
  - [4.4. Conteinerização, Rede e CI/CD](#44-conteinerização-rede-e-cicd)
  - [4.5. Testes e Qualidade](#45-testes-e-qualidade)
- [5. Procedimento de Instalação e Configuração](#5-procedimento-de-instalação-e-configuração)
  - [5.1. Instalando as dependências do sistema](#51-instalando-as-dependências-do-sistema)
    - [5.1.1. Motivo de cada dependência](#511-motivo-de-cada-dependência)
  - [5.2. Instalando as dependências do projeto](#52-instalando-as-dependências-do-projeto)
- [6. Procedimento para Execução](#6-procedimento-para-execução)
  - [6.1. Visualização pelo terminal](#61-visualização-pelo-terminal)
  - [6.2. Visualizando pelo navegador](#62-visualizando-pelo-navegador)
- [7. Integrando os dados com o Grafana](#7-integrando-os-dados-com-o-grafana)
  - [7.1. Por que usar o Grafana](#71-por-que-usar-o-grafana)
  - [7.2. Instalação](#72-instalação)
  - [7.3. Configuração](#73-configuração)
- [8. Visualizando os dados no Grafana](#8-visualizando-os-dados-no-grafana)
- [9. Confirmação do Resultado](#9-confirmação-do-resultado)
- [10. Diagrama de Blocos](#10-diagrama-de-blocos)

---

> [!WARNING]
> Sempre que em algum comando estiver escrito os simbolo "<" ou ">", não inclua eles no comando.
> Exemplo: http://<ip do raspberry pi>:8000
> Deve ser escrito quando usado assim (usado um ip aleatorio como exemplo): http://100.95.153.33:8000
> Substituido <ip do raspeberry pi> por 100.95.153.33
---

## 1 -- Visão Geral da Solução

No ambiente fabril, a etapa de rotulagem na fase final da confecção de produtos frequentemente apresenta gargalos operacionais. Embora seja um processo automatizado, os equipamentos estão sujeitos a falhas. A eventual transferência de produtos defeituosos da esteira de produção para a distribuidora e, consequentemente, para o consumidor final, acarreta insatisfação, prejuízos à credibilidade da marca e perda da confiança na linha de produtos. Adicionalmente, o rótulo constitui a principal fonte de dados para o consumidor, contendo informações essenciais como instruções de uso e prazo de validade; a ausência ou baixa legibilidade desses itens representa um problema crítico de qualidade e conformidade.

A inserção de inspeção humana 100% manual após a rotulagem, para checar a presença e o posicionamento das etiquetas, seria uma alternativa direta. No entanto, essa abordagem mostra-se financeiramente inviável e pouco operacional, dada a alta exigência de tempo, mão de obra e custos associados.

Diante desse cenário, este trabalho propõe uma solução prática, segura e de menor custo por meio do uso de visão computacional. Trata-se do desenvolvimento de um sistema baseado em inteligência artificial para a verificação e classificação de rótulos em tempo real. A solução permite a emissão de alertas inteligentes e a integração com mecanismos automatizados de triagem — como braços robóticos —, garantindo a imediata remoção dos itens com defeitos identificados.

---

## 2 -- Arquitetura do Sistema

```bash

├── models                   
│   └── modelfile.eim                # É o modelo de machine learning compilado
├── README.md                        # Documentação do projeto com instruções de execução
└── scripts
    ├── classificador_interface.py   # Script estruturado para gerenciar a interface de usuário para facilitar a interação e a visualização dos resultados obtidos pelo classificador.   
    ├── classificador.py             #script de execução principal que gerencia a câmera nativa da Raspberry Pi 5 via Picamera2 e realiza o loop de inferência de visão computacional em tempo real
    └── requirements.txt             # Arquivo de dependências necessárias para executar o projeto
```

---
### 2.1 -- Oque os scripts fazem

`classificador.py`:
- realiza a inspeção e o monitoramento do processo em tempo real
`classificador_interface`: 
- enquanto classificador_interface.py apresenta visualmente pelo navegador web, em tempo real, os resultados das inspeções realizadas pela IA.

---

## 3 -- Pré-requisitos e Recursos

### 3.1 -- Componentes utilizados

- Raspberry pi 5 ou microcomputador similar
- Fonte de alimentação USB-C 27 W para Raspberry pi 5
- Módulo câmera Raspberry pi V1.3, 5 MP, interface CSI
- Cabo adaptador CSI para câmera do Raspberry pi
- Cartão de Memória MicroSDXC 128 GB, classificação A2/V30
---

### 3.2 -- Hardware, Dispositivos e Infraestrutura

- **Placa Principal:** Raspberry Pi 5 (8 GB de RAM)  (ARM64) com conexão à internet.. (Plataformas alternativas compatíveis/projetadas: Gigabyte GA-SBCAP3350, Nvidia Jetson Nano, Banana Pi e ESP-32 com módulo câmera).

- **Câmera:** Módulo Câmera Raspberry Pi V1.3 (5 MP) com interface e cabo adaptador CSI.

- **Alimentação:** Fonte oficial USB-C 27 W para Raspberry Pi 5.

- **Armazenamento:** Cartão microSD (com sistema de arquivos Overlay FS para restrição de escrita).


### 3.3 -- Software de Sistema e Drivers de Captura

- **Sistema Operacional:** Raspberry Pi OS 64-bit (Debian Trixie).

- **Drivers & Utilitários de Vídeo:** V4L2 (Video4Linux2), libcamera e pacote rpicam-apps (rpicam-vid, rpicam-still).

---
## 4 -- Dependências

### 4.1 -- Linguagem e Frameworks Web / Servidores

- **Linguagem Base:** Python 3.11.

- **Servidor Web:** Flask, utilizado para disponibilizar a interface web e o streaming MJPEG da câmera.

- **Servidor de Métricas:** `prometheus-client`, utilizado para expor as métricas do classificador em um endpoint HTTP para coleta pelo Grafana Alloy.

### 4.2 -- Visão Computacional, IA e Processamento de Imagem

- **Inferência de IA:** Edge Impulse Linux SDK (`edge_impulse_linux`), utilizado para carregar e executar o modelo `.eim` diretamente na Raspberry Pi 5.

- **Controle da Câmera:** Picamera2, utilizado para configurar e capturar imagens da câmera da Raspberry Pi.

- **Processamento de Imagem:** OpenCV (`python3-opencv`), utilizado para conversão de formatos de imagem, desenho das caixas delimitadoras e codificação dos frames em JPEG para o streaming.

- **Processamento Numérico:** NumPy, utilizado no processamento dos frames capturados pela câmera.

### 4.3 -- MLOps, Engenharia de Dados e Observabilidade

- **Desenvolvimento e Treinamento do Modelo:** Edge Impulse, utilizado para preparação do dataset, treinamento, validação e exportação do modelo de classificação/detecção.

- **Versionamento de Código:** Git e GitHub.

- **Coleta de Métricas:** Prometheus Client, responsável por disponibilizar as métricas produzidas pelo classificador.

- **Agente de Coleta:** Grafana Alloy, responsável por coletar as métricas expostas pelo classificador e encaminhá-las ao Grafana Cloud.

- **Visualização e Monitoramento:** Grafana, utilizado para criação dos dashboards e acompanhamento das métricas do sistema.

### 4.4 -- Conteinerização, Rede e CI/CD

- **Rede / VPN Mesh:** Tailscale, utilizado para acesso remoto à Raspberry Pi e comunicação entre os dispositivos da infraestrutura.

### 4.5 -- Testes e Qualidade

- **Linting:** Ruff, quando utilizado para análise estática e padronização do código Python.

- **Logs:** Logs utilizando a saída padrão da aplicação para acompanhamento da execução, inferências e alertas do classificador.

## 5 -- Procedimento de Instalação e Configuração

### 5.1 -- Instalando as dependencias do sistema
No raspberry pi, em um terminal:
```bash
sudo apt update
sudo apt install -y python3-opencv python3-picamera2 portaudio19-dev
```

#### 5.1.1 -- motivo de cada dependência
- python3-opencv: 
  - Capturar/processar frames
  - Desenhar informações sobre os frames
  - Fazer pré-processamento antes da inferência

- python3-picamera2:
  - Biblioteca usada para controlar a câmera do Raspberry Pi

- portaudio19-dev:
  - Não utilizado, mas é dependência de python3-picamera2

### 5.2 -- Instalando as dependencias do projeto
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
> Apenas instale os comandos estando dentro do ambiente virtual
> Instalar depedencias com pip fora de um ambiente virtual, pode acarretar a problemas sérios no sistema operacional.

após isso, instalamos as dependências do projeto:

```bash
pip install -r scripts/requirements.txt
```

> Se não possuir pip instalado no seu raspberry, instale com:
> ```bash
>  sudo apt install python3-pip
> ```

## 6 -- Procedimento para Execução

### 6.1 -- Visualização pelo terminal:

com o repositório, com todas as dependências instaladas e dentro do ambiente virtual (venv), execute:

```bash
python3 scripts/classificador.py
```
o que irá iniciar o programa, e o log do que a camerâ está classificando, poderá ser visto no terminal através dos logs.

### 6.2 -- Visualizando pelo navegador:

Para ver no navegador, no raspberry pi, dentro do ambiente virtual. Execute:

```bash
python3 scripts/classificador_interface.py
```
depois, no seu navegador web de preferencia, na barra de url, digite:

```
http://<ip do raspberry pi>:1337
```
> Substitua <ip do seu raspberry pi>, pelo ip real do seu raspberry pi


## 7 Integrando os dados com o grafana

### 7.1 -- Porque usar o grafana

o grafana é uma ferramente poderosa, que permite visualizar gráficos em tempo real, fazer alertas customizáveis, e definir quem pode receber os alertas e ver os gráfocos,
sua introdução ao sistema é extremamente útil no monitoramento dos produtos.

--- 
### 7.2 -- Instalação

> [!WARNING]
> Estamos considerando que está a fazer essa configuração no raspberry pi

A integração com o grafana é bastante simples.
Para integrar com o grafana, iremos utilizar o cliente prometheus, com o grafana alloy. para isso, vamos instalar as dependências:
```bash
sudo apt update
sudo apt install -y gpg wget
```

agora vamos adicionar o repositorio do grafana ao apt do rasp:
```bash
echo "deb [signed-by=/usr/share/keyrings/grafana.gpg] https://apt.grafana.com stable main" | sudo tee /etc/apt/sources.list.d/grafana.list
```

com isso feito, instalamos o grafana alloy:
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

### 7.3 -- Configuração
Para começar, iremos criar variaveis de ambiente para o alloy, isso deixa a configuração mais organizada e legível.
Acesse como super usuario:

```bash
sudo nano /etc/alloy/config.alloy
```

e adicione no final do arquivo (se já houver alguma das linhas, ignore e não copie a linha que já tiver no arquivo, o resto copie), o seguinte:
```text
CONFIG_FILE="/etc/alloy/config.alloy"
CUSTOM_ARGS="--disable-reporting"

GRAFANA_CLOUD_URL="<sua_url_do_prometheus>"
GRAFANA_CLOUD_USERNAME="<seu_id_de_usuario_do_prometheus>"
GRAFANA_CLOUD_TOKEN="<seu_token_do_prometheus>"
```

> Substitua oque esta entre "<" ">", pelos dados pedidos. Se não souber como fazer, pode visualizar na documentação do prometheus.
> [Documentação oficial do grafana labs prometheus](https://grafana.com/docs/grafana/latest/datasources/prometheus/configure/)
> aviso dado, porquê configurar errado os dados do prometheus é algo comum.

Restrinja a leitura e escrita para apenas super-usuários(`sudo`):

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

## 8 -- Visualizando os dados no grafana

Com os passos anteriores feitos, podemos começar a fazer o seu dashboard no grafana.
Escolha se prefere rodar `classificador.py` ou `classificador_interface`
O projeto inclui por padrão, as seguintes métricas para o grafana:

> Mais métricas podem ser adicionadas manualmente modificando os arquvivos `classificador.py` ou `classificador_interface.py` (oque estiver usando)

- `classificador_produtos_avaliados_total`
- `classificador_logs_por_classificacao_total`

Serão as métricas acima que iremos visualizar, para poder fazer isso acesse:

<img width="323" height="939" alt="grafana_explore1" src="https://github.com/user-attachments/assets/8f8573f1-035b-45ae-8865-f110a93d8091" />

---
<img width="1568" height="333" alt="grafana_explore2" src="https://github.com/user-attachments/assets/c73c6dd8-f8df-438c-bc4d-c8576f8f7750" />

---
e coloque uma métrica a ser avaliada:

<img width="1398" height="169" alt="grafana_explore3" src="https://github.com/user-attachments/assets/12d8ac02-27ce-4d18-9a98-8e1cdc731342" />

---
e para ver o gráfico, selecione `Run Query`:

<img width="801" height="147" alt="grafana_explore4" src="https://github.com/user-attachments/assets/90e13264-660b-4bfc-8060-0a2eff1b88b8" />

---
Para criar seus próprios painéis (dashboards), pode-se ler a documentação oficial do grafana para isso:
[Documentação oficial do grafana](https://grafana.com/docs/grafana/latest/visualizations/dashboards/build-dashboards/create-dashboard/)

## 9 -- Cofirmação do resultado

Resultado no terminal:
<img width="1600" height="720" alt="WhatsApp Image 2026-09-16 at 18 16 08 (1)" src="https://github.com/user-attachments/assets/c8608472-0db4-4d9f-b2fc-38c9f164fa8f" />
Resposta com alerta de desvio no terminal em caso de rótulo ausente

<img width="1600" height="720" alt="WhatsApp Image 2026-09-16 at 18 16 08 (2)" src="https://github.com/user-attachments/assets/1ca2ee38-3f95-41ac-a8f1-d86a69fe8f2f" />
Saída do terminal em caso de rótulo íntegro

---
Resultado na web:
<img width="1600" height="720" alt="WhatsApp Image 2026-09-16 at 18 16 08" src="https://github.com/user-attachments/assets/ae9ed4d8-6dd8-4b3b-a1be-0f6aa1d69f5c" />
Interface com a captura e classificação em produto com rótulo danificado

<img width="1200" height="1600" alt="WhatsApp Image 2026-09-16 at 14 59 31" src="https://github.com/user-attachments/assets/b47203a2-d671-416d-9d77-3b5943b60832" />
Interface com a captura e classificação em produto com rótulo ausente

---
---
Graficos de demonstração em um painel(dashboard) no grafana:

<img width="1519" height="662" alt="resu1" src="https://github.com/user-attachments/assets/221882cd-4240-4dd5-9854-f7891d0fb56d" />

---
<img width="1197" height="564" alt="resu2" src="https://github.com/user-attachments/assets/c375aef2-2f1b-4ccf-9ae2-8b6c8d350f40" />

---
<img width="1534" height="670" alt="resu3" src="https://github.com/user-attachments/assets/3404d226-2b3d-4791-88ac-11680b1d23bf" />

---
<img width="1535" height="657" alt="resu4" src="https://github.com/user-attachments/assets/ac71c9ba-dda2-485b-b387-ef65efc03451" />

---
## 10 -- Diagrama de blocos

O diagrama de blocos desenvolvido ilustra as entradas, processamento e saídas do nosso sistema, considerando aspectos de hardware e software. A plataforma utilizada para desenvolvê-lo foi o Miro.

Como informações de entrada haverão apenas as capturas de imagem realizadas pela câmera do Raspberry pi.
Já no processamento são consideradas todas as operações realizadas após a captura até a formulação de dados de saída, sendo a inferência, a classificação e o cálculo das métricas, as principais operações dessa etapa.
Por fim, como saída temos as informações expressas no dashboard, a notificação de email em caso de alto índice de passagem de itens defeituosos, a documentação do FastAPI e o sinal de alerta para desvio automático de itens a ser processado por outro dispositivo embarcado. 

<img width="1056" height="1600" alt="diagrama de blocos represenando as entradas, processamento e saídas do nosso sistema" src="https://github.com/user-attachments/assets/eded66d5-f6e2-4575-ae73-ea1bd90b42d4" />

---

