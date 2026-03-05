# 📘 DEVELOPER_GUIDE.md

```markdown
# DROL - Guia para Desenvolvedores

Este documento explica como configurar e executar o projeto DROL localmente.

---

# 📌 Visão Geral da Arquitetura

O DROL utiliza uma arquitetura distribuída:

- `camera_server.py` roda no host (Windows/Linux)
- Container Docker executa:
  - Flask
  - MediaPipe Tasks
  - Processamento de landmarks
- Persistência via volume Docker
- Dataset salvo em `/app/DROL/data/sinais.json`

Fluxo:

Webcam → camera_server → Stream HTTP → Docker → MediaPipe → Flask → Navegador

---

# 🧰 Pré-requisitos

Antes de iniciar, certifique-se de ter instalado:

- Python 3.10+
- Docker Desktop
- Git

---

# 📂 Estrutura do Projeto

/app  
└── DROL  
&nbsp;&nbsp;&nbsp;&nbsp;├── registrar_sinal.py  
&nbsp;&nbsp;&nbsp;&nbsp;├── reconhecer_sinal.py  
&nbsp;&nbsp;&nbsp;&nbsp;├── data/  
&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;└── sinais.json  
&nbsp;&nbsp;&nbsp;&nbsp;├── models/  
&nbsp;&nbsp;&nbsp;&nbsp;└── tools/  

---

# 🚀 Como Rodar o Projeto

## 1️⃣ Clonar o repositório

```

git clone <URL_DO_REPOSITORIO>
cd DROL

```

---

## 2️⃣ Iniciar o servidor de câmera (HOST)

Em um terminal fora do Docker:

```

cd tools
python camera_server.py

```

Verifique se o stream está ativo acessando:

```

http://SEU_IP:5000/video

```

---

## 3️⃣ Construir a imagem Docker (primeira vez)

```

docker build -t drol .

```

---

## 4️⃣ Rodar o container

IMPORTANTE: montar o volume corretamente.

Windows (exemplo):

```

docker run -p 5000:5000 -v "C:\CAMINHO\PARA\DROL\data:/app/DROL/data" drol

```

Linux:

```

docker run -p 5000:5000 -v $(pwd)/data:/app/DROL/data drol

```

---

# 🌐 Acessando a Aplicação

Nunca use o IP interno do container (172.x.x.x).

Use:

```

[http://localhost:5000](http://localhost:5000)

```

---

# ✍ Registrar um Sinal

Acesse:

```

[http://localhost:5000/registrar?nome=A](http://localhost:5000/registrar?nome=A)

```

Mantenha a mão estável por 5 segundos.

Confirmação esperada no terminal:

```

Sinal capturado!
Sinal 'A' salvo com sucesso.

```

---

# 🔎 Ativar Reconhecimento

```

[http://localhost:5000/reconhecer](http://localhost:5000/reconhecer)

```

Para parar:

```

[http://localhost:5000/parar](http://localhost:5000/parar)

```

---

# 📁 Persistência de Dados

O arquivo `sinais.json` é salvo em:

```

/app/DROL/data/sinais.json

```

Se o volume estiver montado corretamente, ele também aparecerá na pasta `data/` do host.

---

# ⚠ Problemas Comuns

## Stream não abre
Verifique se o `camera_server.py` está rodando.

## Não salva sinais
- Verifique se há detecção de mão
- Verifique permissões do volume
- Verifique logs do container

## Não consegue acessar pelo navegador
Confirme:
```

docker ps

```
E verifique se a porta 5000 está publicada:
```

0.0.0.0:5000->5000/tcp

```

---

# 🧠 Boas Práticas

- Nunca acessar webcam diretamente dentro do container
- Sempre usar volume para persistência
- Evitar IP fixo no código
- Preferir variáveis de ambiente para configurações

---

# 🔮 Próximas Evoluções Planejadas

- Normalização de landmarks
- Redução de falso positivo
- Classificador treinado (SVM / MLP)
- Interface web unificada

---

# 📌 Contato

Em caso de dúvida, consulte:
- DOCUMENTATION.md
- ARCHITECTURE.md
- Ou abra uma issue no repositório
```

