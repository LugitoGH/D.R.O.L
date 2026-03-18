import cv2
import mediapipe as mp
import os
import time
import json
import math
import logging
import socket
import numpy as np
from flask import Flask, Response, jsonify, render_template_string, request
from gtts import gTTS

# ================== LOG CONFIG ==================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("drol")

app = Flask(__name__)

# ================== PATHS SEGUROS ==================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "hand_landmarker.task")
FACE_MODEL_PATH = os.path.join(BASE_DIR, "models", "face_landmarker.task")
DATA_DIR = os.path.join(BASE_DIR, "data")
SINAIS_PATH = os.path.join(DATA_DIR, "sinais.json")
THRESHOLD_RECONHECIMENTO = float(os.getenv("DROL_RECOGNITION_THRESHOLD", "0.15"))
REGISTRO_SEGUNDOS = int(os.getenv("DROL_REGISTRATION_SECONDS", "5"))

os.makedirs(DATA_DIR, exist_ok=True)

# ================== MEDIAPIPE ==================
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

optionsFace = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=FACE_MODEL_PATH),
    running_mode=RunningMode.VIDEO
)

landmarkerHead = FaceLandmarker.create_from_options(optionsFace)

try:
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_hands=1,
    )
    landmarker = HandLandmarker.create_from_options(options)
    logger.info("MediaPipe carregado com sucesso (RunningMode.VIDEO).")
except Exception as e:
    logger.error("Erro ao carregar modelo MediaPipe em %s: %s", MODEL_PATH, e)
    raise

edges = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20)
]

face_edges = [
 (33,133),  # olho esquerdo
 (362,263), # olho direito
 (61,291),  # boca
 (199,152), # queixo
]

def detectar_gateway_linux():
    try:
        with open("/proc/net/route", "r", encoding="utf-8") as route_file:
            for line in route_file.readlines()[1:]:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                destination = parts[1]
                gateway_hex = parts[2]
                if destination == "00000000":
                    gateway_raw = bytes.fromhex(gateway_hex)
                    return socket.inet_ntoa(gateway_raw[::-1])
    except Exception:
        return None
    return None


def resolver_urls_camera():
    url_env = os.getenv("CAMERA_STREAM_URL")
    if url_env:
        return [url_env], "env:CAMERA_STREAM_URL"

    host_env = os.getenv("CAMERA_SERVER_HOST")
    port = os.getenv("CAMERA_SERVER_PORT", "5000")
    path = os.getenv("CAMERA_STREAM_PATH", "/video")

    if host_env:
        return [f"http://{host_env}:{port}{path}"], "env:CAMERA_SERVER_HOST"

    urls = [f"http://host.docker.internal:{port}{path}"]
    gateway = detectar_gateway_linux()
    if gateway:
        urls.append(f"http://{gateway}:{port}{path}")

    return urls, "auto"


def abrir_stream_camera():
    urls, origem = resolver_urls_camera()
    logger.info("Resolvendo stream da camera (origem=%s). Candidatos: %s", origem, ", ".join(urls))
    for url in urls:
        tentativa = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if tentativa.isOpened():
            logger.info("Stream conectado com sucesso: %s", url)
            return tentativa, url
        tentativa.release()
        logger.warning("Falha ao conectar no stream: %s", url)

    logger.error(
        "Nao foi possivel conectar ao stream HTTP. Configure CAMERA_STREAM_URL ou CAMERA_SERVER_HOST/CAMERA_SERVER_PORT."
    )
    return None, None


# ================== ESTADO ==================
frame_id = 0
modo = "normal"
nome_sinal_atual = ""
tempo_registro = 0.0
ultimo_reconhecido = ""
sinais = []
stream_url_ativo = ""
falhas_stream = 0
cap = None


def set_modo(novo_modo, motivo):
    global modo
    if modo != novo_modo:
        logger.info("Mudanca de modo: %s -> %s (%s)", modo, novo_modo, motivo)
    else:
        logger.info("Modo mantido em %s (%s)", modo, motivo)
    modo = novo_modo


# ================== FUNCOES ==================
def carregar_sinais():
    global sinais
    sinais = []
    try:
        if os.path.exists(SINAIS_PATH):
            with open(SINAIS_PATH, "r", encoding="utf-8") as f:
                for linha in f:
                    sinais.append(json.loads(linha))
        logger.info("%d sinais carregados de %s.", len(sinais), SINAIS_PATH)
    except Exception as e:
        logger.error("Erro ao carregar sinais de %s: %s", SINAIS_PATH, e)


def salvar_sinal(nome, vetor):
    try:
        with open(SINAIS_PATH, "a", encoding="utf-8") as f:
            json.dump({"nome": nome, "vetor": vetor}, f)
            f.write("\n")
        logger.info("Sinal '%s' salvo com sucesso.", nome)
    except Exception as e:
        logger.error("Erro ao salvar sinal '%s': %s", nome, e)

def normalizar_e_vetorizar(hand_landmarks):
    """
    Recebe lista de landmarks do MediaPipe.
    Retorna vetor normalizado com 63 valores.
    """

    # 1️⃣ Centralização (wrist como referência)
    wrist = hand_landmarks[0]

    pontos_centralizados = []
    for lm in hand_landmarks:
        pontos_centralizados.append({
            "x": lm.x - wrist.x,
            "y": lm.y - wrist.y,
            "z": lm.z - wrist.z
        })

    # 2️⃣ Normalização de escala
    # Usando distância entre wrist (0) e middle_finger_mcp (9)
    ref = pontos_centralizados[9]

    escala = math.sqrt(
        ref["x"]**2 +
        ref["y"]**2 +
        ref["z"]**2
    )

    if escala == 0:
        escala = 1e-6  # evita divisão por zero

    pontos_normalizados = []
    for p in pontos_centralizados:
        pontos_normalizados.append({
            "x": p["x"] / escala,
            "y": p["y"] / escala,
            "z": p["z"] / escala
        })

    # 3️⃣ Vetorização
    vetor = []
    for p in pontos_normalizados:
        vetor.extend([p["x"], p["y"], p["z"]])

    return vetor


def distancia(lm1, lm2):
    return math.sqrt((lm1["x"] - lm2.x) ** 2 + (lm1["y"] - lm2.y) ** 2 + (lm1["z"] - lm2.z) ** 2)


def status_dict():
    return {
        "modo": modo,
        "nome_sinal_atual": nome_sinal_atual,
        "ultimo_reconhecido": ultimo_reconhecido,
        "stream_url": stream_url_ativo,
        "stream_ok": cap is not None and cap.isOpened(),
        "total_sinais": len(sinais),
        "threshold": THRESHOLD_RECONHECIMENTO,
    }


def frame_aguardando_stream():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(
        frame,
        "Aguardando stream da camera...",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 255),
        2,
    )
    return frame


carregar_sinais()
cap, stream_url_ativo = abrir_stream_camera()

# ================== STREAM ==================
def generate_frames():
    global frame_id, tempo_registro, ultimo_reconhecido, cap, stream_url_ativo, falhas_stream

    while True:
        frame = None
        try:
            if cap is None or not cap.isOpened():
                falhas_stream += 1
                if falhas_stream == 1 or falhas_stream % 60 == 0:
                    logger.error(
                        "Stream indisponivel. Tentando reconectar... falhas=%d url=%s",
                        falhas_stream,
                        stream_url_ativo or "desconhecida",
                    )
                time.sleep(0.5)
                cap, stream_url_ativo = abrir_stream_camera()
                continue

            success, frame = cap.read()
            if not success:
                falhas_stream += 1
                if falhas_stream == 1 or falhas_stream % 60 == 0:
                    logger.error(
                        "Falha ao ler frame do stream HTTP. Verifique camera_server.py no host e variaveis de ambiente."
                    )
                cap.release()
                continue

            falhas_stream = 0
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            frame_id += 1
            timestamp_ms = int(time.time() * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)


            resultHead = landmarkerHead.detect_for_video(mp_image, timestamp_ms)

            if resultHead.face_landmarks:
                for face in resultHead.face_landmarks:

                    lm_points = [(int(lm.x*w), int(lm.y*h)) for lm in face]

                    for (i,j) in face_edges:
                        cv2.line(frame, lm_points[i], lm_points[j], (255,0,0), 1)
            
            

            if result.hand_landmarks:
                for hand in result.hand_landmarks:
                    lm_points = [(int(lm.x*w), int(lm.y*h)) for lm in hand]
                    h, w, _ = frame.shape
                
                    
                    
                    for (i,j) in edges:
                        cv2.line(frame, lm_points[i], lm_points[j], (0,255,0), 2)

                    for lm in hand:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

                        if modo == "registrar" and time.time() - tempo_registro >= REGISTRO_SEGUNDOS:
                            vetor = normalizar_e_vetorizar(hand)
                            salvar_sinal(nome_sinal_atual, vetor)
                            carregar_sinais()
                            set_modo("normal", "registro finalizado")

                        if modo == "reconhecer":

                            vetor_atual = normalizar_e_vetorizar(hand)

                            melhor_distancia = float("inf")
                            melhor_sinal = None

                            for sinal in sinais:
                                vetor_salvo = sinal.get("vetor", [])

                                if len(vetor_salvo) != 63:
                                    continue

                                soma = 0
                                for i in range(63):
                                    soma += (vetor_salvo[i] - vetor_atual[i]) ** 2

                                distancia_media = math.sqrt(soma / 63)

                                if distancia_media < melhor_distancia:
                                    melhor_distancia = distancia_media
                                    melhor_sinal = sinal["nome"]

                            #logger.info("Melhor distancia: %.4f", melhor_distancia)

                            if melhor_distancia < THRESHOLD_RECONHECIMENTO:
                                ultimo_reconhecido = melhor_sinal
                                texto = melhor_sinal

                                cv2.putText(
                                    frame,
                                    f"Sinal: {melhor_sinal}",
                                    (20, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1,
                                    (0, 255, 0),
                                    2,
                                )
            else:
                ultimo_reconhecido = ""

            cv2.putText(
                frame,
                f"Modo: {modo}",
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
        except Exception as e:
            logger.error("Erro no loop principal: %s", e)

        if frame is None:
            frame = frame_aguardando_stream()

        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()
        yield b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"


INDEX_HTML = """
<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DROL Controle</title>

<style>
:root {
--bg:#f4f7fb;
--card:#ffffff;
--text:#1f2937;
--accent:#0f766e;
--danger:#b91c1c;
--border:#d1d5db;
}

body{
margin:0;
font-family:Segoe UI,Tahoma,Verdana;
background:linear-gradient(135deg,#dbeafe,var(--bg));
color:var(--text);
}

.container{
max-width:980px;
margin:20px auto;
padding:0 16px;
}

.card{
background:var(--card);
border-radius:14px;
border:1px solid var(--border);
padding:16px;
box-shadow:0 8px 22px rgba(0,0,0,.06);
}

img{
width:100%;
border-radius:10px;
border:1px solid var(--border);
background:#111;
}

.controls{
margin-top:14px;
display:grid;
grid-template-columns:1fr 1fr;
gap:10px;
}

.controls input,.controls button{
padding:10px;
border-radius:10px;
border:1px solid var(--border);
font-size:15px;
}

.controls button{
cursor:pointer;
background:var(--accent);
color:white;
border:none;
}

.controls button.stop{
background:var(--danger);
}

#status{
margin-top:12px;
padding:10px;
border:1px dashed var(--border);
border-radius:8px;
background:#f9fafb;
white-space:pre-wrap;
font-family:Consolas,monospace;
}
</style>
</head>

<body>

<div class="container">
<div class="card">

<h1>DROL - Painel Unificado</h1>

<img src="/video_feed">

<div class="controls">

<input id="nomeSinal" placeholder="Nome do sinal">

<button onclick="registrar()">Registrar sinal</button>

<button onclick="reconhecer()">Ativar reconhecimento</button>

<button class="stop" onclick="parar()">Parar</button>

</div>

<div id="status">Carregando status...</div>

</div>
</div>

<script>

// ===============================
// VOZ DO NAVEGADOR
// ===============================

let ultimoFalado = ""
let ultimoTempo = 0

function falar(texto){

const agora = Date.now()

if(agora - ultimoTempo < 2000) return

ultimoTempo = agora

const fala = new SpeechSynthesisUtterance(texto)

fala.lang = "pt-BR"
fala.rate = 1
fala.pitch = 1

speechSynthesis.speak(fala)

}

// ===============================
// STATUS
// ===============================

async function atualizarStatus(){

try{

const res = await fetch('/status')
const data = await res.json()

document.getElementById('status').textContent =
`modo: ${data.modo}
stream_ok: ${data.stream_ok}
stream_url: ${data.stream_url}
sinal_em_registro: ${data.nome_sinal_atual || '-'}
ultimo_reconhecido: ${data.ultimo_reconhecido || '-'}
total_sinais: ${data.total_sinais}
threshold: ${data.threshold}`


// 🔊 SE RECONHECER SINAL NOVO, FALAR

if(data.ultimo_reconhecido){


falar(data.ultimo_reconhecido)

}

}catch(e){

document.getElementById('status').textContent="Falha ao buscar status"

}

}

// ===============================
// CONTROLES
// ===============================

async function registrar(){

const nome = document.getElementById('nomeSinal').value.trim()

if(!nome){
alert("Informe o nome do sinal")
return
}

await fetch(`/registrar?nome=${encodeURIComponent(nome)}`)

atualizarStatus()

}

async function reconhecer(){

await fetch('/reconhecer')

atualizarStatus()

}

async function parar(){

await fetch('/parar')

atualizarStatus()

}

// ===============================

atualizarStatus()

setInterval(atualizarStatus,1000)

</script>

</body>
</html>
"""

# ================== ROTAS ==================
@app.route("/")
def dashboard():
    return render_template_string(INDEX_HTML)


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/video")
def video_compat():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/status")
def status():
    return jsonify(status_dict())


@app.route("/registrar")
def registrar():
    global nome_sinal_atual, tempo_registro
    nome = request.args.get("nome", "").strip()
    if not nome:
        return jsonify({"ok": False, "erro": "Use /registrar?nome=A ou preencha o formulario"}), 400

    nome_sinal_atual = nome
    tempo_registro = time.time()
    set_modo("registrar", f"registro solicitado para '{nome}'")
    logger.info("Iniciando registro do sinal '%s' por %d segundos.", nome, REGISTRO_SEGUNDOS)
    return jsonify({"ok": True, "mensagem": f"Registrando sinal {nome} por {REGISTRO_SEGUNDOS} segundos"})


@app.route("/reconhecer")
def reconhecer():
    set_modo("reconhecer", "reconhecimento solicitado via rota")
    return jsonify({"ok": True, "mensagem": "Modo reconhecimento ativado"})


@app.route("/parar")
def parar():
    set_modo("normal", "parada solicitada via rota")
    return jsonify({"ok": True, "mensagem": "Modo normal ativado"})


def log_instrucoes_iniciais():
    logger.info("Servidor Flask iniciado em 0.0.0.0:5000")
    logger.info("Painel unificado: http://<IP_DO_HOST>:5000/")
    logger.info("Rotas: /registrar?nome=X | /reconhecer | /parar | /status | /video_feed")
    logger.info(
        "Config stream: CAMERA_STREAM_URL ou CAMERA_SERVER_HOST/CAMERA_SERVER_PORT/CAMERA_STREAM_PATH (Docker-friendly)."
    )
    logger.info("Modo inicial: %s | threshold reconhecimento: %.4f", modo, THRESHOLD_RECONHECIMENTO)


# ================== MAIN ==================
if __name__ == "__main__":
    log_instrucoes_iniciais()
    app.run(host="0.0.0.0", port=5000)