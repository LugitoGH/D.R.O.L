import cv2
import mediapipe as mp
import os
import time
import json
import math
import logging
import socket
import numpy as np
from flask import Flask, Response, jsonify, render_template_string, request, send_file
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
MAX_IMPORT_BYTES = 2 * 1024 * 1024
MAX_SINAIS_IMPORTADOS = 1000
MAX_FRAMES_MOVIMENTO = 300
VETOR_SINAL_TAMANHO = 63
TIPOS_SINAIS_PERMITIDOS = {"sinal", "movimento"}

os.makedirs(DATA_DIR, exist_ok=True)

# ================== MEDIAPIPE ==================
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

movimentos = []
global hand0
moveDb = False
hand0 = None
gravando_movimento = False
detectando_movimento = False

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


def salvar_sinal(tipo, nome, vetor):
    try:
        with open(SINAIS_PATH, "a", encoding="utf-8") as f:
            json.dump({"tipo": tipo, "nome": nome, "vetor": vetor}, f)
            f.write("\n")
        logger.info("Sinal '%s' salvo com sucesso.", nome)
    except Exception as e:
        logger.error("Erro ao salvar sinal '%s': %s", nome, e)


def normalizar_conteudo_sinais(conteudo):
    if len(conteudo) > MAX_IMPORT_BYTES:
        raise ValueError("Arquivo muito grande. Limite: 2 MB.")

    texto = conteudo.decode("utf-8-sig").strip()
    if not texto:
        return []

    try:
        dados = json.loads(texto)
        if isinstance(dados, dict):
            dados = [dados]
        if not isinstance(dados, list):
            raise ValueError("O JSON deve conter um objeto, uma lista ou linhas JSON.")
        return dados
    except json.JSONDecodeError:
        sinais_importados = []
        for numero_linha, linha in enumerate(texto.splitlines(), start=1):
            linha = linha.strip()
            if not linha:
                continue
            try:
                sinais_importados.append(json.loads(linha))
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON invalido na linha {numero_linha}.") from exc
        return sinais_importados


def validar_nome_sinal(nome, indice):
    if not isinstance(nome, str):
        raise ValueError(f"Sinal {indice}: o campo 'nome' deve ser texto.")

    nome = nome.strip()
    if not nome:
        raise ValueError(f"Sinal {indice}: o campo 'nome' nao pode ficar vazio.")

    if len(nome) > 80:
        raise ValueError(f"Sinal {indice}: o campo 'nome' deve ter ate 80 caracteres.")

    if any(ord(char) < 32 for char in nome):
        raise ValueError(f"Sinal {indice}: o campo 'nome' contem caracteres invalidos.")

    return nome


def validar_vetor_numerico(vetor, indice, contexto):
    if not isinstance(vetor, list) or len(vetor) != VETOR_SINAL_TAMANHO:
        raise ValueError(f"Sinal {indice}: {contexto} deve ter {VETOR_SINAL_TAMANHO} numeros.")

    vetor_validado = []
    for posicao, valor in enumerate(vetor, start=1):
        if not isinstance(valor, (int, float)) or isinstance(valor, bool) or not math.isfinite(valor):
            raise ValueError(f"Sinal {indice}: valor invalido em {contexto}[{posicao}].")
        vetor_validado.append(float(valor))

    return vetor_validado


def validar_sinal_importado(sinal, indice):
    if not isinstance(sinal, dict):
        raise ValueError(f"Sinal {indice}: cada item deve ser um objeto JSON.")

    campos = set(sinal.keys())
    campos_esperados = {"tipo", "nome", "vetor"}
    if campos != campos_esperados:
        raise ValueError(f"Sinal {indice}: use apenas os campos tipo, nome e vetor.")

    tipo = sinal["tipo"]
    if tipo not in TIPOS_SINAIS_PERMITIDOS:
        raise ValueError(f"Sinal {indice}: tipo deve ser 'sinal' ou 'movimento'.")

    nome = validar_nome_sinal(sinal["nome"], indice)
    vetor = sinal["vetor"]

    if tipo == "sinal":
        vetor_validado = validar_vetor_numerico(vetor, indice, "vetor")
    else:
        if not isinstance(vetor, list) or not vetor:
            raise ValueError(f"Sinal {indice}: movimento deve ter uma lista de frames.")
        if len(vetor) > MAX_FRAMES_MOVIMENTO:
            raise ValueError(f"Sinal {indice}: movimento deve ter ate {MAX_FRAMES_MOVIMENTO} frames.")
        vetor_validado = [
            validar_vetor_numerico(frame, indice, f"vetor[{frame_indice}]")
            for frame_indice, frame in enumerate(vetor, start=1)
        ]

    return {"tipo": tipo, "nome": nome, "vetor": vetor_validado}


def validar_sinais_importados(sinais_importados):
    if len(sinais_importados) > MAX_SINAIS_IMPORTADOS:
        raise ValueError(f"Importacao limitada a {MAX_SINAIS_IMPORTADOS} sinais por arquivo.")

    return [
        validar_sinal_importado(sinal, indice)
        for indice, sinal in enumerate(sinais_importados, start=1)
    ]


def substituir_sinais(sinais_importados):
    with open(SINAIS_PATH, "w", encoding="utf-8") as f:
        for sinal in sinais_importados:
            json.dump(sinal, f, ensure_ascii=False)
            f.write("\n")
    carregar_sinais()


def normalizar_e_vetorizar(hand_landmarks, centralpoint):
    """
    Recebe lista de landmarks do MediaPipe.
    Retorna vetor normalizado com 63 valores.
    """

    # 1️⃣ Centralização (wrist como referência)
    wrist = centralpoint

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
        "detectando_movimento": detectando_movimento,
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

movimento_atual = None
sinal_movimento_atual = None

texto_exibido = ""
texto_att = None

# ================== STREAM ==================
def generate_frames():
    global frame_id, tempo_registro, ultimo_reconhecido, cap, stream_url_ativo, falhas_stream, hand0, moveDb, movimentos, gravando_movimento, detectando_movimento, movimento_atual, sinal_movimento_atual, texto_exibido, texto_att

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
            frame = cv2.resize(frame, (640, 360))
            
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            frame_id += 1
            timestamp_ms = int(time.time() * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)


            # resultHead = landmarkerHead.detect_for_video(mp_image, timestamp_ms)

                
            # if resultHead.face_landmarks:
            #     for face in resultHead.face_landmarks:

            #         lm_points = [(int(lm.x*w), int(lm.y*h)) for lm in face]

            #         for (i,j) in face_edges:
            #             cv2.line(frame, lm_points[i], lm_points[j], (255,0,0), 1)
            
            

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
                            vetor = normalizar_e_vetorizar(hand, hand[0])
                            salvar_sinal("sinal", nome_sinal_atual, vetor)
                            carregar_sinais()
                            set_modo("normal", "registro finalizado")

                        

                        if modo == "registrarMove":
                            if moveDb == False:
                                moveDb = True
                                movimentos = []
                                
                                vetor = normalizar_e_vetorizar(hand, hand[0])
                                hand0 = hand
                                movimentos.append(vetor)

                            if moveDb == True and gravando_movimento == True:
                                vetor = normalizar_e_vetorizar(hand, hand0[0])
                                movimentos.append(vetor)

                            if gravando_movimento == False:
                                vetor = normalizar_e_vetorizar(hand, hand0[0])
                                movimentos.append(vetor)
                                movimentos = [movimentos[0], movimentos[len(movimentos)//2], movimentos[-1]]
                                salvar_sinal("movimento", nome_sinal_atual, movimentos)
                                set_modo("normal", "registro finalizado")
                                carregar_sinais()
                                moveDb = False

                        if modo == "reconhecer":

                            vetor_atual = normalizar_e_vetorizar(hand, hand[0])

                            melhor_distancia = float("inf")
                            melhor_sinal = None
                            melhor_sinalMove = None
                            melhor_dist_Move = float("inf")

                            if detectando_movimento == False:

                                for sinal in sinais:
                                    if (sinal.get("tipo") == "movimento"):
                                        vetor_salvo = sinal["vetor"][0]
                                        
                                        if len(vetor_salvo) != 63:
                                            continue
                                        
                                        soma = 0
                                        for i in range(63):
                                            soma += (vetor_salvo[i] - vetor_atual[i]) ** 2

                                        distancia_media = math.sqrt(soma / 63)

                                        if distancia_media < melhor_dist_Move:
                                            
                                            melhor_dist_Move = distancia_media
                                            melhor_sinalMove = sinal["nome"]
                                            movimento_atual = melhor_sinalMove
                                
                                    if sinal.get("tipo") == "sinal" and detectando_movimento == False:
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
                                            
                            

                                        
                                        
                                if melhor_dist_Move < THRESHOLD_RECONHECIMENTO:
                                    detectando_movimento = True
                                    sinal_movimento_atual = sinal
                                    movimentos = []
                                    hand0 = hand
                                    texto = melhor_sinalMove
                                    cv2.putText(
                                        frame,
                                        f"Sinal: {melhor_sinalMove}",
                                        (20, 40),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        1,
                                        (0, 255, 0),
                                        2,
                                    )
                                
                                if melhor_distancia < THRESHOLD_RECONHECIMENTO * 1.35 and detectando_movimento == False:
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
                            
                            if detectando_movimento == True:
                                
                                
                                vetor_atual = normalizar_e_vetorizar(hand, hand0[0])

                                sinal = sinal_movimento_atual

                                if sinal is None:
                                    detectando_movimento = False
                                    continue

                                vetor_inicio = sinal["vetor"][0]
                                vetor_meio = sinal["vetor"][1]
                                vetor_fim = sinal["vetor"][2]

                                if len(vetor_fim) != 63:
                                    detectando_movimento = False
                                    continue


                                def dist(v1, v2):
                                    soma = 0
                                    for i in range(63):
                                        soma += (v1[i] - v2[i]) ** 2
                                    return math.sqrt(soma / 63)


                                dist_inicio = dist(vetor_atual, vetor_inicio)
                                dist_fim = dist(vetor_atual, vetor_fim)

                                cv2.putText(
                                    frame,
                                    "Detectando Movimento",
                                    (20, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1,
                                    (0, 255, 0),
                                    2,
                                )

                                if dist_inicio > 0.6 and dist_fim > 0.6:
                                    detectando_movimento = False
                                    movimentos = []
                                    movimento_atual = None
                                    sinal_movimento_atual = None
                                    continue

                                if dist_fim < THRESHOLD_RECONHECIMENTO * 1.7:
                                    detectando_movimento = False
                                    texto = movimento_atual
                                    texto_exibido = texto
                                    texto_att = time.time() + 2

                                    movimentos = []
                                    movimento_atual = None
                                    sinal_movimento_atual = None

                                else:
                                    movimentos.append(vetor_atual)

                                    if len(movimentos) > 4000:
                                        detectando_movimento = False
                                        movimentos = []
                                        movimento_atual = None
                                        sinal_movimento_atual = None



                            #logger.info("Melhor distancia: %.4f", melhor_distancia)

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

            if texto_att is not None and time.time() < texto_att:
                cv2.putText(
                frame,
                f"Sinal: {texto_exibido}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                )
        except Exception as e:
            logger.error("Erro no loop principal: %s", e)

        if frame is None:
            frame = frame_aguardando_stream()

        _, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
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
--bg-start:#a662ee;
--bg-end:#b46bf4;
--header:#6f068d;
--header-dark:#5d0477;
--card:#ffffff;
--text:#111111;
--border:#111111;
--purple:#6d05d7;
--purple-dark:#6a098f;
--danger:#ff140b;
--shadow:0 22px 50px rgba(64, 0, 94, 0.18);
--radius-xl:42px;
--radius-lg:34px;
}

*{
box-sizing:border-box;
}

body{
margin:0;
font-family:"Trebuchet MS","Century Gothic",Verdana,sans-serif;
background:linear-gradient(90deg,var(--bg-start),var(--bg-end));

background-image:
linear-gradient(#36075edb, rgb(52 0 96 / 81%)), url(/static/background.png);

background-opacity: 50%;
background-size:cover;
background-position:center;
background-repeat:no-repeat;
color:var(--text);
min-height:100vh;
}

.container{
min-height:100vh;
}

.topbar{
background:linear-gradient(90deg,var(--header-dark),var(--header));
padding:18px 34px;
display:flex;
align-items:center;
gap:28px;
color:#fff;
box-shadow:0 10px 24px rgba(61, 0, 79, 0.28);
}

.brand-mark{
font-size:5rem;
font-weight:900;
line-height:0.8;
letter-spacing:-4px;
position:relative;
text-transform:lowercase;
}

.divider{
width:4px;
align-self:stretch;
background:rgba(255,255,255,.85);
border-radius:999px;
}

.topbar-title{
margin:0;
font-size:1.85rem;
line-height:1.2;
font-weight:900;
letter-spacing:0.02em;
text-transform:uppercase;
}

.dashboard{
max-width:1700px;
margin:0 auto;
padding:60px 42px 48px;
display:grid;
grid-template-columns:minmax(0,2.3fr) minmax(300px,.95fr);
gap:36px;
}

.panel{
background:var(--card);
border-radius:var(--radius-xl);
padding:28px 30px 34px;
box-shadow:var(--shadow);
height:fit-content;
}

.panel-title{
display:flex;
align-items:center;
gap:18px;
margin:0 0 22px;
font-size:2rem;
font-weight:900;
}

.camera-badge{
display:inline-flex;
align-items:center;
justify-content:center;
width:42px;
height:42px;
border-radius:10px;
background:linear-gradient(180deg,#dc1df7,var(--purple));
position:relative;
flex-shrink:0;
}

.camera-badge::after{
content:"";
position:absolute;
right:-12px;
width:0;
height:0;
border-top:10px solid transparent;
border-bottom:10px solid transparent;
border-left:14px solid #cf1cf1;
}

.video-shell{
border:4px solid var(--border);
border-radius:var(--radius-lg);
overflow:hidden;
background:#f6f3fb;
min-height:520px;
display:flex;
}

.language-panel{
margin-top:24px;
display:grid;
gap:22px;
}

.language-label{
font-size:1.05rem;
font-weight:900;
color:#241034;
}

.language-select-wrap{
position:relative;
}

.language-select-wrap::after{
content:"";
position:absolute;
right:24px;
top:50%;
width:10px;
height:10px;
border-right:3px solid #fff;
border-bottom:3px solid #fff;
transform:translateY(-65%) rotate(45deg);
pointer-events:none;
}

.language-select{
width:100%;
min-height:64px;
padding:16px 58px 16px 22px;
border:none;
border-radius:18px;
background:linear-gradient(180deg,var(--purple-dark),#51106b);
color:#fff;
font-family:inherit;
font-size:1.05rem;
font-weight:900;
appearance:none;
outline:none;
box-shadow:0 12px 24px rgba(82, 0, 145, 0.18);
cursor:pointer;
}

.language-select option{
background:#2d3540;
color:#fff;
font-weight:800;
}

.custom-sign-actions{
display:grid;
grid-template-columns:repeat(3,minmax(0,1fr));
gap:22px;
}

.custom-sign-actions.is-hidden{
display:none;
}

.custom-sign-actions button{
min-height:58px;
padding:14px 18px;
border:none;
border-radius:18px;
background: #6a098f;
color:#fff;
font-family:inherit;
font-size:1rem;
font-weight:900;
box-shadow:0 12px 24px rgba(82, 0, 145, 0.18);
cursor:pointer;
transition:transform .18s ease,box-shadow .18s ease,filter .18s ease;
}

.custom-sign-actions button:hover{
transform:translateY(-2px);
filter:brightness(1.02);
}

.custom-sign-actions button:active{
transform:translateY(0);
}

img{
width:100%;
height:100%;
display:block;
object-fit:cover;
background:#f6f3fb;
}

.sidebar{
display:flex;
flex-direction:column;
gap:22px;
}

.sidebar-title{
margin:0 0 18px;
font-size:1.95rem;
font-weight:900;
text-align:center;
}

.controls{
display:grid;
gap:18px;
}

.controls input,.controls button{
width:100%;
min-height:82px;
padding:18px 22px;
border:none;
border-radius:22px;
font-size:1.15rem;
font-family:inherit;
font-weight:800;
text-align:center;
}

.controls input{
background:#f7f2fd;
color:#42205f;
border:3px solid rgba(109,5,215,.14);
text-align:left;
padding-left:24px;
outline:none;
}

.controls input::placeholder{
color:#7d5a9c;
}

.controls button{
cursor:pointer;
color:white;
box-shadow:0 14px 28px rgba(82, 0, 145, 0.18);
transition:transform .18s ease,box-shadow .18s ease,filter .18s ease;
}

.controls button:hover{
transform:translateY(-2px);
filter:brightness(1.02);
}

.controls button:active{
transform:translateY(0);
}

.controls button:nth-of-type(1){
background:linear-gradient(180deg,var(--purple),var(--purple));
}

.controls button:nth-of-type(2){
background:linear-gradient(180deg,var(--purple-dark),var(--purple-dark));
}

.controls button:nth-of-type(3){
background:linear-gradient(180deg, #FFC107, #FFC107);
}

.controls button:nth-of-type(4){
background:linear-gradient(180deg, #006400, #006400);
}

.controls button.stop{
background:var(--danger);
}

#status{
margin-top:8px;
padding:24px 22px;
min-height:280px;
border:4px solid var(--border);
border-radius:var(--radius-lg);
background:#fff;
white-space:pre-wrap;
font-family:Consolas,"Courier New",monospace;
font-size:.98rem;
line-height:1.55;
overflow:auto;
}

.toast-container{
position:fixed;
right:24px;
bottom:24px;
display:grid;
gap:12px;
z-index:9999;
}

.toast{
min-width:280px;
max-width:360px;
padding:16px 18px;
border-radius:20px;
color:#fff;
box-shadow:0 18px 34px rgba(34, 0, 57, 0.28);
font-weight:800;
line-height:1.35;
animation:toast-in .22s ease;
}

.toast--success{
background:linear-gradient(180deg,#7e17ff,var(--purple));
}

.toast--info{
background:linear-gradient(180deg,#7d109f,var(--purple-dark));
}

.toast--danger{
background:linear-gradient(180deg,#ff5b53,var(--danger));
}

@keyframes toast-in{
from{
opacity:0;
transform:translateY(10px);
}
to{
opacity:1;
transform:translateY(0);
}
}

@media (max-width: 1180px){
.topbar{
padding:18px 24px;
flex-wrap:wrap;
gap:18px;
}

.divider{
display:none;
}

.topbar-title{
font-size:1.45rem;
}

.video-shell{
min-height:400px;
}
}

@media (max-width: 720px){
.brand-mark{
font-size:3.6rem;
letter-spacing:-3px;
}

.panel{
padding:20px 18px 24px;
border-radius:30px;
height:fit-content;
}

.panel-title,.sidebar-title{
font-size:1.7rem;
}

.controls input,.controls button{
min-height:68px;
font-size:1rem;
}

#btnImp{
background-color: #006400;
}

.video-shell,#status{
border-radius:28px;
}

.language-select{
min-height:58px;
font-size:1rem;
}

.custom-sign-actions{
grid-template-columns:1fr;
}

.toast-container{
left:18px;
right:18px;
bottom:18px;
}

.toast{
min-width:auto;
max-width:none;
}
}
</style>
</head>

<body>

<div class="container">
<header class="topbar">
<div class="brand-mark"><img src="{{url_for('static', filename='logo.png') }}" alt="Logo" style="width:120px; background-color:#f6f3fb00; height:auto; object-fit:contain;"/></div>
<div class="divider"></div>
<h1 class="topbar-title">Dispositivo de Reconhecimento Orientação e Tradução de Libras</h1>
</header>

<main class="dashboard">
<section class="panel">
<h2 class="panel-title">
<span><img src="{{url_for('static', filename='video_icon.png')}}" alt="Camera Icone" style="width:42px; height:42px; background-color:#ffff0000; object-fit:contain;"/></span>
<span>Reconhecimento de libras</span>
</h2>

<div class="video-shell">
<img src="/video_feed" alt="Transmissao da camera do DROL">
</div>

<div class="language-panel">
<label class="language-label" for="signLanguage">Linguagem de sinais</label>
<div class="language-select-wrap">
<select id="signLanguage" class="language-select" aria-label="Selecionar linguagem de sinais">
<option value="personalizado" selected>Personalizado</option>
<option value="libras">LIBRAS</option>
<option value="asl">ASL</option>
<option value="bsl">BSL</option>
<option value="lsf">LSF</option>
</select>
</div>

<div id="customSignActions" class="custom-sign-actions">
<button type="button" onclick="abrirImportacaoSinais()">Importar Sinais</button>
<button type="button" onclick="exportarSinais()">Exportar Sinais</button>
<button type="button" onclick="limparSinais()">Limpar Sinais</button>
</div>
<input id="arquivoSinais" type="file" accept=".json,application/json" hidden>
</div>
</section>

<aside class="panel sidebar">
<h2 class="sidebar-title">Controles</h2>

<div class="controls">
<input id="nomeSinal" placeholder="Nome do sinal">

<button onclick="reconhecer()">Reconhecer sinal</button>

<button onclick="registrar()">Registrar sinal</button>

<button id="btnMove" onclick="registrarMove()">Registrar movimento</button>

<button class="stop" onclick="parar()">Reiniciar</button>
</div>

<div id="status">Carregando status...</div>
</aside>
</main>
<div id="toastContainer" class="toast-container" aria-live="polite" aria-atomic="true"></div>
</div>

<script>

// ===============================
// VOZ DO NAVEGADOR
// ===============================

let ultimoFalado = ""
let ultimoTempo = 0
let ultimoReconhecidoNotificado = ""
let ultimoStatus = null
let ultimoNomeRegistrado = ""

function atualizarControlesLinguagem(){

const select = document.getElementById('signLanguage')
const actions = document.getElementById('customSignActions')

if(!select || !actions) return

actions.classList.toggle('is-hidden', select.value !== 'personalizado')

}

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

function notificar(mensagem, tipo="info"){

const container = document.getElementById('toastContainer')
const toast = document.createElement('div')

toast.className = `toast toast--${tipo}`
toast.textContent = mensagem

container.appendChild(toast)

setTimeout(() => {
toast.style.opacity = '0'
toast.style.transform = 'translateY(10px)'
toast.style.transition = 'opacity .2s ease, transform .2s ease'
}, 3200)

setTimeout(() => {
toast.remove()
}, 3450)

}

// ===============================
// STATUS
// ===============================

async function atualizarStatus(){

try{

const res = await fetch('/status')
const data = await res.json()

const registroFinalizado =
ultimoStatus &&
ultimoStatus.modo === 'registrar' &&
data.modo === 'normal' &&
data.total_sinais > ultimoStatus.total_sinais

if(registroFinalizado){
const nomeRegistrado = ultimoStatus.nome_sinal_atual || ultimoNomeRegistrado || 'Sinal'
notificar(`${nomeRegistrado} registrado com sucesso`, 'success')
ultimoNomeRegistrado = ""
}

document.getElementById('status').textContent =
`modo: ${data.modo}
detectando_movimento: ${data.detectando_movimento}
stream_ok: ${data.stream_ok}
stream_url: ${data.stream_url}
sinal_em_registro: ${data.nome_sinal_atual || '-'}
ultimo_reconhecido: ${data.ultimo_reconhecido || '-'}
total_sinais: ${data.total_sinais}
threshold: ${data.threshold}`


// 🔊 SE RECONHECER SINAL NOVO, FALAR

if(data.ultimo_reconhecido){

if(data.ultimo_reconhecido !== ultimoReconhecidoNotificado){
notificar(`Sinal reconhecido: ${data.ultimo_reconhecido}`, 'info')
ultimoReconhecidoNotificado = data.ultimo_reconhecido
}

falar(data.ultimo_reconhecido)

}
else{
ultimoReconhecidoNotificado = ""
}

ultimoStatus = data

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
notificar("Informe o nome do sinal", 'info')
return
}

await fetch(`/registrar?nome=${encodeURIComponent(nome)}`)

}

async function registrarMove(){

const nome = document.getElementById('nomeSinal').value.trim()

const res = await fetch(`/registrarMove?nome=${encodeURIComponent(nome)}`)
const data = await res.json()

notificar(data.mensagem, 'info')

// troca texto do botão
const btn = document.getElementById('btnMove')

if(btn.textContent.includes("Registrar")){
    btn.textContent = "Parar movimento"
    btn.style.background = "#ff140b"
}else{
    btn.textContent = "Registrar movimento"
    btn.style.background = "linear-gradient(180deg, #FFD54F, #FFC107)"
}

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

function abrirImportacaoSinais(){

const input = document.getElementById('arquivoSinais')
input.value = ""
input.click()

}

async function importarSinais(event){

const arquivo = event.target.files[0]

if(!arquivo) return

if(!arquivo.name.toLowerCase().endsWith('.json')){
notificar("Selecione um arquivo .json", 'info')
return
}

const formData = new FormData()
formData.append('arquivo', arquivo)

try{
const res = await fetch('/importar_sinais', {
method: 'POST',
body: formData
})
const data = await res.json()

if(!res.ok || !data.ok){
notificar(data.erro || "Falha ao importar sinais", 'danger')
return
}

notificar(data.mensagem, 'success')
atualizarStatus()
}catch(e){
notificar("Falha ao importar sinais", 'danger')
}

}

function exportarSinais(){

window.location.href = '/exportar_sinais'

}

async function limparSinais(){

if(!confirm("Deseja apagar todos os sinais salvos?")) return

try{
const res = await fetch('/limpar_sinais', {method: 'POST'})
const data = await res.json()

if(!res.ok || !data.ok){
notificar(data.erro || "Falha ao limpar sinais", 'danger')
return
}

notificar(data.mensagem, 'success')
ultimoReconhecidoNotificado = ""
atualizarStatus()
}catch(e){
notificar("Falha ao limpar sinais", 'danger')
}

}

// ===============================

document.getElementById('signLanguage').addEventListener('change', atualizarControlesLinguagem)
document.getElementById('arquivoSinais').addEventListener('change', importarSinais)
atualizarControlesLinguagem()
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


@app.route("/importar_sinais", methods=["POST"])
def importar_sinais():
    arquivo = request.files.get("arquivo")

    if not arquivo or not arquivo.filename:
        return jsonify({"ok": False, "erro": "Envie um arquivo .json."}), 400

    if not arquivo.filename.lower().endswith(".json"):
        return jsonify({"ok": False, "erro": "O arquivo deve ser do tipo .json."}), 400

    try:
        sinais_importados = validar_sinais_importados(normalizar_conteudo_sinais(arquivo.read()))
        substituir_sinais(sinais_importados)
        logger.info("%d sinais importados para %s.", len(sinais_importados), SINAIS_PATH)
        return jsonify({"ok": True, "mensagem": f"{len(sinais_importados)} sinais importados com sucesso."})
    except UnicodeDecodeError:
        return jsonify({"ok": False, "erro": "O arquivo precisa estar em UTF-8."}), 400
    except ValueError as e:
        return jsonify({"ok": False, "erro": str(e)}), 400
    except Exception as e:
        logger.error("Erro ao importar sinais: %s", e)
        return jsonify({"ok": False, "erro": "Erro interno ao importar sinais."}), 500


@app.route("/exportar_sinais")
def exportar_sinais():
    if not os.path.exists(SINAIS_PATH):
        open(SINAIS_PATH, "a", encoding="utf-8").close()

    return send_file(
        SINAIS_PATH,
        mimetype="application/json",
        as_attachment=True,
        download_name="sinais.json",
    )


@app.route("/limpar_sinais", methods=["POST"])
def limpar_sinais():
    global ultimo_reconhecido

    try:
        open(SINAIS_PATH, "w", encoding="utf-8").close()
        carregar_sinais()
        ultimo_reconhecido = ""
        logger.info("Arquivo de sinais limpo: %s", SINAIS_PATH)
        return jsonify({"ok": True, "mensagem": "Sinais apagados com sucesso."})
    except Exception as e:
        logger.error("Erro ao limpar sinais: %s", e)
        return jsonify({"ok": False, "erro": "Erro interno ao limpar sinais."}), 500


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

@app.route("/registrarMove")
def registrarMove():
    global nome_sinal_atual, tempo_registro, gravando_movimento, moveDb, movimentos

    nome = request.args.get("nome", "").strip()

    if not gravando_movimento:
        if not nome:
            return jsonify({"ok": False, "erro": "Informe o nome do sinal"}), 400

        nome_sinal_atual = nome
        tempo_registro = time.time()
        gravando_movimento = True
        moveDb = False
        movimentos = []

        set_modo("registrarMove", f"iniciando '{nome}'")
        logger.info("Iniciando gravação de movimento: %s", nome)

        return jsonify({"ok": True, "mensagem": f"Gravando movimento {nome}..."})

    else:
        gravando_movimento = False

        return jsonify({"ok": True, "mensagem": "Gravação finalizada"})


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
