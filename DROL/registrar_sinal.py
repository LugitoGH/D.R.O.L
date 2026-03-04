import cv2
import mediapipe as mp
import os
import time
import json
import math
import logging
from flask import Flask, Response, request

# ================== LOG CONFIG ==================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

app = Flask(__name__)

# ================== PATHS SEGUROS ==================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "hand_landmarker.task")
DATA_DIR = os.path.join(BASE_DIR, "data")
SINAIS_PATH = os.path.join(DATA_DIR, "sinais.json")

os.makedirs(DATA_DIR, exist_ok=True)

# ================== MEDIAPIPE ==================
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

try:
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_hands=1
    )
    landmarker = HandLandmarker.create_from_options(options)
    logging.info("MediaPipe carregado com sucesso.")
except Exception as e:
    logging.error(f"Erro ao carregar modelo MediaPipe: {e}")
    raise

# ================== CAMERA ==================
cap = cv2.VideoCapture(
    "http://10.135.10.18:5000/video",
    cv2.CAP_FFMPEG
)

if not cap.isOpened():
    logging.error("Erro ao abrir stream HTTP")
    exit()

# ================== VARIÁVEIS ==================
frame_id = 0
start_time = time.time()

modo = "normal"
nome_sinal_atual = ""
tempo_registro = 0
sinais = []

# ================== FUNÇÕES ==================

def carregar_sinais():
    global sinais
    sinais = []
    try:
        if os.path.exists(SINAIS_PATH):
            with open(SINAIS_PATH, "r", encoding="utf-8") as f:
                for linha in f:
                    sinais.append(json.loads(linha))
        logging.info(f"{len(sinais)} sinais carregados.")
    except Exception as e:
        logging.error(f"Erro ao carregar sinais: {e}")

def salvar_sinal(nome, coordenadas):
    try:
        with open(SINAIS_PATH, "a", encoding="utf-8") as f:
            json.dump({
                "nome": nome,
                "landmarks": coordenadas
            }, f)
            f.write("\n")
        logging.info(f"Sinal '{nome}' salvo com sucesso.")
        print("Sinal capturado!")
    except Exception as e:
        logging.error(f"Erro ao salvar sinal: {e}")

def distancia(lm1, lm2):
    return math.sqrt(
        (lm1["x"] - lm2.x) ** 2 +
        (lm1["y"] - lm2.y) ** 2 +
        (lm1["z"] - lm2.z) ** 2
    )

carregar_sinais()

# ================== STREAM ==================

def generate_frames():
    global frame_id, modo, tempo_registro

    while True:
        try:
            success, frame = cap.read()

            if not success:
                logging.warning("Falha ao capturar frame.")
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            frame_id += 1
            timestamp_ms = int(time.time() * 1000)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.hand_landmarks:
                for hand in result.hand_landmarks:

                    h, w, _ = frame.shape
                    coordenadas = []

                    for lm in hand:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

                        coordenadas.append({
                            "x": lm.x,
                            "y": lm.y,
                            "z": lm.z
                        })

                    # ================= REGISTRAR =================
                    if modo == "registrar":
                        if time.time() - tempo_registro >= 5:
                            salvar_sinal(nome_sinal_atual, coordenadas)
                            carregar_sinais()
                            modo = "normal"

                    # ================= RECONHECER =================
                    if modo == "reconhecer":
                        for sinal in sinais:
                            soma = 0
                            for i in range(21):
                                soma += distancia(
                                    sinal["landmarks"][i],
                                    hand[i]
                                )
                            media = soma / 21

                            if media < 0.05:
                                cv2.putText(
                                    frame,
                                    f"Sinal: {sinal['nome']}",
                                    (20, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1,
                                    (0, 255, 0),
                                    2
                                )

        except Exception as e:
            logging.error(f"Erro no loop principal: {e}")

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# ================== ROTAS ==================

@app.route('/')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/registrar')
def registrar():
    global modo, nome_sinal_atual, tempo_registro

    nome = request.args.get("nome")

    if not nome:
        return "Use: /registrar?nome=A"

    nome_sinal_atual = nome
    tempo_registro = time.time()
    modo = "registrar"

    logging.info(f"Iniciando registro do sinal '{nome}'")
    return f"Registrando sinal {nome} por 5 segundos..."

@app.route('/reconhecer')
def reconhecer():
    global modo
    modo = "reconhecer"
    logging.info("Modo reconhecimento ativado.")
    return "Modo reconhecimento ativado!"

@app.route('/parar')
def parar():
    global modo
    modo = "normal"
    logging.info("Modo normal ativado.")
    return "Modo normal."

# ================== MAIN ==================

if __name__ == "__main__":
    logging.info("Servidor iniciado.")
    app.run(host="0.0.0.0", port=5000)