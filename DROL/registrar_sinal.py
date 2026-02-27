import cv2
import mediapipe as mp
import os
import time
import json
import math
from flask import Flask, Response, request

app = Flask(__name__)

# ===== MediaPipe Setup =====
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "hand_landmarker.task")

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_hands=1
)

landmarker = HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(
    "http://10.135.10.205:5000/video",
    cv2.CAP_FFMPEG
)

if not cap.isOpened():
    print("Erro ao abrir stream HTTP")
    exit()

frame_id = 0
start_time = time.time()

# ===== NOVAS VARIÁVEIS =====
modo = "normal"  # normal | registrar | reconhecer
nome_sinal_atual = ""
tempo_registro = 0
sinais = []

# ===== FUNÇÕES =====

def carregar_sinais():
    global sinais
    sinais = []
    if os.path.exists("sinais.json"):
        with open("sinais.json", "r") as f:
            for linha in f:
                sinais.append(json.loads(linha))

def salvar_sinal(nome, coordenadas):
    with open("sinais.json", "a") as f:
        json.dump({
            "nome": nome,
            "landmarks": coordenadas
        }, f)
        f.write("\n")

def distancia(lm1, lm2):
    return math.sqrt(
        (lm1["x"] - lm2.x) ** 2 +
        (lm1["y"] - lm2.y) ** 2 +
        (lm1["z"] - lm2.z) ** 2
    )

carregar_sinais()

# ===== STREAM =====

def generate_frames():
    global frame_id, modo, tempo_registro

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        frame_id += 1
        timestamp_ms = int((time.time() - start_time) * 1000)

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

                # ===== MODO REGISTRAR =====
                if modo == "registrar":
                    restante = 10 - int(time.time() - tempo_registro)

                    cv2.putText(frame,
                                f"Registrando {nome_sinal_atual} em {restante}s",
                                (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0, 255, 255),
                                2)

                    if time.time() - tempo_registro >= 5:
                        salvar_sinal(nome_sinal_atual, coordenadas)
                        carregar_sinais()
                        modo = "normal"

                # ===== MODO RECONHECER =====
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

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# ===== ROTAS =====

@app.route('/')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/registrar')
def registrar():
    global modo, nome_sinal_atual, tempo_registro

    nome = request.args.get("nome")

    if not nome:
        return "Passe o nome do sinal na URL: /registrar?nome=C"

    nome_sinal_atual = nome
    tempo_registro = time.time()
    modo = "registrar"

    return f"Registrando sinal {nome} por 5 segundos..."

@app.route('/reconhecer')
def reconhecer():
    global modo
    modo = "reconhecer"
    return "Modo reconhecimento ativado!"

@app.route('/parar')
def parar():
    global modo
    modo = "normal"
    return "Modo normal."

# ===== MAIN =====
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)