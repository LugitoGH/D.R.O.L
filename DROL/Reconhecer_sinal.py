import cv2
import mediapipe as mp
import os
import json
import math
import time
import logging

# ================== LOG ==================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# ================== PATH FIXO (DOCKER) ==================
DATA_PATH = "/app/DROL/data/sinais.json"
MODEL_PATH = "/app/DROL/models/hand_landmarker.task"

# ================== MEDIAPIPE ==================
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_hands=1
)

landmarker = HandLandmarker.create_from_options(options)

# ================== CARREGAR SINAIS ==================
def carregar_sinais():
    sinais = []

    if not os.path.exists(DATA_PATH):
        logging.warning("Arquivo sinais.json não encontrado.")
        return sinais

    try:
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            for linha in f:
                sinais.append(json.loads(linha))

        logging.info(f"{len(sinais)} sinais carregados.")
    except Exception as e:
        logging.error(f"Erro ao ler sinais: {e}")

    return sinais

sinais = carregar_sinais()

# ================== DISTÂNCIA ==================
def distancia(lm1, lm2):
    return math.sqrt(
        (lm1["x"] - lm2.x) ** 2 +
        (lm1["y"] - lm2.y) ** 2 +
        (lm1["z"] - lm2.z) ** 2
    )

# ================== CÂMERA ==================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    logging.error("Erro ao abrir câmera.")
    exit()

frame_id = 0
start_time = time.time()

# ================== LOOP PRINCIPAL ==================
while True:
    success, frame = cap.read()

    if not success:
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

    texto_exibido = "Faça um sinal!"

    if result.hand_landmarks and sinais:
        menor_distancia = 999
        melhor_sinal = None

        for hand in result.hand_landmarks:
            for sinal in sinais:

                soma = 0
                for i in range(21):
                    soma += distancia(
                        sinal["landmarks"][i],
                        hand[i]
                    )

                media = soma / 21

                if media < menor_distancia:
                    menor_distancia = media
                    melhor_sinal = sinal["nome"]

        # Limiar ajustável
        if menor_distancia < 0.05:
            texto_exibido = f"Sinal: {melhor_sinal}"

    # Texto no canto superior esquerdo
    cv2.putText(
        frame,
        texto_exibido,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Reconhecimento Libras", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()