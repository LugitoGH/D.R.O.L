import cv2
import mediapipe as mp
import numpy as np
import csv

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,   # AGORA ELE CAPTURA 2 MÃOS
                       min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

label = input("Nome do gesto (ex: A, B, C): ")
coleta = []

cap = cv2.VideoCapture(0)
print("Pressione 's' para salvar gesto. 'q' para sair.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # INVERTE A IMAGEM (corrige espelhamento)

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            
            # Desenha a mão na tela
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Coleta pontos da mão
            pontos = []
            for lm in hand_landmarks.landmark:
                pontos.extend([lm.x, lm.y, lm.z])

            # Salva quando apertar S (cada mão salva uma linha separada)
            if cv2.waitKey(1) & 0xFF == ord('s'):
                coleta.append([label, f"mao_{idx+1}"] + pontos)
                print(f"Gesto '{label}' da mão {idx+1} salvo!")

    cv2.imshow("Coletando dados (2 mãos)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Salvar em CSV
with open("gestos_libras_2maos.csv", "a", newline='') as f:
    writer = csv.writer(f)
    writer.writerows(coleta)

print("Dados salvos com sucesso!")