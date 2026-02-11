import cv2
import mediapipe as mp
import os

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "hand_landmarker.task")

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_hands=1
)

landmarker = HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
frame_id = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ✅ FORMA CORRETA
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

    result = landmarker.detect_for_video(mp_image, timestamp_ms)
    #frame_id += 1

    # Se reconhecer os 21 pontos da mão...
    if result.hand_landmarks:

        #Irá desenhar os pontos...
        for hand in result.hand_landmarks:

            for lm in hand:
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

            #Extrair vetor...
            pontos = []
            for lm in hand:
                pontos.extend([lm.x, lm.y, lm.z])

            print(len(pontos))  # sempre 63

            cv2.putText(
                frame,
                "Mao detectada", #E exibir a mensagem no canto superior esquerdo!
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

    cv2.imshow("MediaPipe Tasks - Hands", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
landmarker.close()
cv2.destroyAllWindows()
