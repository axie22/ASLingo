import cv2
import mediapipe as mp
import os
import time
from keras.models import load_model
import numpy as np

model = load_model('asl_model.h5')

class_labels = [str(i) for i in range(10)] + [chr(c) for c in range(ord('a'), ord('z') + 1)]


mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# save_dir = "asl_frames"
# os.makedirs(save_dir, exist_ok=True)

# mediapipe hands model
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)


cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # convert the frame to RGB 
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # draw hand landmarks on the frame if hand is found
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            h, w, _ = frame.shape
            x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
            y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h)
            x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w)
            y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h)

            # Crop and preprocess hand region
            hand_img = frame[y_min:y_max, x_min:x_max]
            hand_img = cv2.resize(hand_img, (64, 64))
            hand_img = hand_img / 255.0  # Normalize
            hand_img = np.expand_dims(hand_img, axis=0)  # Add batch dimension

            # Predict
            prediction = model.predict(hand_img)
            predicted_index = np.argmax(prediction)
            predicted_label = class_labels[predicted_index]

            # Display prediction
            cv2.putText(frame, f"Predicted: {predicted_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('ASL Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
