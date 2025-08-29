import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Finger tip landmarks (index from MediaPipe documentation)
finger_tips = [8, 12, 16, 20]
thumb_tip = 4

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the image and convert BGR to RGB
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process image and find hands
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, _ = frame.shape
                lm_list.append((int(lm.x * w), int(lm.y * h)))

            # Count fingers
            fingers = []

            # Thumb
            if lm_list[thumb_tip][0] > lm_list[thumb_tip - 1][0]:
                fingers.append(1)
            else:
                fingers.append(0)

            # Fingers
            for tip in finger_tips:
                if lm_list[tip][1] < lm_list[tip - 2][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            total_fingers = fingers.count(1)

            # Display result
            cv2.putText(frame, f'Fingers: {total_fingers}', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Draw hand landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Gesture Control", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
