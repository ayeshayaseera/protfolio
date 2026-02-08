import cv2
import mediapipe as mp
import numpy as np
import time
import threading
from playsound import playsound

# Mediapipe face mesh setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Eye landmark indexes (Mediapippipe reference)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Function to play alarm sound in background
def play_alarm():
    playsound("alarm.mp3")   # keep "alarm.mp3" in same folder

# Thresholds
EAR_THRESHOLD = 0.25
CLOSED_FRAMES = 20   # no. of continuous frames eyes must be closed

cap = cv2.VideoCapture(0)
counter = 0
alarm_on = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape
            landmarks = np.array([(lm.x * w, lm.y * h) for lm in face_landmarks.landmark])

            left_eye = landmarks[LEFT_EYE]
            right_eye = landmarks[RIGHT_EYE]

            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0

            if ear < EAR_THRESHOLD:
                counter += 1
                if counter >= CLOSED_FRAMES:
                    cv2.putText(frame, "DROWSY ALERT!", (100, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

                    if not alarm_on:  # play alarm once
                        alarm_on = True
                        threading.Thread(target=play_alarm, daemon=True).start()
            else:
                counter = 0
                alarm_on = False

            # Draw eyes landmarks
            for p in LEFT_EYE + RIGHT_EYE:
                cv2.circle(frame, (int(landmarks[p][0]), int(landmarks[p][1])), 2, (0, 255, 0), -1)

    cv2.imshow("AI Drowsiness Detector", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()