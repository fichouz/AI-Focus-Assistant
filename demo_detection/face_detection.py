import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

FACE_OUTLINE = [
    10, 338, 297, 332, 284, 251, 389,
    356, 454, 323, 361, 288, 397,
    365, 379, 378, 400, 377, 152,
    148, 176, 149, 150, 136, 172,
    58, 132, 93, 234, 127, 162,
    21, 54, 103, 67, 109
]

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:
            face_points = []

            for idx in FACE_OUTLINE:
                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)
                face_points.append((x, y))

            face_points = np.array(face_points, np.int32)
            cv2.polylines(frame, [face_points], True, (0, 255, 0), 1)
            left_eye_points = []

            for idx in LEFT_EYE:
                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)
                left_eye_points.append((x, y))

            left_eye_points = np.array(left_eye_points, np.int32)
            cv2.polylines(frame, [left_eye_points], True, (0, 255, 0), 1)
            right_eye_points = []

            for idx in RIGHT_EYE:
                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)
                right_eye_points.append((x, y))

            right_eye_points = np.array(right_eye_points, np.int32)
            cv2.polylines(frame, [right_eye_points], True, (0, 255, 0), 1)

    cv2.imshow("AI Focus Assistant", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()