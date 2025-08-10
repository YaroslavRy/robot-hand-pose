import numpy as np
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7
)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


def get_hand_keypoints():
    ret, frame = cap.read()
    if not ret:
        return None

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0]
        keypoints = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
        return keypoints
    else:
        return None


def angle_between_points(a, b, c):
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.arccos(np.clip(cosine_angle, -1.0, 1.0))


def extract_finger_angles(keypoints):
    """
    keypoints: np.array shape (21, 3)
    Returns 5 angles for 5 fingers
    """
    indices = [
        (1, 2, 3),  # thumb base
        (2, 3, 4),  # thumb tip
        (5, 6, 7),  # index base
        (6, 7, 8),  # index tip
        (9, 10, 11),  # middle base
        (10, 11, 12),  # middle tip
        (13, 14, 15),  # ring base
        (14, 15, 16),  # ring tip
        (17, 18, 19),  # pinky base
        (18, 19, 20),  # pinky tip
    ]
    angles = []
    for a, b, c in indices:
        angle = angle_between_points(keypoints[a], keypoints[b], keypoints[c])
        # Optionally scale/offset to fit joint limits
        angles.append(np.clip(angle - np.pi / 2, -1.0, 1.0))  # Center at 0
    return angles


def release():
    cap.release()
    cv2.destroyAllWindows()
