import numpy as np
import cv2
import mediapipe as mp
import pybullet as p

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


def get_palm_orientation(keypoints):
    # keypoints: np.array shape (21, 3)
    wrist = keypoints[0]
    index_base = keypoints[5]
    pinky_base = keypoints[17]

    # Palm x-axis: from wrist to middle between index and pinky base
    palm_x = ((index_base + pinky_base) / 2) - wrist
    palm_x /= np.linalg.norm(palm_x)

    # Palm y-axis: from wrist to index base
    palm_y = index_base - wrist
    palm_y /= np.linalg.norm(palm_y)

    # Palm z-axis: cross product (normal to palm)
    palm_z = np.cross(palm_x, palm_y)
    palm_z /= np.linalg.norm(palm_z)

    # Re-orthogonalize axes
    palm_y = np.cross(palm_z, palm_x)
    palm_y /= np.linalg.norm(palm_y)

    # Rotation matrix
    rot = np.stack([palm_x, palm_y, palm_z], axis=1)
    return rot


def rotation_matrix_to_quaternion(rot):
    # PyBullet expects [x, y, z, w]
    return p.getQuaternionFromEuler(list(cv2.Rodrigues(rot)[0].flatten()))


def release():
    cap.release()
    cv2.destroyAllWindows()
