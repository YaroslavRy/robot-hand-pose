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
    """Calculate angle at point b between points a and c"""
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)  # Prevent numerical errors
    return np.arccos(cosine_angle)


def extract_finger_angles(keypoints):
    """
    Extract finger bend angles - with proper neutral position handling
    """
    angles = []
    
    # Define finger joints
    fingers = [
        [1, 2, 3, 4],    # Thumb
        [5, 6, 7, 8],    # Index  
        [9, 10, 11, 12], # Middle
        [13, 14, 15, 16],# Ring
        [17, 18, 19, 20] # Pinky
    ]
    
    for finger_idx, finger_points in enumerate(fingers):
        # First joint: base to middle
        p1, p2, p3 = finger_points[0], finger_points[1], finger_points[2]
        angle1 = angle_between_points(keypoints[p1], keypoints[p2], keypoints[p3])
        bend1 = np.pi - angle1  # Convert to bend angle
        
        # Second joint: middle to tip  
        p1, p2, p3 = finger_points[1], finger_points[2], finger_points[3]
        angle2 = angle_between_points(keypoints[p1], keypoints[p2], keypoints[p3])
        bend2 = np.pi - angle2  # Convert to bend angle
        
        # FIXED: Better scaling with proper neutral position
        if finger_idx == 0:  # Thumb
            # Map from bend angle to robot joint angle
            # When bend=0 (straight), robot should be at -0.2 (slightly back)
            # When bend=high (closed), robot should be at 1.2 (forward)
            scaled1 = np.clip((bend1 * 2.5) - 0.2, -0.3, 1.2)  
            scaled2 = np.clip((bend2 * 2.5) - 0.2, -0.3, 1.2)
        else:  # Other fingers
            # When bend=0 (straight), robot should be at -0.1 (back/straight)
            # When bend=high (closed), robot should be at 1.5 (forward/bent)
            scaled1 = np.clip((bend1 * 3.0) - 0.1, -0.2, 1.5)
            scaled2 = np.clip((bend2 * 3.0) - 0.1, -0.2, 1.5)
        
        angles.extend([scaled1, scaled2])
        
    return angles


def get_palm_orientation(keypoints):
    """
    Calculate palm orientation - simplified approach for better stability
    """
    wrist = keypoints[0]
    middle_mcp = keypoints[9]
    index_mcp = keypoints[5]
    pinky_mcp = keypoints[17]
    
    # Simple approach: just use wrist to middle finger for main direction
    forward = middle_mcp - wrist
    forward = forward / np.linalg.norm(forward)
    
    # Side direction: wrist to index-pinky midpoint
    side = ((index_mcp + pinky_mcp) / 2) - wrist
    side = side / np.linalg.norm(side)
    
    # Up direction: cross product
    up = np.cross(forward, side)
    up = up / np.linalg.norm(up)
    
    return forward, side, up


def get_simple_hand_rotation(keypoints):
    """
    Fixed hand rotation - direct mapping with proper coordinate system
    """
    wrist = keypoints[0]
    index_mcp = keypoints[5]
    pinky_mcp = keypoints[17]
    middle_mcp = keypoints[9]
    
    # Method 1: Use knuckle line (more stable)
    knuckle_vector = index_mcp - pinky_mcp
    knuckle_angle = np.arctan2(knuckle_vector[1], knuckle_vector[0])
    
    # Method 2: Use palm direction (alternative)
    palm_vector = middle_mcp - wrist
    palm_angle = np.arctan2(palm_vector[1], palm_vector[0])
    
    # Use knuckle line for rotation (more reliable)
    angle = knuckle_angle
    
    # CRITICAL FIX: Coordinate system correction
    # MediaPipe: Y increases downward, X increases right
    # PyBullet: Standard XYZ where Z is up
    # We need to adjust for this coordinate difference
    
    # Flip Y component to match standard coordinates
    angle = -angle
    
    # Add 90 degree offset so that horizontal knuckles = 0 rotation
    angle = angle + np.pi/2
    
    # Normalize angle to [-π, π]
    angle = np.arctan2(np.sin(angle), np.cos(angle))
    
    euler = [0, 0, angle]
    return p.getQuaternionFromEuler(euler)


def release():
    cap.release()
    cv2.destroyAllWindows()