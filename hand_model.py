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


def calculate_finger_curl(keypoints, finger_points):
    """
    Calculate finger curl using distance-based method
    More reliable than angle-based for MediaPipe landmarks
    """
    # Get the four points for this finger
    mcp, pip, dip, tip = [keypoints[i] for i in finger_points]
    
    # Calculate the "straight" distance (MCP to tip)
    straight_distance = np.linalg.norm(tip - mcp)
    
    # Calculate the "bent" distance (sum of segments)
    bent_distance = (np.linalg.norm(pip - mcp) + 
                    np.linalg.norm(dip - pip) + 
                    np.linalg.norm(tip - dip))
    
    # Curl ratio: 0 = straight, 1 = fully curled
    # When straight: bent_distance ≈ straight_distance, ratio ≈ 0
    # When curled: bent_distance > straight_distance, ratio approaches 1
    curl_ratio = 1 - (straight_distance / bent_distance) if bent_distance > 0 else 0
    curl_ratio = np.clip(curl_ratio, 0, 1)
    
    return curl_ratio


def extract_finger_angles(keypoints):
    """
    Extract finger bend angles using improved curl detection
    """
    angles = []
    
    # Define finger joints (MediaPipe landmark indices)
    fingers = [
        [1, 2, 3, 4],    # Thumb
        [5, 6, 7, 8],    # Index  
        [9, 10, 11, 12], # Middle
        [13, 14, 15, 16],# Ring
        [17, 18, 19, 20] # Pinky
    ]
    
    for finger_idx, finger_points in enumerate(fingers):
        # Calculate curl ratio for the whole finger
        curl_ratio = calculate_finger_curl(keypoints, finger_points)
        
        # For robot control, we need two joint angles per finger
        # Distribute the curl across both joints
        
        if finger_idx == 0:  # Thumb - different scaling
            # Thumb has different range of motion
            joint1_angle = np.clip(curl_ratio * 1.8 - 0.2, -0.3, 1.5)
            joint2_angle = np.clip(curl_ratio * 1.6 - 0.1, -0.3, 1.3)
        else:  # Other fingers
            # Map curl ratio to joint angles
            # Open hand: curl_ratio ≈ 0 → angles ≈ -0.1 (slightly back)
            # Closed fist: curl_ratio ≈ 1 → angles ≈ 1.4 (fully forward)
            joint1_angle = np.clip(curl_ratio * 1.6 - 0.1, -0.2, 1.5)
            joint2_angle = np.clip(curl_ratio * 1.4 - 0.05, -0.2, 1.3)
        
        angles.extend([joint1_angle, joint2_angle])
        
    return angles


def angle_between_points(a, b, c):
    """Calculate angle at point b between points a and c"""
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    return np.arccos(cosine_angle)


def extract_finger_angles_alternative(keypoints):
    """
    Alternative method using improved angle calculation
    Use this if curl method doesn't work well for you
    """
    angles = []
    
    fingers = [
        [1, 2, 3, 4],    # Thumb
        [5, 6, 7, 8],    # Index  
        [9, 10, 11, 12], # Middle
        [13, 14, 15, 16],# Ring
        [17, 18, 19, 20] # Pinky
    ]
    
    for finger_idx, finger_points in enumerate(fingers):
        # First joint angle (MCP-PIP-DIP)
        angle1 = angle_between_points(
            keypoints[finger_points[0]], 
            keypoints[finger_points[1]], 
            keypoints[finger_points[2]]
        )
        
        # Second joint angle (PIP-DIP-TIP)
        angle2 = angle_between_points(
            keypoints[finger_points[1]], 
            keypoints[finger_points[2]], 
            keypoints[finger_points[3]]
        )
        
        # Convert to bend angles and scale properly
        bend1 = np.pi - angle1
        bend2 = np.pi - angle2
        
        # Improved scaling with better range mapping
        if finger_idx == 0:  # Thumb
            scaled1 = np.clip((bend1 * 1.8) - 0.3, -0.4, 1.6)  
            scaled2 = np.clip((bend2 * 1.6) - 0.2, -0.3, 1.4)
        else:  # Other fingers  
            scaled1 = np.clip((bend1 * 2.2) - 0.2, -0.3, 1.7)
            scaled2 = np.clip((bend2 * 2.0) - 0.1, -0.2, 1.5)
        
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


def debug_finger_curl(keypoints, finger_idx, finger_points):
    """
    Debug function to print curl values for a specific finger
    """
    curl = calculate_finger_curl(keypoints, finger_points)
    finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
    print(f"{finger_names[finger_idx]}: curl = {curl:.3f}")
    return curl


def release():
    cap.release()
    cv2.destroyAllWindows()