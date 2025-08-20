import pybullet as p
import pybullet_data
from hand_model import (
    get_hand_keypoints,
    extract_finger_angles,
    release,
    cap,
    hands,
    mp_hands,
    get_simple_hand_rotation,
)
import cv2
import time
import mediapipe as mp
import numpy as np
from scipy.spatial.transform import Rotation as R
import math

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf")
p.setGravity(0, 0, -9.8)

# Better initial orientation - hand pointing up
start_orientation = p.getQuaternionFromEuler([0, 0, 0])
robot = p.loadURDF("robot_arm.urdf", [0, 0, 0.1], start_orientation, useFixedBase=True)

# Set up camera to be close to the hand
p.resetDebugVisualizerCamera(
    cameraDistance=0.3,  # Much closer - was default ~1.0
    cameraYaw=45,  # Angled view
    cameraPitch=-20,  # Looking slightly down
    cameraTargetPosition=[0, 0, 0.15],  # Focus on hand area
)

# Speed up physics for faster response
p.setTimeStep(1.0 / 240.0)  # Faster physics timestep
p.setRealTimeSimulation(0)  # Disable real-time for maximum speed

joint_indices = [i for i in range(p.getNumJoints(robot))]
print("joint_indices", joint_indices)

for i in range(p.getNumJoints(robot)):
    info = p.getJointInfo(robot, i)
    print(f"Joint index: {i}, name: {info[1].decode('utf-8')}")

# Add sliders for each joint with better default ranges
slider_ids = []
for i in joint_indices:
    info = p.getJointInfo(robot, i)
    joint_name = info[1].decode("utf-8")
    # Set slider range to match our scaled angles
    slider = p.addUserDebugParameter(joint_name, -0.5, 1.5, 0.0)
    slider_ids.append(slider)

# Store previous angles for smoothing
previous_angles = [0.0] * len(joint_indices)
smoothing_factor = 0.05  # Minimal smoothing for maximum speed


def smooth_angles(new_angles, prev_angles, factor):
    """Apply exponential smoothing to reduce jitter"""
    return [
        factor * prev + (1 - factor) * new for new, prev in zip(new_angles, prev_angles)
    ]


try:
    while True:
        # Read slider values as fallback
        slider_angles = [p.readUserDebugParameter(s) for s in slider_ids]

        ret, frame = cap.read()
        if not ret:
            continue

        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            # Draw hand landmarks
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

            # Get keypoints and calculate angles
            keypoints = np.array(
                [[lm.x, lm.y, lm.z] for lm in results.multi_hand_landmarks[0].landmark]
            )

            # Extract finger angles (now fixed for correct direction)
            raw_angles = extract_finger_angles(keypoints)

            # Apply minimal smoothing for faster response
            angles = smooth_angles(raw_angles, previous_angles, smoothing_factor)
            previous_angles = angles.copy()

            # --- Visual feedback ---
            # Draw line between wrist and middle finger base
            pt1 = keypoints[0][:2] * np.array([frame.shape[1], frame.shape[0]])
            pt2 = keypoints[9][:2] * np.array([frame.shape[1], frame.shape[0]])
            pt1 = pt1.astype(int)
            pt2 = pt2.astype(int)
            cv2.line(frame, tuple(pt1), tuple(pt2), (0, 255, 0), 2)

            # Calculate palm angle for display
            dx = pt2[0] - pt1[0]
            dy = pt2[1] - pt1[1]
            angle_rad = np.arctan2(dy, dx)
            angle_deg = np.degrees(angle_rad)
            text = f"Palm angle: {angle_deg:.1f} deg"
            cv2.putText(
                frame, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1
            )

            # --- Palm orientation ---
            if keypoints.shape == (21, 3):
                try:
                    quat = get_simple_hand_rotation(keypoints)
                    current_pos, _ = p.getBasePositionAndOrientation(robot)
                    p.resetBasePositionAndOrientation(robot, current_pos, quat)
                except Exception as e:
                    pass

            # Display finger angles for debugging
            finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
            for i, finger in enumerate(finger_names):
                if i * 2 < len(angles) and i * 2 + 1 < len(angles):
                    text = f"{finger}: {angles[i*2]:.2f}, {angles[i*2+1]:.2f}"
                    cv2.putText(
                        frame,
                        text,
                        (10, 100 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 255, 0),
                        1,
                    )

        else:
            # When no hand detected - return to neutral/open position
            neutral_angles = [-0.1, -0.1] * 5  # Open position for all fingers
            angles = smooth_angles(neutral_angles, previous_angles, 0.05)
            previous_angles = angles.copy()

        # Apply angles to robot joints
        for i in range(len(joint_indices)):
            if i < len(angles):
                angle = angles[i]
                joint_id = joint_indices[i]

                p.setJointMotorControl2(
                    robot,
                    joint_id,
                    p.POSITION_CONTROL,
                    targetPosition=angle,
                    force=600,  # Moderate force - enough power but not excessive
                    maxVelocity=35.0,  # Fast enough for real-time tracking
                    positionGain=1.5,  # Balanced - responsive but not jittery
                    velocityGain=0.4,  # Good damping to prevent oscillation
                )

        p.stepSimulation()

        cv2.imshow("Hand Pose", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        time.sleep(1.0 / 240)  # Even higher frame rate to match physics

except KeyboardInterrupt:
    pass
finally:
    release()
    p.disconnect()
