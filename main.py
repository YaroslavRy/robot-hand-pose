import pybullet as p
import pybullet_data
from hand_model import (
    get_hand_keypoints,
    extract_finger_angles,
    release,
    cap,
    hands,
    mp_hands,
    get_palm_orientation,
    rotation_matrix_to_quaternion,
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

start_orientation = p.getQuaternionFromEuler([math.pi * 2, 0, 0])
robot = p.loadURDF("robot_arm.urdf", [0, 0, 0.08], start_orientation, useFixedBase=True)

joint_indices = [i for i in range(p.getNumJoints(robot))]

print("joint_indices", joint_indices)

for i in range(p.getNumJoints(robot)):
    info = p.getJointInfo(robot, i)
    print(f"Joint index: {i}, name: {info[1].decode('utf-8')}")

# Add sliders for each joint
slider_ids = []
for i in joint_indices:
    info = p.getJointInfo(robot, i)
    joint_name = info[1].decode("utf-8")
    slider = p.addUserDebugParameter(joint_name, -0.1, 1.0, 0.0)
    slider_ids.append(slider)

# Mapping: robot joint index -> (landmark_a, landmark_b, landmark_c)
# These are the indices for angle_between_points(a, b, c)
KEYPOINT_TO_JOINT_MAP = {
    0: (1, 2, 3),  # thumb_joint1: Thumb CMC-MCP-IP
    1: (2, 3, 4),  # thumb_joint2: Thumb MCP-IP-tip
    2: (5, 6, 7),  # index_joint1: Index MCP-PIP-DIP
    3: (6, 7, 8),  # index_joint2: Index PIP-DIP-tip
    4: (9, 10, 11),  # middle_joint1: Middle MCP-PIP-DIP
    5: (10, 11, 12),  # middle_joint2: Middle PIP-DIP-tip
    6: (13, 14, 15),  # ring_joint1: Ring MCP-PIP-DIP
    7: (14, 15, 16),  # ring_joint2: Ring PIP-DIP-tip
    8: (17, 18, 19),  # pinky_joint1: Pinky MCP-PIP-DIP
    9: (18, 19, 20),  # pinky_joint2: Pinky PIP-DIP-tip
}

try:
    while True:
        # Read slider values
        slider_angles = [p.readUserDebugParameter(s) for s in slider_ids]

        ret, frame = cap.read()
        if not ret:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )
            keypoints = np.array(
                [[lm.x, lm.y, lm.z] for lm in results.multi_hand_landmarks[0].landmark]
            )
            angles = extract_finger_angles(keypoints)

            # --- Line between wrist (0) and middle finger base (9) ---
            pt1 = keypoints[0][:2] * np.array([frame.shape[1], frame.shape[0]])
            pt2 = keypoints[9][:2] * np.array([frame.shape[1], frame.shape[0]])
            pt1 = pt1.astype(int)
            pt2 = pt2.astype(int)
            cv2.line(frame, tuple(pt1), tuple(pt2), (0, 0, 255), 2)

            # --- Angle with respect to horizon (X axis) ---
            dx = pt2[0] - pt1[0]
            dy = pt2[1] - pt1[1]
            angle_rad = np.arctan2(dy, dx)
            angle_deg = np.degrees(angle_rad)
            text = f"Palm angle: {angle_deg:.1f} deg"
            cv2.putText(
                frame, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2
            )

            # --- Palm orientation ---
            if keypoints.shape == (21, 3):
                rot = get_palm_orientation(keypoints)
                # Rotate by -90° around X to make vertical palm = 0 roll
                correction = R.from_euler("x", -90, degrees=True).as_matrix()
                rot = correction @ rot
                quat = rotation_matrix_to_quaternion(rot)
                p.resetBasePositionAndOrientation(robot, [0, 0, 0], quat)

                # Convert rotation matrix to Euler angles (degrees)
                euler = R.from_matrix(rot).as_euler("xyz", degrees=True)
                text = f"Palm Euler: Pitch={euler[0]:.1f}, Yaw={euler[1]:.1f}, Roll={euler[2]:.1f}"
                cv2.putText(
                    frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
                )

                # Draw palm roll direction as a line
                # Palm center in image coordinates
                palm_center = keypoints[0][:2] * np.array(
                    [frame.shape[1], frame.shape[0]]
                )
                palm_center = palm_center.astype(int)

                # Use roll angle to draw direction
                roll_rad = np.deg2rad(euler[2])
                length = 50  # length of the line
                dx = int(length * np.cos(roll_rad))
                dy = int(length * np.sin(roll_rad))
                end_point = (palm_center[0] + dx, palm_center[1] + dy)

                cv2.line(frame, tuple(palm_center), end_point, (0, 0, 255), 2)
        else:
            angles = slider_angles

        for i, angle in zip(joint_indices, angles):
            p.setJointMotorControl2(robot, i, p.POSITION_CONTROL, targetPosition=angle)
        p.stepSimulation()

        cv2.imshow("Hand Pose", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        time.sleep(1.0 / 60)
except KeyboardInterrupt:
    pass
finally:
    release()
    p.disconnect()
