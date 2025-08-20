import pybullet as p
import pybullet_data
from hand_model import (
    get_hand_keypoints,
    extract_finger_angles,
    extract_finger_angles_alternative,
    release,
    cap,
    hands,
    mp_hands,
    get_simple_hand_rotation,
    debug_finger_curl,
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
start_orientation = p.getQuaternionFromEuler([0, 0, 0])
robot = p.loadURDF("robot_arm.urdf", [0, 0, 0.1], start_orientation, useFixedBase=True)
p.resetDebugVisualizerCamera(
    cameraDistance=0.3,
    cameraYaw=45,
    cameraPitch=-20,
    cameraTargetPosition=[0, 0, 0.15],
)
p.setTimeStep(1.0 / 240.0)
p.setRealTimeSimulation(0)

joint_indices = [i for i in range(p.getNumJoints(robot))]
for i in range(p.getNumJoints(robot)):
    info = p.getJointInfo(robot, i)
    print(f"Joint index: {i}, name: {info[1].decode('utf-8')}")

slider_ids = []
for i in joint_indices:
    info = p.getJointInfo(robot, i)
    joint_name = info[1].decode("utf-8")
    slider = p.addUserDebugParameter(joint_name, -0.5, 1.8, 0.0)
    slider_ids.append(slider)

method_slider = p.addUserDebugParameter("Method (0=curl, 1=angle)", 0, 1, 0)
sensitivity_slider = p.addUserDebugParameter("Sensitivity", 0.5, 2.0, 1.0)
smoothing_slider = p.addUserDebugParameter("Smoothing", 0.01, 0.2, 0.05)

previous_angles = [0.0] * len(joint_indices)


def smooth_angles(new_angles, prev_angles, factor):
    """Apply exponential smoothing to reduce jitter"""
    return [
        factor * prev + (1 - factor) * new for new, prev in zip(new_angles, prev_angles)
    ]


def apply_sensitivity(angles, sensitivity):
    """Apply sensitivity scaling to angles"""
    return [angle * sensitivity for angle in angles]


try:
    frame_count = 0
    while True:
        frame_count += 1

        # Read control parameters
        use_angle_method = p.readUserDebugParameter(method_slider) > 0.5
        sensitivity = p.readUserDebugParameter(sensitivity_slider)
        smoothing_factor = p.readUserDebugParameter(smoothing_slider)

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

            # Choose method based on slider
            if use_angle_method:
                raw_angles = extract_finger_angles_alternative(keypoints)
                method_text = "Method: Angle-based"
            else:
                raw_angles = extract_finger_angles(keypoints)
                method_text = "Method: Curl-based"

            # Apply sensitivity
            raw_angles = apply_sensitivity(raw_angles, sensitivity)

            # Apply smoothing
            angles = smooth_angles(raw_angles, previous_angles, smoothing_factor)
            previous_angles = angles.copy()

            # Debug: Print curl values every 30 frames (about once per second)
            if frame_count % 30 == 0 and not use_angle_method:
                fingers = [
                    [1, 2, 3, 4],  # Thumb
                    [5, 6, 7, 8],  # Index
                    [9, 10, 11, 12],  # Middle
                    [13, 14, 15, 16],  # Ring
                    [17, 18, 19, 20],  # Pinky
                ]
                for i, finger_points in enumerate(fingers):
                    debug_finger_curl(keypoints, i, finger_points)

            # --- Visual feedback ---
            # Draw line between wrist and middle finger base
            pt1 = keypoints[0][:2] * np.array([frame.shape[1], frame.shape[0]])
            pt2 = keypoints[9][:2] * np.array([frame.shape[1], frame.shape[0]])
            pt1 = pt1.astype(int)
            pt2 = pt2.astype(int)
            cv2.line(frame, tuple(pt1), tuple(pt2), (0, 255, 0), 2)

            # Display method and sensitivity
            cv2.putText(
                frame,
                method_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
            )
            cv2.putText(
                frame,
                f"Sensitivity: {sensitivity:.1f}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 255),
                1,
            )
            cv2.putText(
                frame,
                f"Smoothing: {smoothing_factor:.2f}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 255),
                1,
            )

            # --- Palm orientation ---
            if keypoints.shape == (21, 3):
                try:
                    quat = get_simple_hand_rotation(keypoints)
                    current_pos, _ = p.getBasePositionAndOrientation(robot)
                    p.resetBasePositionAndOrientation(robot, current_pos, quat)
                except Exception:
                    pass

            # Display finger angles for debugging
            finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
            for i, finger in enumerate(finger_names):
                if i * 2 < len(angles) and i * 2 + 1 < len(angles):
                    angle1, angle2 = angles[i * 2], angles[i * 2 + 1]
                    # Color code: green for normal range, red for extreme values
                    color = (
                        (0, 255, 0)
                        if -0.5 < angle1 < 1.5 and -0.5 < angle2 < 1.5
                        else (0, 0, 255)
                    )
                    text = f"{finger}: {angle1:.2f}, {angle2:.2f}"
                    cv2.putText(
                        frame,
                        text,
                        (10, 100 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        color,
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

                # Enhanced motor control with better parameters
                p.setJointMotorControl2(
                    robot,
                    joint_id,
                    p.POSITION_CONTROL,
                    targetPosition=angle,
                    force=800,  # Increased force for faster response
                    maxVelocity=50.0,  # Increased max velocity
                    positionGain=2.0,  # Higher gain for more responsive tracking
                    velocityGain=0.3,  # Lower velocity gain to prevent oscillation
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
