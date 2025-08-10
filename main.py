import pybullet as p
import pybullet_data
from hand_model import get_hand_keypoints, extract_finger_angles, release, cap, hands, mp_hands
import cv2
import time
import mediapipe as mp
import numpy as np


p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf")
p.setGravity(0, 0, -9.8)


robot = p.loadURDF("robot_arm.urdf", [0, 0, 0], useFixedBase=True)


joint_indices = [i for i in range(p.getNumJoints(robot))]


try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks on the frame
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )
            # Use your existing functions to get angles
            keypoints = np.array([[lm.x, lm.y, lm.z] for lm in results.multi_hand_landmarks[0].landmark])
            angles = extract_finger_angles(keypoints)
            for i, angle in zip(joint_indices, angles):
                p.setJointMotorControl2(robot, i, p.POSITION_CONTROL, targetPosition=angle)
        p.stepSimulation()

        cv2.imshow("Hand Pose", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        time.sleep(1./60)
except KeyboardInterrupt:
    pass
finally:
    release()
    p.disconnect()
