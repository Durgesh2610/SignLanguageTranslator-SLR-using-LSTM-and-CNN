# my_functions.py
import mediapipe as mp
import cv2
import numpy as np

def draw_landmarks(image, results):
    output_image = image.copy()
    if results.left_hand_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            output_image, 
            results.left_hand_landmarks, 
            mp.solutions.holistic.HAND_CONNECTIONS
        )
    if results.right_hand_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            output_image, 
            results.right_hand_landmarks, 
            mp.solutions.holistic.HAND_CONNECTIONS
        )
    return output_image

def image_process(image, model):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model.process(image_rgb)
    return results

def keypoint_extraction(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    keypoints = np.concatenate([lh, rh])
    return keypoints
