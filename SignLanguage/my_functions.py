import mediapipe as mp
import cv2
import numpy as np


def draw_landmarks(image, results):
    """
    Draw the landmarks on the image.

    Args:
        image (cv2.UMat): The input image as a UMat object.
        results: The landmarks detected by Mediapipe.

    Returns:
        cv2.UMat: Image with landmarks drawn.
    """
    # Draw landmarks for left hand
    mp.solutions.drawing_utils.draw_landmarks(image, results.left_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)
    # Draw landmarks for right hand
    mp.solutions.drawing_utils.draw_landmarks(image, results.right_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)


def image_process(image, model):
    """
    Process the image and obtain sign landmarks.

    Args:
        image (numpy.ndarray): The input image.
           model: The Mediapipe holistic object.

    Returns:
        results: The processed results containing sign landmarks.
    """
    # Set the image to read-only mode
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)#color conversion from bgr to rgb
    image.flags.writeable =  False#image is not writeable
    results = model.process(image)#make prediction
    image.flags.writeable = True#image is set to writeable
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)#color conversion back from rgb to bgr
    return image,results

def extract_keypoints(results):
   
    lh = np.array([[res.x,res.y,res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x,res.y,res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh,rh])
