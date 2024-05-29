# %%

# Import necessary libraries
import numpy as np
import os
import string
import mediapipe as mp
import cv2
#from my_functions import *
import keyboard
from tensorflow.keras.models import load_model
import language_tool_python



def draw_landmarks(image, results):
    mp.solutions.drawing_utils.draw_landmarks(image, results.left_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)
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
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image.flags.writeable =  False
    results = model.process(image)#make prediction
    image.flags.writeable = True#image is set to writeable
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)#color conversion back from rgb to bgr
    return image,results

def extract_keypoints(results):
   
    lh = np.array([[res.x,res.y,res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x,res.y,res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh,rh])


# Set the path to the data directory
PATH = os.path.join('data_3')

# Create an array of action labels by listing the contents of the data directory
actions = np.array(os.listdir(PATH))

# actions = np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 
#     'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'])
    

# Load the trained model
model = load_model('my_model3.keras')

# Create an instance of the grammar correction tool
tool = language_tool_python.LanguageToolPublicAPI('en-UK')

# Initialize the lists
sentence, keypoints, last_prediction, grammar, grammar_result = [], [], [], [], []

# Access the camera and check if the camera is opened successfully
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot access camera.")
    exit()
else:
    print("successful access")

#cap = cv2.VideoCapture(0)
# Set mediapipe model 
with mp.solutions.holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()

        # Make detections
        image, results = image_process(frame, holistic)
        print(results)
        
        # Draw landmarks
        draw_landmarks(image, results)

        keypoints.append(extract_keypoints(results))


        # Check if 10 frames have been accumulated
        if len(keypoints) == 10:
            # Convert keypoints list to a numpy array
            keypoints = np.array(keypoints)
            # Make a prediction on the keypoints using the loaded model
            prediction = model.predict(keypoints[np.newaxis, :, :])
            # Clear the keypoints list for the next set of frames
            keypoints = []

            # Check if the maximum prediction value is above 0.9
            if np.amax(prediction) > 0.9:
                # Get the current predicted sign
                current_prediction = actions[np.argmax(prediction)]
                # Check if the current predicted sign is not empty and different from the previous prediction
                if current_prediction != "" and current_prediction != last_prediction:
                    # If the sentence is not empty, add a space before appending the current prediction
                    if sentence:
                        sentence.append(" ")
                    # Append the current prediction to the sentence list
                    sentence.append(current_prediction)
                    # Record the current prediction to use it on the next cycle
                    last_prediction = current_prediction



        # Limit the sentence length to 7 elements to make sure it fits on the screen
        if len(sentence) > 7:
            sentence = sentence[-7:]

        # Reset if the "Spacebar" is pressed
        if keyboard.is_pressed(' '):
            sentence, keypoints, last_prediction, grammar, grammar_result = [], [], [], [], []

        #Check if the list is not empty
        if sentence:
            # Capitalize the first word of the sentence
            sentence[0] = sentence[0].capitalize()

        # Check if the sentence has at least two elements
        if len(sentence) >= 2:
            # Check if the last element of the sentence belongs to the alphabet (lower or upper cases)
            if sentence[-1] in string.ascii_lowercase or sentence[-1] in string.ascii_uppercase:
                # Check if the second last element of sentence belongs to the alphabet or is a new word
                if sentence[-2] in string.ascii_lowercase or sentence[-2] in string.ascii_uppercase or (sentence[-2] not in actions and sentence[-2] not in list(x.capitalize() for x in actions)):
                    # Combine last two elements
                    sentence[-1] = sentence[-2] + sentence[-1]
                    sentence.pop(len(sentence) - 2)
                    sentence[-1] = sentence[-1].capitalize()

        # Perform grammar check if "Enter" is pressed
        if keyboard.is_pressed('enter'):
            # Record the words in the sentence list into a single string
            text = ' '.join(sentence)
            # Apply grammar correction tool and extract the corrected result
            grammar_result = tool.correct(text)

        if grammar_result:
            # Calculate the size of the text to be displayed and the X coordinate for centering the text on the image
            textsize = cv2.getTextSize(grammar_result, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_X_coord = (image.shape[1] - textsize[0]) // 2

            # Draw the sentence on the image
            cv2.putText(image, grammar_result, (text_X_coord, 470),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            # Calculate the size of the text to be displayed and the X coordinate for centering the text on the image
            textsize = cv2.getTextSize(' '.join(sentence), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_X_coord = (image.shape[1] - textsize[0]) // 2

            # Draw the sentence on the image
            cv2.putText(image, ' '.join(sentence), (text_X_coord, 470),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Show the image on the display
        cv2.imshow('Camera', image)
        for i in sentence:
            print(i,end = " ")
        #print(text)

        cv2.waitKey(1)

        # Check if the 'Camera' window was closed and break the loop
        if cv2.getWindowProperty('Camera',cv2.WND_PROP_VISIBLE) < 1:
            break

    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()

    # Shut off the server
    tool.close()

