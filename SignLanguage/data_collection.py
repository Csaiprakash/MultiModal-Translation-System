# Import necessary libraries
import os
import numpy as np
import cv2
import mediapipe as mp
from itertools import product
import keyboard

# Import user-defined functions
from my_functions import image_process, draw_landmarks, extract_keypoints

# Define the actions (signs) that will be recorded and stored in the dataset
actions = np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z','bye','hello','i love you'
    ])

# Define the number of sequences and frames to be recorded for each action
sequences = 30
frames = 10

# Set the path where the dataset will be stored
PATH = os.path.join('data_3')

# Create directories for each action, sequence, and frame in the dataset
for action, sequence in product(actions, range(sequences)):
    try:
        os.makedirs(os.path.join(PATH, action, str(sequence)))
    except:
        pass

# Access the camera and check if the camera is opened successfully
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot access camera.")
    exit()

# Create a MediaPipe Holistic object for hand tracking and landmark extraction
cap = cv2.VideoCapture(0)
#set mediapipe model
with mp.solutions.holistic.Holistic(min_detection_confidence = 0.5,min_tracking_confidence = 0.5) as holistic:
    for action in actions:
        for sequence in range(sequences):
            for frame_num in range(frames):
                
                #read feed
                ret,frame = cap.read()
                #make detections
                image,results = image_process(frame,holistic)
                #print(results) uncomment this if u would like to see the results
                #model name is holistic as in the second line of this cell, 
                #we have created a model as holistic
                #frame is the last snapshot kind of thing captured by cam 
                #if you want it in real time then change frame in cv2.imshow('opencvfeed',frame) to image
                draw_landmarks(image,results)
                
                
                
                #apply collection break logic
                if frame_num == 0:
                    cv2.putText(image,'STARTING COLLECTION',(120,200),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),4,cv2.LINE_AA)
                    cv2.putText(image,'collecting frames for {} video number {}'.format(action,sequence),(15,12),
                                cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1,cv2.LINE_AA)
                    cv2.imshow('opencvfeed',image)
                    cv2.waitKey(2000)
                    
                else:
                    cv2.putText(image,'collecting frames for {} video number {}'.format(action,sequence),(15,12),
                                cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1,cv2.LINE_AA)
                cv2.imshow('opencvfeed',image)
                #new export keypoints
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(PATH,action,str(sequence),str(frame_num))
                np.save(npy_path,keypoints)
        

                    
                if cv2.waitKey(10) == ord('q'):
                    break
    
    cap.release()
    cv2.destroyAllWindows()
