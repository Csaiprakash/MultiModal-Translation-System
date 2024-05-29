
from flask import Flask, render_template, request, Response, send_file
import numpy as np
import os
import string
import mediapipe as mp
import cv2
import keyboard
from tensorflow.keras.models import load_model
import language_tool_python
import io

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    def draw_landmarks(image, results):
        # Draw landmarks on the image
        mp.solutions.drawing_utils.draw_landmarks(image, results.left_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)
        mp.solutions.drawing_utils.draw_landmarks(image, results.right_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)

    def image_process(image, model):
        # Process the image and obtain sign landmarks
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable =  False
        results = model.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, results

    def extract_keypoints(results):
        # Extract keypoints from the results
        lh = np.array([[res.x,res.y,res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x,res.y,res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([lh, rh])

    # Set the path to the data directory
    PATH = os.path.join('data_3')
    actions = np.array(os.listdir(PATH))

    # Load the trained model
    model = load_model('my_model3.keras')

    # Create an instance of the grammar correction tool
    tool = language_tool_python.LanguageToolPublicAPI('en-UK')

    # Initialize variables  
    sentence, keypoints, last_prediction, grammar, grammar_result = [], [], [], [], []

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot access camera.")
        exit()
    else:
        print("successful access")

    # Store the captured text for download
    captured_text = ''

    with mp.solutions.holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            image, results = image_process(frame, holistic)
            draw_landmarks(image, results)
            keypoints.append(extract_keypoints(results))

            if len(keypoints) == 10:
                keypoints = np.array(keypoints)
                prediction = model.predict(keypoints[np.newaxis, :, :])
                keypoints = []

                if np.amax(prediction) > 0.8:  # Check if maximum prediction accuracy is at least 80%
                    if last_prediction != actions[np.argmax(prediction)]:
                        sentence.append(actions[np.argmax(prediction)])
                        last_prediction = actions[np.argmax(prediction)]

            if len(sentence) > 7:
                sentence = sentence[-7:]

            if keyboard.is_pressed(' '):
                sentence, keypoints, last_prediction, grammar, grammar_result = [], [], [], [], []

            if sentence:
                sentence[0] = sentence[0].capitalize()

            if len(sentence) >= 2:
                if sentence[-1] in string.ascii_lowercase or sentence[-1] in string.ascii_uppercase:
                    if sentence[-2] in string.ascii_lowercase or sentence[-2] in string.ascii_uppercase or (sentence[-2] not in actions and sentence[-2] not in list(x.capitalize() for x in actions)):
                        sentence[-1] = sentence[-2] + sentence[-1]
                        sentence.pop(len(sentence) - 2)
                        sentence[-1] = sentence[-1].capitalize()

            if keyboard.is_pressed('enter'):
                text = ' '.join(sentence)
                grammar_result = tool.correct(text)

            if grammar_result:
                text = grammar_result
            else:
                text = ' '.join(sentence)

            captured_text = text  # Store the captured text for download

            # Display text on the image
            if grammar_result:
                text = grammar_result
            else:
                text = ' '.join(sentence)
            cv2.putText(image, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('Camera', image)
            cv2.waitKey(1)

            if cv2.getWindowProperty('Camera', cv2.WND_PROP_VISIBLE) < 1:
                break

        cap.release()
        cv2.destroyAllWindows()
        tool.close()

        # Prepare the captured text for download
        output = io.BytesIO()
        output.write(captured_text.encode())
        output.seek(0)

        return Response(output, mimetype="text/plain",
                        headers={"Content-Disposition": "attachment;filename=translated_text.txt"})

if __name__ == '__main__':
    app.run(debug=True)
