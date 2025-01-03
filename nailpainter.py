import mediapipe as mp
import cv2
import numpy as np
import time
import speech_recognition as sr

# Initialize Mediapipe Holistic
mp_holistic = mp.solutions.holistic
holistic_model = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Speech recognition setup
recognizer = sr.Recognizer()

# Function to get color from user input
def get_color_from_audio():
    with sr.Microphone() as source:
        print("Listening for a color name...")
        try:
            audio = recognizer.listen(source, timeout=5)
            color_name = recognizer.recognize_google(audio).lower()
            print(f"You said: {color_name}")
            # Map common color names to RGB values
            color_map = {
                "red": (0, 0, 255),
                "blue": (255, 0, 0),
                "green": (0, 255, 0),
                "yellow": (0, 255, 255),
                "purple": (128, 0, 128),
                "pink": (255, 105, 180),
                "white": (255, 255, 255),
                "black": (0, 0, 0),
                "orange": (255, 165, 0)
            }
            return color_map.get(color_name, (0, 255, 0))  # Default to green if not recognized
        except sr.UnknownValueError:
            print("Sorry, I could not understand the color name.")
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
        except TimeoutError:
            print("Listening timed out.")
        return (0, 255, 0)  # Default color

# Start capturing video
capture = cv2.VideoCapture(0)

previousTime = 0
color = (0, 255, 0)  # Default color (green)

while capture.isOpened():
    ret, frame = capture.read()
    if not ret:
        break

    frame = cv2.resize(frame, (800, 600))
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = holistic_model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Extract hand landmarks and paint nails
    if results.right_hand_landmarks:
        for idx in [4, 8, 12, 16, 20]:  # Fingertip landmarks
            x = int(results.right_hand_landmarks.landmark[idx].x * image.shape[1])
            y = int(results.right_hand_landmarks.landmark[idx].y * image.shape[0])
            cv2.circle(image, (x, y), 15, color, -1)

    if results.left_hand_landmarks:
        for idx in [4, 8, 12, 16, 20]:  # Fingertip landmarks
            x = int(results.left_hand_landmarks.landmark[idx].x * image.shape[1])
            y = int(results.left_hand_landmarks.landmark[idx].y * image.shape[0])
            cv2.circle(image, (x, y), 15, color, -1)

    # Display FPS
    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime
    cv2.putText(image, f"{int(fps)} FPS", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    # Display the image
    cv2.imshow("Fingernail Detection", image)

    # Change color based on audio input
    if cv2.waitKey(5) & 0xFF == ord('c'):  # Press 'c' to trigger audio input
        color = get_color_from_audio()

    if cv2.waitKey(5) & 0xFF == ord('q'):  # Press 'q' to quit
        break

# Cleanup
capture.release()
cv2.destroyAllWindows()
