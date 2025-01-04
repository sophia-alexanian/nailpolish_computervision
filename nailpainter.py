print("Please ignore any arbitrary warnings. The program will greet you when it's ready. :)")
print(".")
print(".")
print(".")

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

# Function for getting colour from audio!
def get_color_from_audio():
    with sr.Microphone() as source:
        print("You can choose between the following colours: red, orange, yellow, green, blue, purple, pink, black or white :)")
        print("Calibrating for ambient noise to prepare for voice recognition... :)")
        recognizer.adjust_for_ambient_noise(source, duration=5)  # Adjusts to background noise
        print("Energy threshold set :)")
        print("Listening for a color name... :)")
        try:
            audio = recognizer.listen(source, timeout=15, phrase_time_limit=5)
            color_name = recognizer.recognize_google(audio).lower()
            print(f"You said: {color_name} :)")
            # Map common color names to RGB values (BGR)
            color_map = {
                "red": (0, 0, 255),
                "blue": (255, 0, 0),
                "green": (0, 255, 0),
                "yellow": (0, 255, 255),
                "pink": (203, 192, 255),
                "purple": (128, 0, 128),
                "orange": (0, 165, 255),
                "white": (255, 255, 255),
                "black": (0, 0, 0),
            }
            if color_name not in color_map:
                print("That colour is not available, defaulting to green. Press 'c' to try again. :)")
            return color_map.get(color_name, (0, 255, 0))  # Default to green if not recognized
        except sr.UnknownValueError:
            print("Sorry, I could not understand the color name, defaulting to green. Press 'c' to try again. :)")
        except sr.RequestError as e:
            print(f"Could not request results :(; {e}")
        except TimeoutError:
            print("Listening timed out. :(")
        return (0, 255, 0)  # Default color

# Start capturing video
capture = cv2.VideoCapture(0)

# Capture a frame to get the dimensions (height, width)
ret, frame = capture.read()
height, width, _ = frame.shape  # height and width of the captured frame

previousTime = 0
color = (0, 255, 0)  # Default color (green)

# Printing on terminal
print("Welcome to our nail polish try-on software! Explore and see what style nails suit you best! :)")
print("Press 'c' to change colour or 'q' to exit program. :)")

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

            # Draw ellipse instead of a circle
            axes = (18, 10)  
            angle = 0  # No rotation
            start_angle = 0
            end_angle = 360
            cv2.ellipse(image, (x, y), axes, angle, start_angle, end_angle, color, -1)

    if results.left_hand_landmarks:
        for idx in [4, 8, 12, 16, 20]:  # Fingertip landmarks
            x = int(results.left_hand_landmarks.landmark[idx].x * image.shape[1])
            y = int(results.left_hand_landmarks.landmark[idx].y * image.shape[0])

            # Draw ellipse instead of a circle
            axes = (18, 10)  
            angle = 0  # No rotation
            start_angle = 0
            end_angle = 360
            cv2.ellipse(image, (x, y), axes, angle, start_angle, end_angle, color, -1)


    # Display FPS
    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime
    cv2.putText(image, f"{int(fps)} FPS", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    # Display the image
    cv2.imshow("Fingernail Detection", image)

    # Change color based on audio input
    key = cv2.waitKey(1) & 0xFF  # Use a small delay for key input to work
    if key == ord('c'):  # Press 'c' to trigger audio input
        color = get_color_from_audio()
    elif key == ord('q'):  # Press 'q' to quit
        break

# Cleanup
capture.release()
cv2.destroyAllWindows()
