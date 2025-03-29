import cv2
import os
import threading
import tkinter as tk
from tkinter import Label, Button
from keras.models import load_model
import numpy as np
from pygame import mixer

# Initialize paths and directories
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Initialize alarm sound
mixer.init()
sound = mixer.Sound('alarm.wav')

# Debugging: Verify cascade files exist
print("Checking for cascade files...")
print("Face cascade exists:", os.path.exists('haar cascade files/haarcascade_frontalface_alt.xml'))
print("Left eye cascade exists:", os.path.exists('haar cascade files/haarcascade_lefteye_2splits.xml'))
print("Right eye cascade exists:", os.path.exists('haar cascade files/haarcascade_righteye_2splits.xml'))

# Load Haar cascades with error handling
try:
    face_cascade = cv2.CascadeClassifier('haar cascade files/haarcascade_frontalface_alt.xml')
    leye_cascade = cv2.CascadeClassifier('haar cascade files/haarcascade_lefteye_2splits.xml')
    reye_cascade = cv2.CascadeClassifier('haar cascade files/haarcascade_righteye_2splits.xml')
    
    # Verify cascades loaded properly
    if face_cascade.empty() or leye_cascade.empty() or reye_cascade.empty():
        raise ValueError("One or more cascade files failed to load")
except Exception as e:
    print(f"❌ Error loading cascade files: {e}")
    exit()

# Load model with error handling
try:
    model = load_model('models/cnnCat2.h5')
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# Global variables
cap = None
running = False
score = 0
drowsy_count = 0

def start_detection():
    global cap, running, score, drowsy_count
    running = True
    score = 0
    drowsy_count = 0
    
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Could not open webcam")
    except Exception as e:
        print(f"❌ Webcam error: {e}")
        return

    def detect_drowsiness():
        global running, score, drowsy_count
        thicc = 2
        
        while running:
            ret, frame = cap.read()
            if not ret:
                print("❌ Could not read frame from webcam")
                break

            height, width = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces with more sensitive parameters
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Draw rectangles around detected faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                
                # Detect eyes within the face region (improves accuracy)
                left_eye = leye_cascade.detectMultiScale(
                    roi_gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(20, 20)
                )
                
                right_eye = reye_cascade.detectMultiScale(
                    roi_gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(20, 20)
                )
                
                # Process left eye
                for (ex, ey, ew, eh) in left_eye:
                    l_eye = roi_color[ey:ey+eh, ex:ex+ew]
                    l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
                    l_eye = cv2.resize(l_eye, (24, 24))
                    l_eye = l_eye / 255.0
                    l_eye = l_eye.reshape(24, 24, -1)
                    l_eye = np.expand_dims(l_eye, axis=0)
                    lpred = np.argmax(model.predict(l_eye), axis=-1)
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                    break
                
                # Process right eye
                for (ex, ey, ew, eh) in right_eye:
                    r_eye = roi_color[ey:ey+eh, ex:ex+ew]
                    r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
                    r_eye = cv2.resize(r_eye, (24, 24))
                    r_eye = r_eye / 255.0
                    r_eye = r_eye.reshape(24, 24, -1)
                    r_eye = np.expand_dims(r_eye, axis=0)
                    rpred = np.argmax(model.predict(r_eye), axis=-1)
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 255), 2)
                    break

            # Update score based on eye status
            if 'lpred' in locals() and 'rpred' in locals():
                if lpred[0] == 0 and rpred[0] == 0:  # Both eyes closed
                    score += 1
                    cv2.putText(frame, "Closed", (10, height-20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    score = max(0, score - 1)
                    cv2.putText(frame, "Open", (10, height-20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Eyes not detected", (10, height-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # Display score
            cv2.putText(frame, f'Score: {score}', (100, height-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Trigger alarm if score exceeds threshold (but don't stop automatically)
            if score > 15:
                try:
                    sound.play()
                except:
                    print("⚠️ Could not play alarm sound")
                
                drowsy_count += 1
                print(f"Drowsiness detected {drowsy_count} times")
                
                if thicc < 16:
                    thicc += 2
                else:
                    thicc -= 2
                    if thicc < 2:
                        thicc = 2
                
                cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)

            cv2.imshow('Drowsiness Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                running = False
                break

        cap.release()
        cv2.destroyAllWindows()

    detection_thread = threading.Thread(target=detect_drowsiness)
    detection_thread.start()

def stop_detection():
    global running, cap
    running = False
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    print("Detection stopped by user")

# GUI Setup
root = tk.Tk()
root.title("Drowsiness Detection")
root.geometry("300x200")

title_label = Label(root, text="Drowsiness Detection", font=("Arial", 14, "bold"))
title_label.pack(pady=10)

start_button = Button(root, text="Start Detection", font=("Arial", 12), 
                    bg="green", fg="white", command=start_detection)
start_button.pack(pady=5)

stop_button = Button(root, text="Stop Detection", font=("Arial", 12), 
                   bg="red", fg="white", command=stop_detection)
stop_button.pack(pady=5)

exit_button = Button(root, text="Exit", font=("Arial", 12), command=root.quit)
exit_button.pack(pady=10)

root.mainloop()