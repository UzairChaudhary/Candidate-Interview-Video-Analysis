import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
from collections import Counter
import matplotlib.pyplot as plt

# Load model and cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = model_from_json(open('fer.json', 'r').read())
model.load_weights('fer.h5')

# Define emotions
emotions = ('Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral')

video_path = 'C:/Users/Muhammad Uzair/Downloads/Video/demoInterviewVideo.mp4'

# Setup main window
root = tk.Tk()
root.title("Live Emotion Detection")

# Set up GUI elements
frame_label = ttk.Label(root)
frame_label.grid(row=0, column=0, columnspan=4)

emotion_label = ttk.Label(root, text="Emotions will appear here", font=('Helvetica', 14))
emotion_label.grid(row=1, column=0, columnspan=4)

# Function to update GUI with new frame
def update_frame(frame, emotion_text):
    cv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(cv_image)
    imgtk = ImageTk.PhotoImage(image=pil_image)
    frame_label.imgtk = imgtk
    frame_label.configure(image=imgtk)
    emotion_label.config(text=emotion_text)

# Function to handle video processing
def video_loop():
    emotion_list = []
    cap = cv2.VideoCapture('Test/demoInterviewVideo.mp4')
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.32, 5)
            emotion_text = "Detected Emotions: "
            for (x, y, w, h) in faces:
                face_img = gray[y:y+h, x:x+w]
                face_img = cv2.resize(face_img, (48, 48))
                img_pixels = image.img_to_array(face_img)
                img_pixels = np.expand_dims(img_pixels, axis=0)
                img_pixels /= 255
                predictions = model.predict(img_pixels)
                max_index = np.argmax(predictions[0])
                emotion = emotions[max_index]
                emotion_list.append(emotion)
                emotion_text += f"{emotion} "
            update_frame(frame, emotion_text)
            root.update()
        else:
            break
    cap.release()

# Run the video processing in a separate thread
import threading
thread = threading.Thread(target=video_loop)
thread.start()

root.mainloop()
