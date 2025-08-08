import tkinter as tk
from tkinter import filedialog, Label
import cv2
import pickle
import numpy as np
from PIL import Image, ImageTk

# Load the trained model
with open("face_classifier.pkl", "rb") as f:
    model = pickle.load(f)

def extract_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (100, 100))  # Must match what was used during training
    return image.flatten()

def predict_image(image_path):
    features = extract_features(image_path)
    prediction = model.predict([features])[0]
    return "REAL" if prediction == 1 else "FAKE"

def upload_and_predict():
    file_path = filedialog.askopenfilename()
    if file_path:
        result = predict_image(file_path)
        img = Image.open(file_path).resize((150, 150))
        img_tk = ImageTk.PhotoImage(img)

        image_label.config(image=img_tk)
        image_label.image = img_tk
        result_label.config(text=f"Prediction: {result}")

# GUI Setup
root = tk.Tk()
root.title("Face Classifier")
root.geometry("300x400")

btn = tk.Button(root, text="Upload Face", command=upload_and_predict)
btn.pack(pady=10)

image_label = Label(root)
image_label.pack()

result_label = Label(root, text="", font=("Arial", 14))
result_label.pack(pady=10)

root.mainloop()
