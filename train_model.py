import os
import cv2
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Set the data path
data_dir = "data"
categories = ["real", "fake"]
img_size = 100  # Size to resize images (100x100)

def load_data():
    data = []
    labels = []

    for category in categories:
        folder_path = os.path.join(data_dir, category)
        label = categories.index(category)

        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            try:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (img_size, img_size))
                data.append(img.flatten())
                labels.append(label)
            except Exception as e:
                print(f"[WARNING] Skipping {img_path}: {e}")

    return np.array(data), np.array(labels)

# Load and split the dataset
print("[INFO] Loading data...")
X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
print("[INFO] Training model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate the model
print("[INFO] Evaluating model...")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the model using pickle
model_filename = "face_classifier.pkl"
with open(model_filename, "wb") as f:
    pickle.dump(model, f)

print(f"[INFO] Model saved as {model_filename}")
