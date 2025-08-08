import joblib
import cv2
import numpy as np

# Load model
model = joblib.load("face_classifier.pkl")

def predict_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        return "❌ Error: Image not found or not readable."

    # Resize image to 100x100 (must match training)
    image = cv2.resize(image, (100, 100))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Flatten image to 1D (100*100 = 10,000 pixels)
    features = gray.flatten()

    prediction = model.predict([features])[0]
    return "✅ Real Face" if prediction == 1 else "⚠️ Fake Face"

if __name__ == "__main__":
    img_path = input("Enter full image path: ").strip()
    result = predict_image(img_path)
    print(result)
