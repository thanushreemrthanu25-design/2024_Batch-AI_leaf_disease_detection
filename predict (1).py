import tensorflow as tf
import numpy as np
import cv2
import os   # 👈 add this

# Load model
model = tf.keras.models.load_model("leaf_model.h5")

# 👇 ADD THIS HERE
class_names = sorted([
    folder for folder in os.listdir("dataset")
    if os.path.isdir(os.path.join("dataset", folder))
])
print("Classes:", class_names)

# Load image
img = cv2.imread("test.jpg")
img = cv2.resize(img, (224, 224))
img = img / 255.0
img = np.expand_dims(img, axis=0)

# Predict
prediction = model.predict(img)
class_index = np.argmax(prediction)
confidence = np.max(prediction)

print("Prediction:", class_names[class_index])
print("Confidence:", confidence)