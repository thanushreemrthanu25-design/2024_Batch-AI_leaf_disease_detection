from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2
import os
app = Flask(__name__)

# Disease Information

disease_info = {

    "Tomato_septora_leaf_spot": {
        "cause": "Fungal infection due to high humidity",
        "treatment": "Remove infected leaves and apply fungicide regularly",
        "prevention": "Avoid overwatering and ensure good air circulation"
    },

    "Tomato_Early_blight": {
        "cause": "Fungus (Alternaria solani)",
        "treatment": "Use fungicide sprays and remove affected leaves",
        "prevention": "Crop rotation and avoid wet leaves"
    },

    "Tomato_leaf_late_blight": {
        "cause": "Water mold infection",
        "treatment": "Apply copper-based fungicides",
        "prevention": "Avoid overhead watering"
    },

    "Tomato_Healthy": {
        "cause": "No disease",
        "treatment": "No treatment needed",
        "prevention": "Maintain proper watering and sunlight"
    },

    "Tomato_leaf_yellow_curl_virus": {
        "cause": "Virus transmitted by whiteflies",
        "treatment": "Remove infected plants and control whiteflies",
        "prevention": "Use resistant varieties and insect control"
    },

    "Tomato_mold_leaf": {
        "cause": "Fungal infection in humid conditions",
        "treatment": "Apply fungicide and remove affected leaves",
        "prevention": "Improve air circulation and reduce humidity"
    }

}

# Load AI Model

model = tf.keras.models.load_model("leaf_model.h5")

# Get Class Names

class_names = sorted(os.listdir("dataset"))

print("CLASS NAMES:", class_names)

# Home Route

@app.route('/')

def home():

    return render_template("index.html")

# Prediction Route

@app.route('/predict', methods=['POST'])

def predict():

    file = request.files['file']

    filepath = "static/temp.jpg"

    file.save(filepath)

    # Read Image

    img = cv2.imread(filepath)

    img = cv2.resize(img, (224, 224))

    original_img = img.copy()

    img = img / 255.0

    img = np.expand_dims(img, axis=0)

    # Model Prediction

    prediction = model.predict(img)

    probabilities = prediction[0] * 100

    prediction_data = []

    for i in range(len(class_names)):

        prediction_data.append({

            "label": class_names[i].replace("_", " ").title(),

            "value": round(float(probabilities[i]), 2)

        })

    prediction_data = sorted(

        prediction_data,

        key=lambda x: x['value'],

        reverse=True

    )

    # Main Prediction

    class_index = np.argmax(prediction)

    confidence = round(float(np.max(prediction)) * 100, 2)

    result = class_names[class_index]

    result_key = result.strip().lower()

    # Disease Information

    disease_info_lower = {

        k.lower(): v for k, v in disease_info.items()

    }

    info = disease_info_lower.get(result_key, {

        "cause": "Not available",

        "treatment": "Not available",

        "prevention": "Not available"

    })

    # Return Results

    return render_template(

        "index.html",

        prediction=result.replace("_", " ").title(),

        confidence=confidence,

        image_path=filepath,

        cause=info["cause"],

        treatment=info["treatment"],

        prevention=info["prevention"],

        prediction_data=prediction_data

    )

# Run App

if __name__ == "__main__":

    app.run(debug=True)