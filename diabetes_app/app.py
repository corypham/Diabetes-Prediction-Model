from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# Reduce TensorFlow logging
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.get_logger().setLevel("ERROR")

app = Flask(__name__)

# Define feature names exactly as in training data
FEATURE_NAMES = [
    "age",
    "hypertension",
    "heart_disease",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level",
    "gender_female",
    "smoking_history_mapped",
]

try:
    print("Loading model and scaler...")
    model = tf.keras.models.load_model("model/diabetes_model.h5")
    scaler = joblib.load("model/scaler.pkl")

    # Warm-up prediction
    warmup_input = np.zeros((1, len(FEATURE_NAMES)))  # Create dummy input
    warmup_scaled = scaler.transform(warmup_input)
    _ = model.predict(warmup_scaled, verbose=0)  # Warm-up prediction
    print("Model warm-up completed")

except Exception as e:
    print("Error loading model or scaler:", str(e))
    raise e


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get raw values using exact same names as training
        raw_values = [
            float(request.form["age"]),
            int(request.form["hypertension"]),
            int(request.form["heart_disease"]),
            float(request.form["bmi"]),
            float(request.form["hba1c"]),  # HbA1c_level
            float(request.form["blood_glucose"]),  # blood_glucose_level
            int(request.form["gender_female"]),
            int(request.form["smoking_history"]),  # smoking_history_mapped
        ]

        # Debug prints
        print("\nInput values:")
        for name, value in zip(FEATURE_NAMES, raw_values):
            print(f"{name}: {value}")

        # Create numpy array and predict
        raw_input = np.array([raw_values])
        print("Raw input:", raw_input)

        scaled_input = scaler.transform(raw_input)
        print("Scaled input:", scaled_input)

        # Use model.predict with same parameters as notebook
        prediction = model.predict(scaled_input, verbose=0)
        probability = float(prediction[0][0])
        print("Prediction probability:", probability)

        result = {
            "prediction": round(float(probability * 100), 2),
            "risk_level": "High" if probability > 0.5 else "Low",
        }

        return jsonify(result)

    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5000)
