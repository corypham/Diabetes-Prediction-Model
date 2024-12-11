# Diabetes Prediction Model

A machine learning web application that predicts diabetes risk based on various health metrics. The model uses a deep neural network trained on a dataset of 100,000 patient records to provide risk assessment and probability scores.

## Features

- Interactive web interface for inputting health metrics
- Real-time predictions using a trained neural network model
- Risk level assessment (High/Low)
- Probability score calculation
- Responsive design for both desktop and mobile use

## Health Metrics Used

- Age
- Hypertension status
- Heart disease status
- BMI (Body Mass Index)
- HbA1c level
- Blood glucose level
- Gender
- Smoking history

## Installation

1. Clone the repository:

```

git clone https://github.com/yourusername/diabetes-prediction-model.git
```

2.
3. Create and activate a virtual environment (recommended):

```
python -m venv venv
```

3. Install required dependencies:

```

pip install -r diabetes_app/requirements.txt
```

## Running Application

1. Navigate to the app directory:

```
cd diabetes_app
```

2. Start Flask Server

```
python3 app.py
```

3. Open web browser and visit:
   http://localhost:5000

## Model Details

- Architecture: Deep Neural Network
- Input Features: 8 health metrics
- Output: Binary classification (diabetes risk)
- Training Dataset: 100,000 patient records
- Evaluation Metrics: Accuracy, ROC-AUC, Precision, Recall

## Project Structure

diabetes_app/
├── app.py # Flask application
├── requirements.txt # Python dependencies
├── model/ # Trained model files
│ ├── diabetes_model.h5
│ └── scaler.pkl
├── static/ # CSS and JavaScript files
└── templates/ # HTML templates
└── index.html

## Requirements

- Python 3.8+
- Flask 2.0.1
- TensorFlow 2.12.0
- NumPy 1.21.0
- scikit-learn 0.24.2
- Additional dependencies listed in requirements.txt

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.
