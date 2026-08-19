import joblib
import numpy as np
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Load the model directly
model = joblib.load("model.pkl")

# List of expected features in order
FEATURE_NAMES = [
    "age",
    "gender",
    "city",
    "bmi",
    "family_history_diabetes",
    "physical_activity_level",
    "diet_type",
    "smoking_status",
    "alcohol_consumption",
    "hours_sleep_per_night",
    "stress_level",
    "fasting_blood_sugar",
    "hba1c_level",
    "blood_pressure_systolic",
    "blood_pressure_diastolic",
    "waist_circumference_cm",
    "income_bracket",
]

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Diabetes Risk Assessment</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --input-bg: #334155;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --accent: #38bdf8;
            --accent-hover: #0284c7;
            --border: #475569;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', sans-serif;
        }

        body {
            background-color: var(--bg-color);
            color: var(--text-main);
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            padding: 2rem 1rem;
        }

        .container {
            width: 100%;
            max-width: 900px;
            background: var(--card-bg);
            border-radius: 16px;
            padding: 2.5rem;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        header {
            margin-bottom: 2rem;
            text-align: center;
        }

        header h1 {
            font-size: 2rem;
            font-weight: 700;
            color: var(--accent);
            margin-bottom: 0.5rem;
        }

        header p {
            color: var(--text-muted);
            font-size: 0.95rem;
        }

        .result-box {
            margin-bottom: 2rem;
            padding: 1.25rem;
            border-radius: 10px;
            text-align: center;
            font-weight: 600;
            font-size: 1.1rem;
            background-color: rgba(56, 189, 248, 0.1);
            border: 1px solid var(--accent);
            color: var(--accent);
        }

        .grid-form {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 1.25rem;
        }

        .input-group {
            display: flex;
            flex-direction: column;
        }

        .input-group label {
            font-size: 0.85rem;
            font-weight: 500;
            color: var(--text-muted);
            margin-bottom: 0.4rem;
            text-transform: capitalize;
        }

        .input-group input {
            background-color: var(--input-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 0.75rem 1rem;
            color: var(--text-main);
            font-size: 0.95rem;
            outline: none;
            transition: all 0.2s ease;
        }

        .input-group input:focus {
            border-color: var(--accent);
            box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.2);
        }

        .submit-btn {
            grid-column: 1 / -1;
            margin-top: 1rem;
            background-color: var(--accent);
            color: #0f172a;
            font-weight: 600;
            font-size: 1rem;
            padding: 0.85rem;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: background 0.2s ease;
        }

        .submit-btn:hover {
            background-color: var(--accent-hover);
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Diabetes Risk Prediction</h1>
            <p>Enter the patient metrics below to generate a model prediction.</p>
        </header>

        {% if prediction_text %}
            <div class="result-box">
                {{ prediction_text }}
            </div>
        {% endif %}

        <form action="/predict" method="post" class="grid-form">
            {% for feature in features %}
                <div class="input-group">
                    <label for="{{ feature }}">{{ feature.replace('_', ' ') }}</label>
                    <input 
                        type="number" 
                        step="any" 
                        id="{{ feature }}" 
                        name="{{ feature }}" 
                        placeholder="Enter value" 
                        required
                    >
                </div>
            {% endfor %}
            <button type="submit" class="submit-btn">Run Prediction</button>
        </form>
    </div>
</body>
</html>
"""


@app.route("/")
def home():
    return render_template_string(HTML_TEMPLATE, features=FEATURE_NAMES)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract features in the precise order expected by the model
        input_data = [
            float(request.form.get(feature, 0)) for feature in FEATURE_NAMES
        ]
        features_array = np.array([input_data])

        # Run model prediction
        prediction = model.predict(features_array)[0]
        probabilities = model.predict_proba(features_array)[0]

        result = f"Prediction Class: {prediction} (Probability: {max(probabilities):.2%})"
    except Exception as e:
        result = f"Error during prediction: {str(e)}"

    return render_template_string(
        HTML_TEMPLATE, features=FEATURE_NAMES, prediction_text=result
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
