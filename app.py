```python
from flask import Flask, request, render_template_string
import joblib
import numpy as np
import os

# ============================================================
# Flask App
# ============================================================

app = Flask(__name__)

# Load trained model
MODEL_PATH = "model.pkl"
model = joblib.load(MODEL_PATH)


# ============================================================
# Prediction Labels
# ============================================================

CLASS_LABELS = {
    0: "Low Risk",
    1: "Medium Risk",
    2: "High Risk"
}


# ============================================================
# HTML + CSS
# ============================================================

HTML = """
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <title>Health Risk Prediction</title>

    <style>

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: "Segoe UI", Arial, sans-serif;
        }

        body {
            min-height: 100vh;
            background:
                radial-gradient(circle at top left, #dbeafe 0%, transparent 35%),
                radial-gradient(circle at bottom right, #e0e7ff 0%, transparent 35%),
                linear-gradient(135deg, #f8fafc, #eef2ff);

            color: #1e293b;
        }

        /* Main container */

        .container {
            width: 92%;
            max-width: 1100px;
            margin: 40px auto;
        }

        /* Header */

        .header {
            text-align: center;
            margin-bottom: 30px;
        }

        .header .icon {
            width: 70px;
            height: 70px;
            margin: 0 auto 15px;

            display: flex;
            align-items: center;
            justify-content: center;

            background: linear-gradient(135deg, #4f46e5, #7c3aed);
            color: white;

            border-radius: 20px;

            font-size: 32px;

            box-shadow: 0 12px 30px rgba(79, 70, 229, 0.25);
        }

        .header h1 {
            font-size: 34px;
            font-weight: 700;
            color: #111827;
        }

        .header p {
            margin-top: 8px;
            color: #64748b;
            font-size: 15px;
        }

        /* Card */

        .card {
            background: rgba(255, 255, 255, 0.92);

            border: 1px solid rgba(226, 232, 240, 0.9);

            border-radius: 24px;

            padding: 35px;

            box-shadow:
                0 20px 50px rgba(15, 23, 42, 0.08),
                0 4px 15px rgba(15, 23, 42, 0.04);
        }

        /* Section */

        .section {
            margin-bottom: 30px;
        }

        .section-title {
            display: flex;
            align-items: center;
            gap: 10px;

            margin-bottom: 20px;

            font-size: 19px;
            font-weight: 650;

            color: #1e293b;
        }

        .section-title span {
            width: 34px;
            height: 34px;

            display: flex;
            align-items: center;
            justify-content: center;

            border-radius: 10px;

            background: #eef2ff;
            color: #4f46e5;

            font-size: 16px;
        }

        /* Grid */

        .form-grid {
            display: grid;

            grid-template-columns:
                repeat(3, minmax(0, 1fr));

            gap: 18px;
        }

        /* Form group */

        .form-group {
            display: flex;
            flex-direction: column;
        }

        .form-group label {
            margin-bottom: 7px;

            font-size: 13px;
            font-weight: 600;

            color: #475569;
        }

        .form-group input,
        .form-group select {
            width: 100%;

            padding: 12px 13px;

            border: 1px solid #dbe2ea;

            border-radius: 11px;

            background: #f8fafc;

            color: #1e293b;

            font-size: 14px;

            outline: none;

            transition: 0.2s ease;
        }

        .form-group input:focus,
        .form-group select:focus {
            border-color: #6366f1;

            background: white;

            box-shadow:
                0 0 0 3px rgba(99, 102, 241, 0.10);
        }

        .form-group input::placeholder {
            color: #94a3b8;
        }

        /* Help text */

        .help {
            margin-top: 5px;

            font-size: 11px;

            color: #94a3b8;
        }

        /* Button */

        .button-container {
            margin-top: 10px;
            text-align: center;
        }

        .predict-btn {
            width: 100%;

            padding: 15px;

            border: none;

            border-radius: 13px;

            background:
                linear-gradient(
                    135deg,
                    #4f46e5,
                    #7c3aed
                );

            color: white;

            font-size: 16px;

            font-weight: 650;

            cursor: pointer;

            box-shadow:
                0 10px 25px rgba(79, 70, 229, 0.25);

            transition: 0.25s ease;
        }

        .predict-btn:hover {
            transform: translateY(-2px);

            box-shadow:
                0 14px 30px rgba(79, 70, 229, 0.35);
        }

        .predict-btn:active {
            transform: translateY(0);
        }

        /* Result */

        .result {
            margin-top: 30px;

            padding: 25px;

            border-radius: 18px;

            text-align: center;

            background: linear-gradient(
                135deg,
                #eef2ff,
                #f5f3ff
            );

            border: 1px solid #ddd6fe;
        }

        .result-title {
            font-size: 13px;

            text-transform: uppercase;

            letter-spacing: 1px;

            color: #64748b;

            margin-bottom: 8px;
        }

        .result-value {
            font-size: 30px;

            font-weight: 750;

            color: #4f46e5;
        }

        .confidence {
            margin-top: 8px;

            font-size: 13px;

            color: #64748b;
        }

        /* Error */

        .error {
            margin-top: 20px;

            padding: 15px;

            border-radius: 12px;

            background: #fef2f2;

            border: 1px solid #fecaca;

            color: #b91c1c;

            text-align: center;

            font-size: 14px;
        }

        /* Footer */

        .footer {
            text-align: center;

            margin-top: 25px;

            font-size: 12px;

            color: #94a3b8;
        }

        /* Responsive */

        @media (max-width: 850px) {

            .form-grid {
                grid-template-columns:
                    repeat(2, minmax(0, 1fr));
            }

        }

        @media (max-width: 550px) {

            .container {
                width: 95%;
                margin: 20px auto;
            }

            .card {
                padding: 22px;
            }

            .form-grid {
                grid-template-columns: 1fr;
            }

            .header h1 {
                font-size: 27px;
            }

        }

    </style>
</head>


<body>

<div class="container">

    <div class="header">

        <div class="icon">
            ♥
        </div>

        <h1>Health Risk Prediction</h1>

        <p>
            AI-powered health risk assessment using your trained machine learning model
        </p>

    </div>


    <div class="card">

        <form method="POST">

            <!-- ================================================= -->
            <!-- Personal Information -->
            <!-- ================================================= -->

            <div class="section">

                <div class="section-title">
                    <span>👤</span>
                    Personal Information
                </div>

                <div class="form-grid">

                    <div class="form-group">
                        <label>Age</label>
                        <input
                            type="number"
                            name="age"
                            min="1"
                            max="120"
                            step="1"
                            placeholder="e.g. 35"
                            required
                        >
                    </div>


                    <div class="form-group">

                        <label>Gender</label>

                        <input
                            type="number"
                            name="gender"
                            step="1"
                            placeholder="Encoded value"
                            required
                        >

                        <span class="help">
                            Enter the numeric encoding used during training
                        </span>

                    </div>


                    <div class="form-group">

                        <label>City</label>

                        <input
                            type="number"
                            name="city"
                            step="1"
                            placeholder="Encoded value"
                            required
                        >

                        <span class="help">
                            Enter the numeric encoding used during training
                        </span>

                    </div>


                    <div class="form-group">

                        <label>BMI</label>

                        <input
                            type="number"
                            name="bmi"
                            min="0"
                            step="0.01"
                            placeholder="e.g. 24.5"
                            required
                        >

                    </div>


                    <div class="form-group">

                        <label>Waist Circumference (cm)</label>

                        <input
                            type="number"
                            name="waist_circumference_cm"
                            min="0"
                            step="0.1"
                            placeholder="e.g. 82"
                            required
                        >

                    </div>


                    <div class="form-group">

                        <label>Income Bracket</label>

                        <input
                            type="number"
                            name="income_bracket"
                            step="1"
                            placeholder="Encoded value"
                            required
                        >

                        <span class="help">
                            Enter the numeric encoding used during training
                        </span>

                    </div>

                </div>

            </div>


            <!-- ================================================= -->
            <!-- Lifestyle -->
            <!-- ================================================= -->

            <div class="section">

                <div class="section-title">
                    <span>🏃</span>
                    Lifestyle Information
                </div>

                <div class="form-grid">

                    <div class="form-group">

                        <label>Family History of Diabetes</label>

                        <select name="family_history_diabetes" required>
                            <option value="">Select</option>
                            <option value="0">No (0)</option>
                            <option value="1">Yes (1)</option>
                        </select>

                    </div>


                    <div class="form-group">

                        <label>Physical Activity Level</label>

                        <input
                            type="number"
                            name="physical_activity_level"
                            step="1"
                            placeholder="Encoded value"
                            required
                        >

                        <span class="help">
                            Numeric encoding from training dataset
                        </span>

                    </div>


                    <div class="form-group">

                        <label>Diet Type</label>

                        <input
                            type="number"
                            name="diet_type"
                            step="1"
                            placeholder="Encoded value"
                            required
                        >

                        <span class="help">
                            Numeric encoding from training dataset
                        </span>

                    </div>


                    <div class="form-group">

                        <label>Smoking Status</label>

                        <input
                            type="number"
                            name="smoking_status"
                            step="1"
                            placeholder="Encoded value"
                            required
                        >

                        <span class="help">
                            Numeric encoding from training dataset
                        </span>

                    </div>


                    <div class="form-group">

                        <label>Alcohol Consumption</label>

                        <input
                            type="number"
                            name="alcohol_consumption"
                            step="1"
                            placeholder="Encoded value"
                            required
                        >

                        <span class="help">
                            Numeric encoding from training dataset
                        </span>

                    </div>


                    <div class="form-group">

                        <label>Hours of Sleep / Night</label>

                        <input
                            type="number"
                            name="hours_sleep_per_night"
                            min="0"
                            max="24"
                            step="0.1"
                            placeholder="e.g. 7.5"
                            required
                        >

                    </div>


                    <div class="form-group">

                        <label>Stress Level</label>

                        <input
                            type="number"
                            name="stress_level"
                            step="1"
                            placeholder="Encoded value"
                            required
                        >

                    </div>

                </div>

            </div>


            <!-- ================================================= -->
            <!-- Medical Information -->
            <!-- ================================================= -->

            <div class="section">

                <div class="section-title">
                    <span>🩺</span>
                    Medical Information
                </div>

                <div class="form-grid">

                    <div class="form-group">

                        <label>Fasting Blood Sugar</label>

                        <input
                            type="number"
                            name="fasting_blood_sugar"
                            min="0"
                            step="0.1"
                            placeholder="mg/dL"
                            required
                        >

                    </div>


                    <div class="form-group">

                        <label>HbA1c Level</label>

                        <input
                            type="number"
                            name="hba1c_level"
                            min="0"
                            step="0.1"
                            placeholder="e.g. 5.6"
                            required
                        >

                    </div>


                    <div class="form-group">

                        <label>Systolic Blood Pressure</label>

                        <input
                            type="number"
                            name="blood_pressure_systolic"
                            min="0"
                            step="1"
                            placeholder="mmHg"
                            required
                        >

                    </div>


                    <div class="form-group">

                        <label>Diastolic Blood Pressure</label>

                        <input
                            type="number"
                            name="blood_pressure_diastolic"
                            min="0"
                            step="1"
                            placeholder="mmHg"
                            required
                        >

                    </div>

                </div>

            </div>


            <!-- ================================================= -->
            <!-- Button -->
            <!-- ================================================= -->

            <div class="button-container">

                <button
                    type="submit"
                    class="predict-btn"
                >
                    🔍 Predict Health Risk
                </button>

            </div>

        </form>


        {% if prediction %}

        <div class="result">

            <div class="result-title">
                Prediction Result
            </div>

            <div class="result-value">
                {{ prediction }}
            </div>

            {% if probability %}
            <div class="confidence">
                Model confidence: {{ probability }}%
            </div>
            {% endif %}

        </div>

        {% endif %}


        {% if error %}

        <div class="error">
            ⚠ {{ error }}
        </div>

        {% endif %}

    </div>


    <div class="footer">

        Health Risk Prediction System • Machine Learning Powered

    </div>

</div>

</body>

</html>
"""


# ============================================================
# Routes
# ============================================================

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    probability = None
    error = None

    if request.method == "POST":

        try:

            # ------------------------------------------------
            # Read inputs in EXACT model feature order
            # ------------------------------------------------

            features = [

                float(request.form["age"]),

                float(request.form["gender"]),

                float(request.form["city"]),

                float(request.form["bmi"]),

                float(request.form["family_history_diabetes"]),

                float(request.form["physical_activity_level"]),

                float(request.form["diet_type"]),

                float(request.form["smoking_status"]),

                float(request.form["alcohol_consumption"]),

                float(request.form["hours_sleep_per_night"]),

                float(request.form["stress_level"]),

                float(request.form["fasting_blood_sugar"]),

                float(request.form["hba1c_level"]),

                float(request.form["blood_pressure_systolic"]),

                float(request.form["blood_pressure_diastolic"]),

                float(request.form["waist_circumference_cm"]),

                float(request.form["income_bracket"])

            ]


            # Convert to numpy array

            input_data = np.array(features).reshape(1, -1)


            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            result = model.predict(input_data)[0]

            prediction = CLASS_LABELS.get(
                int(result),
                f"Class {int(result)}"
            )


            # ------------------------------------------------
            # Probability
            # ------------------------------------------------

            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(input_data)[0]

                probability = round(
                    float(max(probabilities)) * 100,
                    2
                )


        except Exception as e:

            error = (
                "Unable to process the prediction. "
                "Please check that all values are entered correctly."
            )

            print("Prediction Error:", e)


    return render_template_string(
        HTML,
        prediction=prediction,
        probability=probability,
        error=error
    )


# ============================================================
# Health Check
# ============================================================

@app.route("/health")
def health():

    return {
        "status": "healthy",
        "model": "loaded"
    }


# ============================================================
# Run App
# ============================================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
```
