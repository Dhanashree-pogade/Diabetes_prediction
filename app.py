from flask import Flask, request, render_template_string
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
model = joblib.load(MODEL_PATH)

# Model output labels
CLASS_LABELS = {
    0: "Low Risk",
    1: "Medium Risk",
    2: "High Risk"
}

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
                radial-gradient(
                    circle at top left,
                    rgba(99, 102, 241, 0.20),
                    transparent 35%
                ),
                radial-gradient(
                    circle at bottom right,
                    rgba(139, 92, 246, 0.18),
                    transparent 35%
                ),
                linear-gradient(
                    135deg,
                    #f8fafc,
                    #eef2ff
                );

            color: #1e293b;
        }

        .page {
            width: 94%;
            max-width: 1150px;
            margin: 35px auto;
        }

        /* HEADER */

        .header {
            text-align: center;
            margin-bottom: 30px;
        }

        .logo {
            width: 70px;
            height: 70px;

            margin: auto;
            margin-bottom: 16px;

            display: flex;
            align-items: center;
            justify-content: center;

            border-radius: 20px;

            background:
                linear-gradient(
                    135deg,
                    #4f46e5,
                    #7c3aed
                );

            color: white;

            font-size: 32px;

            box-shadow:
                0 15px 35px rgba(79, 70, 229, 0.25);
        }

        .header h1 {
            font-size: 36px;
            color: #111827;
            margin-bottom: 8px;
        }

        .header p {
            color: #64748b;
            font-size: 15px;
        }

        /* CARD */

        .card {
            background: rgba(255,255,255,0.96);

            border-radius: 25px;

            padding: 35px;

            border: 1px solid #e2e8f0;

            box-shadow:
                0 25px 60px rgba(15,23,42,0.08);
        }

        /* SECTION */

        .section {
            margin-bottom: 35px;
        }

        .section-header {
            display: flex;
            align-items: center;

            gap: 12px;

            margin-bottom: 22px;

            padding-bottom: 12px;

            border-bottom: 1px solid #e2e8f0;
        }

        .section-icon {
            width: 38px;
            height: 38px;

            border-radius: 11px;

            display: flex;
            align-items: center;
            justify-content: center;

            background: #eef2ff;

            font-size: 18px;
        }

        .section-header h2 {
            font-size: 19px;
            color: #1e293b;
        }

        /* FORM GRID */

        .form-grid {
            display: grid;

            grid-template-columns:
                repeat(3, minmax(0, 1fr));

            gap: 20px;
        }

        .form-group {
            display: flex;
            flex-direction: column;
        }

        label {
            font-size: 13px;

            font-weight: 600;

            color: #475569;

            margin-bottom: 7px;
        }

        input,
        select {

            width: 100%;

            padding: 13px 14px;

            border: 1px solid #dbe2ea;

            border-radius: 11px;

            background: #f8fafc;

            color: #1e293b;

            font-size: 14px;

            outline: none;

            transition: all 0.2s ease;
        }

        input:hover,
        select:hover {
            border-color: #a5b4fc;
        }

        input:focus,
        select:focus {

            background: white;

            border-color: #6366f1;

            box-shadow:
                0 0 0 4px rgba(99,102,241,0.10);
        }

        .help {
            font-size: 11px;

            color: #94a3b8;

            margin-top: 5px;
        }

        /* BUTTON */

        .button-area {
            margin-top: 10px;
        }

        .predict-button {

            width: 100%;

            border: none;

            padding: 16px;

            border-radius: 13px;

            background:
                linear-gradient(
                    135deg,
                    #4f46e5,
                    #7c3aed
                );

            color: white;

            font-size: 16px;

            font-weight: 700;

            cursor: pointer;

            box-shadow:
                0 12px 25px rgba(79,70,229,0.25);

            transition: all 0.25s ease;
        }

        .predict-button:hover {

            transform: translateY(-2px);

            box-shadow:
                0 17px 32px rgba(79,70,229,0.35);
        }

        .predict-button:active {
            transform: translateY(0);
        }

        /* RESULT */

        .result {

            margin-top: 30px;

            padding: 28px;

            border-radius: 18px;

            text-align: center;

            background:
                linear-gradient(
                    135deg,
                    #eef2ff,
                    #f5f3ff
                );

            border:
                1px solid #ddd6fe;
        }

        .result-label {

            font-size: 12px;

            text-transform: uppercase;

            letter-spacing: 1.5px;

            color: #64748b;

            margin-bottom: 8px;
        }

        .result-value {

            font-size: 32px;

            font-weight: 800;

            color: #4f46e5;
        }

        .confidence {

            margin-top: 10px;

            color: #64748b;

            font-size: 14px;
        }

        /* ERROR */

        .error {

            margin-top: 25px;

            padding: 15px;

            border-radius: 12px;

            background: #fef2f2;

            border: 1px solid #fecaca;

            color: #b91c1c;

            text-align: center;

            font-size: 14px;
        }

        /* INFO */

        .info {

            margin-top: 25px;

            padding: 14px 16px;

            border-radius: 12px;

            background: #f8fafc;

            border: 1px solid #e2e8f0;

            color: #64748b;

            font-size: 12px;

            line-height: 1.6;
        }

        /* FOOTER */

        .footer {

            text-align: center;

            margin-top: 25px;

            color: #94a3b8;

            font-size: 12px;
        }

        /* RESPONSIVE */

        @media (max-width: 900px) {

            .form-grid {
                grid-template-columns: repeat(2, 1fr);
            }

        }

        @media (max-width: 600px) {

            .page {
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
                font-size: 28px;
            }

            .logo {
                width: 60px;
                height: 60px;
            }

        }

    </style>

</head>


<body>

<div class="page">

    <div class="header">

        <div class="logo">
            ♥
        </div>

        <h1>Health Risk Prediction</h1>

        <p>
            Machine Learning Based Health Risk Assessment System
        </p>

    </div>


    <div class="card">

        <form method="POST">


            <!-- PERSONAL INFORMATION -->

            <div class="section">

                <div class="section-header">

                    <div class="section-icon">
                        👤
                    </div>

                    <h2>Personal Information</h2>

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
                            Enter the value used during model training
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
                            Enter the value used during model training
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
                            Enter the value used during model training
                        </span>

                    </div>

                </div>

            </div>


            <!-- LIFESTYLE -->

            <div class="section">

                <div class="section-header">

                    <div class="section-icon">
                        🏃
                    </div>

                    <h2>Lifestyle Information</h2>

                </div>


                <div class="form-grid">


                    <div class="form-group">

                        <label>Family History of Diabetes</label>

                        <select
                            name="family_history_diabetes"
                            required
                        >

                            <option value="">
                                Select
                            </option>

                            <option value="0">
                                No
                            </option>

                            <option value="1">
                                Yes
                            </option>

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


            <!-- MEDICAL INFORMATION -->

            <div class="section">

                <div class="section-header">

                    <div class="section-icon">
                        🩺
                    </div>

                    <h2>Medical Information</h2>

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


            <!-- BUTTON -->

            <div class="button-area">

                <button
                    type="submit"
                    class="predict-button"
                >
                    🔍 Predict Health Risk
                </button>

            </div>


        </form>


        {% if prediction %}

        <div class="result">

            <div class="result-label">
                Prediction Result
            </div>

            <div class="result-value">
                {{ prediction }}
            </div>

            {% if probability %}

            <div class="confidence">
                Model Confidence: {{ probability }}%
            </div>

            {% endif %}

        </div>

        {% endif %}


        {% if error %}

        <div class="error">
            ⚠️ {{ error }}
        </div>

        {% endif %}


        <div class="info">

            <strong>Note:</strong>
            This application provides a machine-learning-based
            prediction and should not be considered a medical diagnosis.
            For categorical fields, enter the numeric encoding used
            when the model was trained.

        </div>


    </div>


    <div class="footer">

        Health Risk Prediction System
        •
        Machine Learning Powered

    </div>


</div>

</body>

</html>
"""


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    probability = None
    error = None

    if request.method == "POST":

        try:

            features = [

                float(request.form["age"]),

                float(request.form["gender"]),

                float(request.form["city"]),

                float(request.form["bmi"]),

                float(
                    request.form[
                        "family_history_diabetes"
                    ]
                ),

                float(
                    request.form[
                        "physical_activity_level"
                    ]
                ),

                float(
                    request.form["diet_type"]
                ),

                float(
                    request.form["smoking_status"]
                ),

                float(
                    request.form["alcohol_consumption"]
                ),

                float(
                    request.form[
                        "hours_sleep_per_night"
                    ]
                ),

                float(
                    request.form["stress_level"]
                ),

                float(
                    request.form[
                        "fasting_blood_sugar"
                    ]
                ),

                float(
                    request.form["hba1c_level"]
                ),

                float(
                    request.form[
                        "blood_pressure_systolic"
                    ]
                ),

                float(
                    request.form[
                        "blood_pressure_diastolic"
                    ]
                ),

                float(
                    request.form[
                        "waist_circumference_cm"
                    ]
                ),

                float(
                    request.form["income_bracket"]
                )

            ]


            input_data = np.array(
                features,
                dtype=float
            ).reshape(1, -1)


            # Prediction

            result = model.predict(input_data)[0]

            prediction = CLASS_LABELS.get(
                int(result),
                f"Class {int(result)}"
            )


            # Prediction probability

            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(
                    input_data
                )[0]

                probability = round(
                    float(
                        np.max(probabilities)
                    ) * 100,
                    2
                )


        except Exception as e:

            print(
                "Prediction Error:",
                str(e)
            )

            error = (
                "Prediction could not be completed. "
                "Please check all entered values and "
                "make sure they match the model's training format."
            )


    return render_template_string(

        HTML,

        prediction=prediction,

        probability=probability,

        error=error

    )


@app.route("/health")
def health():

    return {
        "status": "healthy",
        "model": "loaded"
    }


if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
