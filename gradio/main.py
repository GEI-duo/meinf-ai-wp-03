import gradio as gr
import requests
import argparse
from datetime import datetime

parser = argparse.ArgumentParser(description="Gradio Insurance Premium Prediction App")
parser.add_argument(
    "--endpoint",
    type=str,
    required=False,
    default="http://localhost:8000/predict",
    help="API endpoint for prediction (e.g., http://localhost:8000/predict)",
)
args = parser.parse_args()
API_ENDPOINT = args.endpoint


# Prediction Function
def predict_premium(
    age,
    gender,
    income,
    marital_status,
    dependents,
    education,
    occupation,
    health_score,
    location,
    policy_type,
    claims,
    vehicle_age,
    credit_score,
    insurance_duration,
    policy_start_date,
    feedback,
    smoking_status,
    exercise_frequency,
    property_type,
):
    data = {
        "Age": age,
        "Gender": gender,
        "Annual Income": income,
        "Marital Status": marital_status,
        "Number of Dependents": dependents,
        "Education Level": education,
        "Occupation": occupation if occupation else None,
        "Health Score": health_score,
        "Location": location,
        "Policy Type": policy_type,
        "Previous Claims": claims,
        "Vehicle Age": vehicle_age,
        "Credit Score": credit_score if credit_score else None,
        "Insurance Duration": insurance_duration,
        "Policy Start Date": datetime.fromtimestamp(policy_start_date).strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "Customer Feedback": feedback,
        "Smoking Status": smoking_status,
        "Exercise Frequency": exercise_frequency,
        "Property Type": property_type,
    }

    try:
        response = requests.post(API_ENDPOINT, json=data)
        response.raise_for_status()
        result = float(response.json())
        return result
    except requests.exceptions.RequestException as e:
        return f"Error: {e}"


with gr.Blocks(theme="default", title="Insurance Premium Prediction App") as demo:
    gr.Markdown(
        """
        # 🚗 **Insurance Premium Prediction App**  
        This application helps you predict insurance premiums based on various user inputs.  
        Simply fill in the required information, click 'Predict Premium', and view the result.  
        """
    )

    with gr.Column():
        age = gr.Slider(18, 100, step=1, label="Age", value=35)
        gender = gr.Radio(["Male", "Female"], label="Gender", value="Male")
        income = gr.Number(label="Annual Income", value=24862.0)
        marital_status = gr.Radio(
            ["Single", "Married", "Divorced"], label="Marital Status", value="Married"
        )
        dependents = gr.Number(label="Number of Dependents", value=3.0)
        education = gr.Radio(
            ["High School", "Bachelor's", "Master's", "PhD"],
            label="Education Level",
            value="PhD",
        )
        occupation = gr.Radio(
            ["Employed", "Self-Employed", "Unemployed"],
            label="Occupation",
            value="Employed",
        )
        health_score = gr.Slider(1, 100, label="Health Score", value=15.6867)
        location = gr.Radio(
            ["Urban", "Suburban", "Rural"], label="Location", value="Urban"
        )
        policy_type = gr.Radio(
            ["Basic", "Comprehensive", "Premium"], label="Policy Type", value="Basic"
        )
        claims = gr.Slider(0, 10, step=1, label="Previous Claims", value=3.0)
        vehicle_age = gr.Slider(1, 100, step=1, label="Vehicle Age (Years)", value=3.0)
        credit_score = gr.Slider(0, 1000, label="Credit Score (Optional)")
        insurance_duration = gr.Slider(
            1, 10, step=1, label="Insurance Duration (Years)", value=6.0
        )
        policy_start_date = gr.DateTime(
            label="Policy Start Date (YYYY-MM-DD HH:MM:SS)",
            value="2019-12-19 15:21:39",
        )
        feedback = gr.Radio(
            ["Poor", "Average", "Good"], label="Customer Feedback", value="Average"
        )
        smoking_status = gr.Radio(["Yes", "No"], label="Smoking Status", value="Yes")
        exercise_frequency = gr.Radio(
            ["Daily", "Weekly", "Monthly", "Rarely"],
            label="Exercise Frequency",
            value="Rarely",
        )
        property_type = gr.Radio(
            ["House", "Apartment", "Condo"], label="Property Type", value="Condo"
        )

    with gr.Row():
        predict_button = gr.Button("🚀 Predict Premium")
        result = gr.Number(label="Premium Amount", value=0.0, precision=4)

    predict_button.click(
        predict_premium,
        inputs=[
            age,
            gender,
            income,
            marital_status,
            dependents,
            education,
            occupation,
            health_score,
            location,
            policy_type,
            claims,
            vehicle_age,
            credit_score,
            insurance_duration,
            policy_start_date,
            feedback,
            smoking_status,
            exercise_frequency,
            property_type,
        ],
        outputs=result,
    )

demo.launch()
