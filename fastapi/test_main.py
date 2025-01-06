from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

payload = {
    "Age": 21.0,
    "Gender": "Female",
    "Annual Income": 18053.0,
    "Marital Status": "Divorced",
    "Number of Dependents": 2.0,
    "Education Level": "PhD",
    "Occupation": "Self-Employed",
    "Health Score": 17.8897152903,
    "Location": "Rural",
    "Policy Type": "Premium",
    "Previous Claims": 2.0,
    "Vehicle Age": 2.0,
    "Credit Score": 336.0,
    "Insurance Duration": 3.0,
    "Policy Start Date": "2022-09-17 15:21:39.198406",
    "Customer Feedback": "Poor",
    "Smoking Status": "No",
    "Exercise Frequency": "Daily",
    "Property Type": "Apartment",
}

true_target = 705.0


def test_predict():
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    predicted_value = response.json()
    assert isinstance(predicted_value, float)
    print(f"\nPredicted value: {predicted_value}")
    print(f"Expected value: {true_target}")
    assert abs(predicted_value - true_target) / true_target <= 1
