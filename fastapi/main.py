from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from joblib import load
import pandas as pd
from pathlib import Path
import logging
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

DATA_PATH = Path.joinpath(Path.cwd().parent, "data")
CACHE_PATH = Path.joinpath(Path.cwd().parent, "cache")
CACHE_MODELS_PATH = Path.joinpath(CACHE_PATH, "models")

PIPELINE_PATH = CACHE_MODELS_PATH / "pipeline.pkl"
pipeline = load(PIPELINE_PATH)

app = FastAPI()


# Define the request body schema
class PredictionRequest(BaseModel):
    Age: float
    Gender: str = Field(..., pattern="^(Male|Female)$")
    Annual_Income: float = Field(alias="Annual Income")
    Marital_Status: str = Field(
        ..., pattern="^(Single|Married|Divorced)$", alias="Marital Status"
    )
    Number_of_Dependents: Optional[float] = Field(None, alias="Number of Dependents")
    Education_Level: str = Field(
        ..., pattern="^(High School|Bachelor's|Master's|PhD)$", alias="Education Level"
    )
    Occupation: str = Field(..., pattern="^(Employed|Self-Employed|Unemployed)$")
    Health_Score: float = Field(..., alias="Health Score")
    Location: str = Field(..., pattern="^(Urban|Suburban|Rural)$")
    Policy_Type: str = Field(
        ..., pattern="^(Basic|Comprehensive|Premium)$", alias="Policy Type"
    )
    Previous_Claims: float = Field(..., alias="Previous Claims")
    Vehicle_Age: float = Field(..., alias="Vehicle Age")
    Credit_Score: Optional[float] = Field(None, alias="Credit Score")
    Insurance_Duration: float = Field(..., alias="Insurance Duration")
    Policy_Start_Date: str = Field(..., alias="Policy Start Date")
    Customer_Feedback: Optional[str] = Field(None, alias="Customer Feedback")
    Smoking_Status: str = Field(..., pattern="^(Yes|No)$", alias="Smoking Status")
    Exercise_Frequency: str = Field(
        ..., pattern="^(Daily|Weekly|Monthly|Rarely)$", alias="Exercise Frequency"
    )
    Property_Type: str = Field(
        ..., pattern="^(House|Apartment|Condo)$", alias="Property Type"
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    exc_str = f"{exc}".replace("\n", " ").replace("   ", " ")
    logging.error(f"{request}: {exc_str}")
    content = {"status_code": 10422, "message": exc_str, "data": None}
    return JSONResponse(
        content=content, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
    )


@app.post("/predict")
async def predict_premium(request: PredictionRequest) -> float:
    try:
        request_dict = request.model_dump(by_alias=True)
        data = pd.DataFrame([request_dict])
        prediction = pipeline.predict(data)
        return prediction[0]
    except Exception as e:
        print(e)
        raise HTTPException(status_code=400, detail=str(e))
