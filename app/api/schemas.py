"""
This module contains the pydantic models (schemas) for the API.
09/01/2024
"""

from pydantic import BaseModel

class PredictiveModel(BaseModel):
    """Model for predictive model"""
    b64_encoded_model: str
    pred_model_size: int

class PredictionRequest(BaseModel):
    """Request body for prediction endpoint"""
    measurement: float

class PredictionResponse(BaseModel):
    """Response body for prediction endpoint"""
    measurement: float
    prediction: float

