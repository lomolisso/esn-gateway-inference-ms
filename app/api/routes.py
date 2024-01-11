import base64
from app.api import schemas
from app.dependencies import get_model_handler
from fastapi import APIRouter, Depends, File, UploadFile

pred_node_router = APIRouter(prefix="/pred-node-api")


@pred_node_router.post("/test/predictive-model")
async def test_predictive_model(
    predictive_model: UploadFile = File(...), model_handler=Depends(get_model_handler)
):
    model_bytes = await predictive_model.read()
    b64_encoded_model = base64.b64encode(model_bytes).decode("utf-8")
    model_handler.load_model(b64_encoded_model)
    return {"message": "Predictive model is loaded."}


@pred_node_router.post("/predictive-model", status_code=202)
async def upload_predictive_model(
    prediction_request: schemas.PredictiveModel,
    model_handler=Depends(get_model_handler),
):
    model_handler.load_model(prediction_request.b64_encoded_model)
    return {"message": "Model loaded successfully."}


@pred_node_router.post("/predict", response_model=schemas.PredictionResponse)
async def predict(
    prediction_request: schemas.PredictionRequest,
    model_handler=Depends(get_model_handler),
):
    prediction = model_handler.predict(prediction_request.measurement)
    return {
        "measurement": prediction_request.measurement,
        "prediction": prediction[0],
    }
