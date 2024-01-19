from app.api import schemas
from app.celery_worker import tasks
from fastapi import APIRouter

pred_node_router = APIRouter(prefix="/pred-node-api")

@pred_node_router.post("/predictive-model", status_code=202)
async def upload_predictive_model(
    prediction_request: schemas.PredictiveModel
):
    tasks.load_model_task.delay(prediction_request.b64_encoded_model)
    return {"message": "Model is being loaded."}


@pred_node_router.post("/predict")
async def predict(
    prediction_request: schemas.PredictionRequest
):
    tasks.predict_task.delay(**prediction_request.model_dump())
    return {"message": "Prediction is being processed."}
    
