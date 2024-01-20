"""
celery tasks for esn-predictive-node

lib: celery
broker: redis

18-01-2024
"""

from app.celery_worker import celery
from app.celery_worker.tensorflow import PredictiveModelHandler
from app.config import IS_GATEWAY
from app.celery_worker import utils


@celery.task(name="load_model_task", ignore_result=True)
def load_model_task(b64_encoded_model: str):
    """
    celery task for loading model
    """
    model_handler = PredictiveModelHandler()
    model_handler.load_model(b64_encoded_model)


@celery.task(name="predict_task", ignore_result=True)
def predict_task(device_name: str, gateway_name: str, request_timestamp: str, measurement: float, debug_mode: bool):
    """
    celery task for prediction
    """
    model_handler = PredictiveModelHandler()
    prediction = float(model_handler.predict(measurement)[0])
    utils.check_prediction(measurement=measurement, prediction=prediction)
    prediction_source_layer = "gateway" if IS_GATEWAY else "cloud"
    
    if debug_mode:
        payload = {
            "prediction_source_layer": prediction_source_layer,
            "request_timestamp": request_timestamp,
            "measurement": measurement,
            "prediction": prediction,
        }
        
        utils.prediction_command(
            device_name=device_name,
            gateway_name=gateway_name,
            payload=payload
        )
    