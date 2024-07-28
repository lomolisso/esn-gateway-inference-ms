import redis
from collections import deque
import json
from app.core.config import MAX_INFERENCE_QUEUE_SIZE, MAX_PREDICTION_HISTORY_LENGTH, ABNORMAL_LABELS, ABNORMAL_PREDICTION_THRESHOLD, CLOUD_INFERENCE_LAYER, GATEWAY_INFERENCE_LAYER, SENSOR_INFERENCE_LAYER, HEURISTIC_ERROR_CODE, ADAPTIVE_INFERENCE, MAX_INFERENCE_QUEUE_SIZE

def _is_prediction_abnormal(prediction) -> int:
    """
    If the prediction is abnormal, return 1. Otherwise, return 0.
    """
    return prediction in ABNORMAL_LABELS
    
def _update_prediction_history(redis_client: redis.Redis, gateway_name: str, sensor_name: str, prediction: int):
    is_abnormal: bool = _is_prediction_abnormal(prediction)

    key = f"history:{gateway_name}:{sensor_name}"
    _redis_value = redis_client.get(key)
    prediction_history = deque(json.loads(_redis_value), maxlen=MAX_PREDICTION_HISTORY_LENGTH) if _redis_value else deque(maxlen=MAX_PREDICTION_HISTORY_LENGTH)
    prediction_history.append(1 if is_abnormal else 0)
    redis_client.set(key, json.dumps(list(prediction_history)))

    return prediction_history


def clear_prediction_history(redis_client: redis.Redis, gateway_name: str, sensor_name: str):
    key = f"history:{gateway_name}:{sensor_name}"
    redis_client.delete(key)


def gateway_adaptive_heuristic(redis_client: redis.Redis, gateway_name: str, sensor_name: str, inf_queue_size: int, low_battery: bool, prediction: int) -> int:
    """
    If the heuristic is abnormal, return 1. Otherwise, return 0.
    """
    
    prediction_history = _update_prediction_history(redis_client, gateway_name, sensor_name, prediction)
 
    t = len(prediction_history)
    k = MAX_PREDICTION_HISTORY_LENGTH
    n = sum(prediction_history)
    q = inf_queue_size
    phi_n = ABNORMAL_PREDICTION_THRESHOLD
    
    print("=================================")
    print("GATEWAY ADAPTIVE HEURISTIC PARAMS:")
    print("=================================")
    print(f"Current length of prediction history: {t}")
    print(f"Max length of prediction history: {k}")
    print("=================================")
    print(f"Number of abnormal predictions: {n}")
    print(f"Threshold for abnormal predictions: {phi_n}")
    print("=================================")
    print(f"Current length of inference queue: {q}")
    print(f"Max length of inference queue: {MAX_INFERENCE_QUEUE_SIZE}")
    print("=================================")


    if q >= MAX_INFERENCE_QUEUE_SIZE:
        # set inference layer to cloud
        return CLOUD_INFERENCE_LAYER
    else: # q < MAX_INFERENCE_QUEUE_SIZE
        if t < k:
            return GATEWAY_INFERENCE_LAYER
        elif t == k:
            if n == phi_n:
                # set inference layer to cloud
                return CLOUD_INFERENCE_LAYER
            else: # 0 <= n < phi_n
                if n == 0:
                    if low_battery:
                        # maintain inference layer as gateway
                        return GATEWAY_INFERENCE_LAYER
                    else:
                        # set inference layer to sensor
                        return SENSOR_INFERENCE_LAYER
        else:
            raise ValueError(f"Invalid history length: {t}")
