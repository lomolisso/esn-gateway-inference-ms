import redis
from collections import deque
import json
from app.api.schemas import PredictionResult
from redis import Redis
from app.core.config import (
    MAX_INFERENCE_QUEUE_SIZE,
    PREDICTION_HISTORY_LENGTH,
    ABNORMAL_LABELS,
    NORMAL_PREDICTION_THRESHOLD,
    ABNORMAL_PREDICTION_THRESHOLD,
    CLOUD_INFERENCE_LAYER,
    GATEWAY_INFERENCE_LAYER,
    SENSOR_INFERENCE_LAYER,
    HEURISTIC_ERROR_CODE,
    ADAPTIVE_INFERENCE,
    MAX_INFERENCE_QUEUE_SIZE,
    REDIS_HOST,
    REDIS_PORT,
    REDIS_DB_HISTORY,
    REDIS_DB_CELERY_BROKER,
)

class RedisContextManager:
    def __init__(self, *args, **kwargs):
        self.redis = redis.Redis(*args, **kwargs)

    def __enter__(self):
        return self.redis

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.redis.close()

def _get_prediction_queue_size() -> int:
    with RedisContextManager(
        host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB_CELERY_BROKER
    ) as redis_client:
        return redis_client.llen("prediction_queue")


def _is_prediction_abnormal(prediction) -> int:
    """
    If the prediction is abnormal, return 1. Otherwise, return 0.
    """
    return prediction in ABNORMAL_LABELS


def _get_prediction_counter(gateway_name: str, sensor_name: str):
    with RedisContextManager(
        host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB_HISTORY
    ) as redis_client:
        key = f"counter:{gateway_name}:{sensor_name}"
        _redis_value = redis_client.get(key)
        prediction_counter = int(_redis_value) if _redis_value else 0

        return prediction_counter


def _set_prediction_counter(
    gateway_name: str,
    sensor_name: str,
    prediction_counter: int,
):
    with RedisContextManager(
        host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB_HISTORY
    ) as redis_client:
        key = f"counter:{gateway_name}:{sensor_name}"
        redis_client.set(key, prediction_counter)


def update_prediction_counter(gateway_name: str, sensor_name: str):
    prediction_counter = _get_prediction_counter(
        gateway_name, sensor_name
    )
    prediction_counter += 1
    _set_prediction_counter(gateway_name, sensor_name, prediction_counter)

    return prediction_counter


def _get_prediction_history(gateway_name: str, sensor_name: str):
    with RedisContextManager(
        host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB_HISTORY
    ) as redis_client:
        key = f"history:{gateway_name}:{sensor_name}"
        _redis_value = redis_client.get(key)
        prediction_history = (
            deque(json.loads(_redis_value), maxlen=PREDICTION_HISTORY_LENGTH)
            if _redis_value
            else deque(maxlen=PREDICTION_HISTORY_LENGTH)
        )

        return prediction_history


def _set_prediction_history(
    gateway_name: str,
    sensor_name: str,
    prediction_history: deque,
):
    with RedisContextManager(
        host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB_HISTORY
    ) as redis_client:
        key = f"history:{gateway_name}:{sensor_name}"
        redis_client.set(key, json.dumps(list(prediction_history)))


def update_prediction_history(gateway_name: str, sensor_name: str, prediction: int):
    is_abnormal: bool = _is_prediction_abnormal(prediction)
    prediction_history = _get_prediction_history(
        gateway_name, sensor_name
    )
    prediction_history.append(1 if is_abnormal else 0)
    _set_prediction_history(gateway_name, sensor_name, prediction_history)

    return prediction_history


def clear_prediction_counter(gateway_name: str, sensor_name: str):
    with RedisContextManager(
        host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB_HISTORY
    ) as redis_client:
        key = f"counter:{gateway_name}:{sensor_name}"
        redis_client.delete(key)


def clear_prediction_history(gateway_name: str, sensor_name: str):
    with RedisContextManager(
        host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB_HISTORY
    ) as redis_client:
        key = f"history:{gateway_name}:{sensor_name}"
        redis_client.delete(key)


def gateway_adaptive_inference_heuristic(prediction_result: PredictionResult) -> int:
    """
    Gateway Adaptive Inference Heuristic

    u_t: state counter at time step t
    M_t: prediction history at time step t
    sigma(M_t): number of abnormal predictions in history at time step t
    q_t: length of inference queue at time step t
    psi_q: max length of inference queue
    phi_g: threshold for normal predictions, if less than phi_g, set inference layer to sensor
    psi_g: threshold for abnormal predictions, if greater than psi_g, set inference layer to cloud
    """

    gateway_name = prediction_result.gateway_name
    sensor_name = prediction_result.sensor_name
    low_battery = prediction_result.low_battery
    prediction = prediction_result.prediction

    u_t = update_prediction_counter(gateway_name, sensor_name)
    prediction_history = update_prediction_history(
        gateway_name, sensor_name, prediction
    )
    assert u_t >= len(prediction_history)
    sigma_M_t = sum(prediction_history)
    q_t = _get_prediction_queue_size()

    m = PREDICTION_HISTORY_LENGTH
    psi_q = MAX_INFERENCE_QUEUE_SIZE
    phi_g = NORMAL_PREDICTION_THRESHOLD
    psi_g = ABNORMAL_PREDICTION_THRESHOLD

    print("=================================")
    print("GATEWAY ADAPTIVE HEURISTIC PARAMS:")
    print("=================================")
    print(f"State counter: {u_t}")
    print(f"Length of prediction history: {m}")
    print("=================================")
    print(f"Current length of inference queue: {q_t}")
    print(f"Max length of inference queue: {psi_q}")
    print("=================================")
    print(f"Total abnormal prediction in history: {sigma_M_t}")
    print(f"Threshold for normal predictions: {phi_g}")
    print(f"Threshold for abnormal predictions: {psi_g}")
    print("=================================")
    print(f"Low battery detected: {low_battery}")
    print("=================================")

    if q_t >= psi_q:  #  inference queue is full => cloud
        return CLOUD_INFERENCE_LAYER
    else:  # q_t < psi_q
        if u_t < m:  # => M_t is not full
            return GATEWAY_INFERENCE_LAYER
        else:  # u_t >= m => M_t is full
            if sigma_M_t < phi_g:
                if low_battery:  # b_t < phi_b
                    return GATEWAY_INFERENCE_LAYER
                else:  # b_t >= phi_b
                    return SENSOR_INFERENCE_LAYER
            elif phi_g <= sigma_M_t and sigma_M_t < psi_g:
                return GATEWAY_INFERENCE_LAYER
            else:  # sigma_M_t >= psi_g
                return CLOUD_INFERENCE_LAYER
