import asyncio
import json
import aioredis
import requests

from app.config import ESN_REDIS_URL, IS_GATEWAY, ESN_CLOUD_APP_BACKEND_URL

def _post_json_to_cloud_app_backend(endpoint, payload):
    response = requests.post(
        f"{ESN_CLOUD_APP_BACKEND_URL}/{endpoint}",
        json=payload,
    )
    if response.status_code != 200:
        raise Exception(f"Error: {response.status_code} - {response.text}")



async def _redis_enqueue_command(device_name, command, params=None):
    redis_client = await aioredis.create_redis_pool(ESN_REDIS_URL)
    queue_key = f"commands:queue:{device_name}"
    await redis_client.rpush(
        queue_key,
        json.dumps(
            {
                "command": command,
                "device_name": device_name,
                "params": params,
            }
        ),
    )


def prediction_command(device_name, gateway_name, payload):
    if IS_GATEWAY:
        event_loop = asyncio.get_event_loop()
        event_loop.run_until_complete(_redis_enqueue_command(
                device_name=device_name,
                command="debug-prediction-command",
                params=payload,
            )
        )
    else: # IS_CLOUD
        _post_json_to_cloud_app_backend(
            endpoint=f"gateway/{gateway_name}/device/{device_name}/debug/prediction-command",
            payload=payload,
        )
        

def check_prediction(*args, **kwargs):
    pass
