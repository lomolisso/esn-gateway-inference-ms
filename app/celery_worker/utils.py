import asyncio
import json
import aioredis

from app.config import ESN_REDIS_URL

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


def prediction_command(device_name, payload):
    event_loop = asyncio.get_event_loop()
    event_loop.run_until_complete(_redis_enqueue_command(
            device_name=device_name,
            command="debug-prediction-command",
            params=payload,
        )
    )

def check_prediction(*args, **kwargs):
    pass
