from app.config import USE_TFLITE

class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

class PredictiveModelHandler(metaclass=SingletonMeta):
    _handler = None

    def load_model(self, b64_encoded_model):
        if USE_TFLITE:
            from app.celery_worker.tensorflow.lite.handler import TFLiteHandler
            self._handler = TFLiteHandler()
        else:
            from app.celery_worker.tensorflow.keras.handler import TFKerasHandler
            self._handler = TFKerasHandler()
        self._handler.load_model(b64_encoded_model)
    

    def predict(self, input_data):
        return self._handler.predict(input_data)
