import os
import base64
import numpy as np
import tflite_runtime.interpreter as tf

# --- TensorFlow Lite handler ---

class TFLiteHandler(object):
    def load_model(self, b64_encoded_model):
        # Decode b64 encoded model into bytes
        _decoded_model = base64.b64decode(b64_encoded_model)

        # Load the model
        self.model = tf.Interpreter(model_content=_decoded_model)
        self.model.allocate_tensors()

        # Get input and output tensors.
        self._input_details = self.model.get_input_details()
        self._output_details = self.model.get_output_details()

    def predict(self, input_data):
        # Check if the input data type matches the model's input type
        input_data = self._preprocess_input(input_data)

        # Set the value for the model's input
        self.model.set_tensor(self._input_details[0]['index'], [input_data])

        # Run the model
        self.model.invoke()

        # Extract the output data from the tensor
        output_data = self.model.get_tensor(self._output_details[0]['index'])

        return self._postprocess_output(output_data[0])
        
    def _preprocess_input(self, input_data):
        # Convert input_data to a numpy array and reshape it
        input_data = np.array([input_data], dtype=np.float32)

        # Multiply the input by 2pi
        input_data *= 2 * np.pi

        # Check if the model requires quantized input
        if self._input_details[0]['dtype'] == np.int8:
            # Quantize the input
            scale, zero_point = self._input_details[0]['quantization']
            input_data = input_data / scale + zero_point
            input_data = np.clip(input_data, -128, 127)
            return np.array(input_data, dtype=np.int8)

        return input_data

    def _postprocess_output(self, output_data):
        # Check if the model requires quantized output
        if self._output_details[0]['dtype'] == np.int8:
            # Dequantize the output
            scale, zero_point = self._output_details[0]['quantization']
            output_data = output_data.astype(np.float32)
            output_data = (output_data - zero_point) * scale
            return output_data

        return output_data
