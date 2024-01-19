import numpy as np
import tensorflow as tf
import base64
import tempfile


# --- TensorFlow using Keras handler ---

class TFKerasHandler(object):
    def load_model(self, b64_encoded_model):
        # Decode b64 encoded model into bytes
        _decoded_model = base64.b64decode(b64_encoded_model)
        
        # Create a temporary file and write the model to it,
        # note that it should be erased after the model is loaded
        with tempfile.NamedTemporaryFile() as temp:
            temp.write(_decoded_model)
            temp.flush()
            
            # Load the model
            self.model = tf.keras.models.load_model(temp.name)
        
    def predict(self, input_data):
        # Preprocess the input
        input_data = self._preprocess_input(input_data)

        # Run the model and get the output
        output_data = self.model.predict(np.array([input_data]))

        # Postprocess the output
        return self._postprocess_output(output_data[0])

    def _preprocess_input(self, input_data):
        # Convert input_data to a numpy array and reshape it
        input_data = np.array(input_data, dtype=np.float32)

        # Multiply the input by 2pi
        input_data *= 2 * np.pi

        return input_data

    def _postprocess_output(self, output_data):
        # The post-processing step might change depending on the model and application
        # For now, just returning the output data directly
        return output_data
