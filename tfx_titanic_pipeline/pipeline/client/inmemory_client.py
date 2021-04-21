import numpy as np
import tensorflow as tf
from tensorflow import keras

import logging
import time

import client.client_util as client_util


# tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY

class InMemoryClient:
    def __init__(self, model_path, in_tensor_name='examples', signature_name='serving_default'):

        self.logger = logging.getLogger(self.__class__.__name__)
        self.model_path = model_path
        self.in_tensor_name = in_tensor_name
        self.signature_name = signature_name

        try:
            self.model = tf.saved_model.load(model_path)

            self.logger.info(list(self.model.signatures.keys()))  # ["serving_default"]
            self.infer = self.model.signatures[self.signature_name]
            self.logger.info('structured_input_signature:', self.infer.structured_input_signature)
            self.logger.info('structured_outputs:', self.infer.structured_outputs)
        except IOError:
            self.logger.error("Invalid model path")
            raise IOError("Invalid model path.")

    def predict(self, request_data, **kwargs):

        self.logger.info('Sending request to inmemory model')
        self.logger.info('Model path: ' + str(self.model_path))

        tensor_proto = client_util.make_tensor_proto_for_request(request_data)
        tensor = tf.constant(tf.make_ndarray(tensor_proto), name=self.in_tensor_name)

        t = time.time()
        predict_response = self.infer(tensor)

        self.logger.info('Actual request took: {} seconds'.format(time.time() - t))

        predict_response_dict = predict_response

        keys = [k for k in predict_response_dict]
        self.logger.info('Got predict_response with keys: {}'.format(keys))

        return predict_response_dict
