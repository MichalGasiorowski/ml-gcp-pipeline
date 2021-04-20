import logging
import time
from typing import List, Dict, Any, Tuple, Optional

import grpc
from grpc import RpcError
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

import client.client_util as client_util


class ProdClient:
    def __init__(self, host: str, model_name: str, model_version: Optional[int] = None,
                 in_tensor_name: str = 'examples', signature_name: str = 'serving_default',
                 options: List[Tuple[str, Any]] = None):

        self.logger = logging.getLogger(self.__class__.__name__)

        self.host = host
        self.model_name = model_name
        self.model_version = model_version
        self.in_tensor_name = in_tensor_name
        self.signature_name = signature_name
        self.options = options

    def predict(self, request_data: List[Dict[str, Any]], request_timeout: int = 10):
        t = time.time()

        with grpc.insecure_channel(self.host, options=self.options) as channel:

            self.logger.debug('Establishing insecure channel took: {}'.format(time.time() - t))

            t = time.time()
            stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

            self.logger.debug('Creating stub took: {}'.format(time.time() - t))

            t = time.time()
            request = predict_pb2.PredictRequest()

            self.logger.debug('Creating request object took: {}'.format(time.time() - t))

            request.model_spec.name = self.model_name

            if self.model_version is not None and self.model_version > 0:
                request.model_spec.version.value = self.model_version
            request.model_spec.signature_name = self.signature_name

            t = time.time()

            tensor_proto = client_util.make_tensor_proto_for_request(request_data)

            request.inputs[self.in_tensor_name].CopyFrom(tensor_proto)

            self.logger.debug('Making tensor protos took: {}'.format(time.time() - t))

            try:
                t = time.time()
                predict_response = stub.Predict(request, timeout=request_timeout)

                self.logger.debug('Actual request took: {} seconds'.format(time.time() - t))

                predict_response_dict = client_util.predict_response_to_dict(predict_response)

                keys = [k for k in predict_response_dict]
                self.logger.info('Got predict_response with keys: {}'.format(keys))

                return predict_response_dict

            except RpcError as e:
                self.logger.error(e)
                self.logger.error('Prediction failed!')
