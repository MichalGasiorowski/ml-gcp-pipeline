import logging
import tensorflow as tf
import pandas as pd
import numpy as np

from tensorflow.core.framework import types_pb2
from tensorflow.core.framework import tensor_pb2
from tensorflow.core.framework import tensor_shape_pb2

logger = logging.getLogger(__name__)

# Should be all values from protos/types.proto
dtype_to_number = {
    'DT_INVALID': 0,
    'DT_FLOAT': 1,
    'DT_DOUBLE': 2,
    'DT_INT32': 3,
    'DT_UINT8': 4,
    'DT_INT16': 5,
    'DT_INT8': 6,
    'DT_STRING': 7,
    'DT_COMPLEX64': 8,
    'DT_INT64': 9,
    'DT_BOOL': 10,
    'DT_QINT8': 11,
    'DT_QUINT8': 12,
    'DT_QINT32': 13,
    'DT_BFLOAT16': 14,
    'DT_QINT16': 15,
    'DT_QUINT16': 16,
    'DT_UINT16': 17,
    'DT_COMPLEX128': 18,
    'DT_HALF': 19,
    'DT_RESOURCE': 20
}

number_to_dtype_value = {
    1: 'float_val',
    2: 'double_val',
    3: 'int_val',
    4: 'int_val',
    5: 'int_val',
    6: 'int_val',
    7: 'string_val',
    8: 'scomplex_val',
    9: 'int64_val',
    10: 'bool_val',
    18: 'dcomplex_val',
    19: 'half_val',
    20: 'resource_handle_val'
}


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    if isinstance(value, str):
        value = str.encode(value)  # str wont work, we need bytes
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


feature_titanic_column_2_tf_example_mapping = {
    'Embarked': _bytes_feature,
    'Ticket': _bytes_feature,
    'Sex': _bytes_feature,
    'Name': _bytes_feature,
    'Cabin': _bytes_feature,
    'Age': _float_feature,
    'Fare': _float_feature,
    'Parch': _int64_feature,
    'PassengerId': _int64_feature,
    'Pclass': _int64_feature,
    'SibSp': _int64_feature
}

LABEL_KEY = 'Survived'


def serialize_example(data, feature_2_tf_example_mapping=feature_titanic_column_2_tf_example_mapping):
    """
    Creates a serialized tf.train.Example message ready to be written to a file.
    data : dict
              dictionary with data in key: value format
    """
    if isinstance(data, pd.core.frame.DataFrame):
        data = data.to_dict(orient='records')

    # Create a dictionary mapping the feature name to the tf.train.Example-compatible
    # data type.
    feature = {key: feature_2_tf_example_mapping[key](data[key]) for key in data.keys()}

    # Create a Features message using tf.train.Example.

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def make_tensor_proto(data, dtype, size) -> tensor_pb2.TensorProto:
    dims = [tensor_shape_pb2.TensorShapeProto.Dim(size=size)]
    tensor_shape_proto = tensor_shape_pb2.TensorShapeProto(dim=dims)
    tensor_proto = tensor_pb2.TensorProto(
        dtype=dtype,
        tensor_shape=tensor_shape_proto,
        string_val=data)
    return tensor_proto


def make_tensor_proto_for_request(request_data) -> tensor_pb2.TensorProto:
    serialized_examples_array = [serialize_example(row) for row in request_data]  # array od serialized examples
    size = len(request_data)

    return make_tensor_proto(data=serialized_examples_array, dtype=types_pb2.DT_STRING, size=size)


def parse_prediction_result(prediction_result):
    outputs_tensor_proto = prediction_result.outputs["output_0"]
    shape = tf.TensorShape(outputs_tensor_proto.tensor_shape)
    outputs = np.array(outputs_tensor_proto.float_val).reshape(shape)
    return outputs


converters = {'Cabin': str, 'Name': str, 'Ticket': str, 'Sex': str, 'Embarked': str}


def read_csv_2_dict(filepath, type_converters=converters):
    return pd.read_csv(filepath, converters=converters).to_dict(orient='records')


def read_csv_for_prediction(filepath, type_converters=converters, label_key=LABEL_KEY):
    df = pd.read_csv(filepath, converters=converters)
    if label_key in df.columns:
        df.drop([label_key], axis=1, inplace=True)
    return df


def predict_response_to_dict(predict_response):
    predict_response_dict = dict()

    for k in predict_response.outputs:
        shape = [x.size for x in predict_response.outputs[k].tensor_shape.dim]

        logger.debug('Key: ' + k + ', shape: ' + str(shape))

        dtype_constant = predict_response.outputs[k].dtype

        if dtype_constant not in number_to_dtype_value:
            logger.error('Tensor output data type not supported. Returning empty dict.')
            predict_response_dict[k] = 'value not found'

        if shape == [1]:
            predict_response_dict[k] = eval('predict_response.outputs[k].' + number_to_dtype_value[dtype_constant])[0]
        else:
            predict_response_dict[k] = np.array(
                eval('predict_response.outputs[k].' + number_to_dtype_value[dtype_constant])).reshape(shape)

    return predict_response_dict
