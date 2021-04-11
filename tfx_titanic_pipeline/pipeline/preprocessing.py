# Copyright 2021 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Titanic preprocessing.
This file defines a template for TFX Transform component.
"""
from __future__ import absolute_import

import tensorflow as tf
import tensorflow_transform as tft
import absl

import features as features


def _fill_in_missing(x):
    """Replace missing values in a SparseTensor.
    Fills in missing values of `x` with '' or 0, and converts to a dense tensor.
    Args:
      x: A `SparseTensor` of rank 2.  Its dense shape should have size at most 1
        in the second dimension.
    Returns:
      A rank 1 tensor where missing values of `x` have been filled in.
    """
    default_value = '' if x.dtype == tf.string else 0
    return tf.squeeze(_impute(x, default_value), axis=1)


def _fill_in_missing_with_impute(x, imputed_value):
    """Replace missing values in a SparseTensor with imputed_value.
    Fills in missing values of `x` with '' or 0, and converts to a dense tensor.
    Args:
      x: A `SparseTensor` of rank 2.  Its dense shape should have size at most 1
        in the second dimension.
      imputed_value: value for which missing values
    Returns:
      A rank 1 tensor where missing values of `x` have been filled in.
    """
    return tf.squeeze(_impute_numerical(x, imputed_value), axis=1)


def _impute_numerical(tensor, value):
    tensor = tf.sparse.SparseTensor(
        tensor.indices, tensor.values, [tensor.dense_shape[0], 1]
    )
    dense = tf.sparse.to_dense(sp_input=tensor, default_value=value)
    return tf.where(tf.math.is_nan(dense), value, dense)


def _impute(tensor, value):
    tensor = tf.sparse.SparseTensor(
        tensor.indices, tensor.values, [tensor.dense_shape[0], 1]
    )
    dense = tf.sparse.to_dense(sp_input=tensor, default_value=value)
    return dense


def compute_mean_ignore_nan(values):
    finite_indices = tf.math.is_finite(values)
    finite_values = tf.boolean_mask(values, finite_indices)
    mean_value = tft.mean(finite_values)
    return mean_value


def preprocessing_fn(inputs):
    """Preprocesses Titanic Dataset."""

    outputs = {}

    # Scale numerical features
    for key in features.NUMERIC_FEATURE_KEYS:
        mean_value = compute_mean_ignore_nan(inputs[key].values)
        absl.logging.info(f'TFT preprocessing. Mean value for {key} = {mean_value}')
        outputs[features.transformed_name(key)] = tft.scale_to_z_score(
            _fill_in_missing_with_impute(inputs[key], mean_value))

    for key in features.VOCAB_FEATURE_KEYS:
        # Build a vocabulary for this feature.
        outputs[features.transformed_name(key)] = tft.compute_and_apply_vocabulary(
            _fill_in_missing(inputs[key]),
            top_k=features.VOCAB_SIZE_MAP.get(key, features.VOCAB_SIZE),
            num_oov_buckets=features.OOV_SIZE)

    for key in features.BUCKET_FEATURE_KEYS:
        if key in features.FEATURE_BUCKET_BOUNDARIES:
            bucket_boundaries = tf.constant(features.FEATURE_BUCKET_BOUNDARIES.get(key))
            # tf.print("bucket_boundaries:", bucket_boundaries, output_stream=absl.logging.info)
            outputs[features.transformed_name(key)] = tft.apply_buckets(_fill_in_missing(inputs[key]),
                                                                        bucket_boundaries)
        else:
            outputs[features.transformed_name(key)] = tft.bucketize(
                _fill_in_missing(inputs[key]),
                features.FEATURE_BUCKET_COUNT_MAP.get(key, features.FEATURE_BUCKET_COUNT))

    # Generate vocabularies and maps categorical features
    for key in features.CATEGORICAL_FEATURE_KEYS:
        outputs[features.transformed_name(key)] = tft.compute_and_apply_vocabulary(
            x=_fill_in_missing(inputs[key]), num_oov_buckets=1, vocab_filename=key)

    # Convert Cover_Type to dense tensor
    outputs[features.transformed_name(features.LABEL_KEY)] = _fill_in_missing(
        inputs[features.LABEL_KEY])

    return outputs
