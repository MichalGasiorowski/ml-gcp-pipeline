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
"""Covertype preprocessing.
This file defines a template for TFX Transform component.
"""

import tensorflow as tf
import tensorflow_transform as tft
import absl

import features

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
  return tf.squeeze(
      tf.sparse.to_dense(
          tf.SparseTensor(x.indices, x.values, [x.dense_shape[0], 1]),
          default_value),
      axis=1)

def preprocessing_fn(inputs):
  """Preprocesses Covertype Dataset."""

  outputs = {}

  # Scale numerical features
  for key in features.NUMERIC_FEATURE_KEYS:
    outputs[features.transformed_name(key)] = tft.scale_to_z_score(
        _fill_in_missing(inputs[key]))

  for key in features.VOCAB_FEATURE_KEYS:
    # Build a vocabulary for this feature.
    outputs[features.transformed_name(key)] = tft.compute_and_apply_vocabulary(
        _fill_in_missing(inputs[key]),
        top_k=features.VOCAB_SIZE_MAP.get(key, features.VOCAB_SIZE),
        num_oov_buckets=features.OOV_SIZE)

  for key in features.BUCKET_FEATURE_KEYS:
    if key in features.FEATURE_BUCKET_BOUNDARIES:
      bucket_boundaries = tf.constant(features.FEATURE_BUCKET_BOUNDARIES.get(key))
      #tf.print("bucket_boundaries:", bucket_boundaries, output_stream=absl.logging.info)
      outputs[features.transformed_name(key)] = tft.apply_buckets(_fill_in_missing(inputs[key]),
                                                          bucket_boundaries)
    else:
      outputs[features.transformed_name(key)] = tft.bucketize(
        _fill_in_missing(inputs[key]), features.FEATURE_BUCKET_COUNT_MAP.get(key, features.FEATURE_BUCKET_COUNT))

  # Generate vocabularies and maps categorical features
  for key in features.CATEGORICAL_FEATURE_KEYS:
    outputs[features.transformed_name(key)] = tft.compute_and_apply_vocabulary(
        x=_fill_in_missing(inputs[key]), num_oov_buckets=1, vocab_filename=key)

  # Convert Cover_Type to dense tensor
  outputs[features.transformed_name(features.LABEL_KEY)] = _fill_in_missing(
      inputs[features.LABEL_KEY])

  return outputs

