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
"""Titanic model features."""

from __future__ import absolute_import

import tensorflow as tf
from typing import Text, List

DROPPPED_FEATURES = [
  'PassengerId', 'Name'
]

# At least one feature is needed.
#DENSE_FLOAT_FEATURE_KEYS = ['Age', 'Fare']
NUMERIC_FEATURE_KEYS = ['Age', 'Fare']

# Name of features which have continuous float values. These features will be
# bucketized using `tft.bucketize`, and will be used as categorical features.
BUCKET_FEATURE_KEYS = ['Parch', 'SibSp']
# Number of buckets used by tf.transform for encoding each feature. The length
# of this list should be the same with BUCKET_FEATURE_KEYS.
BUCKET_FEATURE_BUCKET_COUNT = [10, 10]


# Number of buckets used by tf.transform for encoding each feature.
FEATURE_BUCKET_COUNT = 10
FEATURE_BUCKET_COUNT_MAP = {'Parch': 3, 'SibSp': 3}
FEATURE_BUCKET_BOUNDARIES = {'Parch': [[0.0, 1.0, 2.0]], 'SibSp': [[0.0, 1.0, 2.0]]}

# Count of out-of-vocab buckets in which unrecognized VOCAB_FEATURES are hashed.
OOV_SIZE = 10


CATEGORICAL_FEATURE_KEYS = [ #'Embarked', 'Pclass', 'Sex'
]
CATEGORICAL_FEATURE_MAX_VALUES = [#10, 10, 10
]

# Name of features which have string values and are mapped to integers.
VOCAB_FEATURE_KEYS = ['Embarked', 'Pclass', 'Sex']

# Number of vocabulary terms used for encoding VOCAB_FEATURES by tf.transform
VOCAB_SIZE = 1000
VOCAB_SIZE_MAP = {'Embarked': 3, 'Pclass': 3, 'Sex':2}

# Keys
LABEL_KEY = 'Survived'
FARE_KEY = 'Fare'
CABIN_KEY = 'Cabin'
PARCH_KEY = 'Parch'
SIBSP_KEY = 'SibSp'
#NUM_CLASSES = 2


#_FEATURES = {
#    'culmen_length_mm': tf.io.FixedLenFeature([], dtype=tf.float32),
#    'culmen_depth_mm': tf.io.FixedLenFeature([], dtype=tf.float32),
#    'flipper_length_mm': tf.io.FixedLenFeature([], dtype=tf.float32),
#    'body_mass_g': tf.io.FixedLenFeature([], dtype=tf.float32),
#    'species': tf.io.FixedLenFeature([], dtype=tf.int64)
#}


# taken from schema.pbtxt :
#name: "Embarked"    BYTES
#name: "Ticket"      BYTES
#name: "Sex"         BYTES
#name: "Name"        BYTES
#name: "Cabin"       BYTES
#name: "Age"         FLOAT
#name: "Fare"        FLOAT
#name: "Parch"       INT
#name: "PassengerId" INT
#name: "Pclass"      INT
#name: "SibSp"       INT
#name: "Survived"    INT


RAW_DATA_FEATURE_SPEC = {
  'Embarked': tf.io.FixedLenFeature([], dtype=tf.string),
  'Ticket': tf.io.FixedLenFeature([], dtype=tf.string),
  'Sex': tf.io.FixedLenFeature([], dtype=tf.string),
  'Name': tf.io.FixedLenFeature([], dtype=tf.string),
  'Cabin': tf.io.FixedLenFeature([], dtype=tf.string),
  'Age': tf.io.FixedLenFeature([], dtype=tf.float32),
  'Fare': tf.io.FixedLenFeature([], dtype=tf.float32),
  'Parch': tf.io.FixedLenFeature([], dtype=tf.int64),
  'PassengerId': tf.io.FixedLenFeature([], dtype=tf.int64),
  'Pclass': tf.io.FixedLenFeature([], dtype=tf.int64),
  'SibSp': tf.io.FixedLenFeature([], dtype=tf.int64),
  'Survived': tf.io.FixedLenFeature([], dtype=tf.int64),
}

#RAW_DATA_FEATURE_SPEC = dict(
#    [(name, tf.io.FixedLenFeature([], tf.string))
#     for name in CATEGORICAL_FEATURE_KEYS] +
#    [(name, tf.io.FixedLenFeature([], tf.float32))
#     for name in NUMERIC_FEATURE_KEYS] +
#    [(name, tf.io.VarLenFeature(tf.float32))
#     for name in OPTIONAL_NUMERIC_FEATURE_KEYS] +
#    [(LABEL_KEY, tf.io.FixedLenFeature([], tf.string))]
#)


def transformed_name(key: Text) -> Text:
  """Generate the name of the transformed feature from original name."""
  return key + '_xf'


def vocabulary_name(key: Text) -> Text:
  """Generate the name of the vocabulary feature from original name."""
  return key + '_vocab'


def transformed_names(keys: List[Text]) -> List[Text]:
  """Transform multiple feature names at once."""
  return [transformed_name(key) for key in keys]


