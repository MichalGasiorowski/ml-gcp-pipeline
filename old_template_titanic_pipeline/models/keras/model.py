# Lint as: python2, python3
# Copyright 2020 Google LLC. All Rights Reserved.
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
"""TFX template taxi model.

A DNN keras model which uses features defined in features.py and network
parameters defined in constants.py.
"""

from __future__ import division
from __future__ import print_function

import os
from absl import logging
import absl
import tensorflow as tf
import tensorflow_transform as tft

from models import features
from models.keras import constants
from tfx_bsl.tfxio import dataset_options

import kerastuner
from tfx.components.trainer.executor import TrainerFnArgs
from tfx.components.trainer.fn_args_utils import FnArgs
from tfx.components.tuner.component import TunerFnResult

import ast 


def _get_serve_tf_examples_fn(model, tf_transform_output):
  """Returns a function that parses a serialized tf.Example and applies TFT."""

  model.tft_layer = tf_transform_output.transform_features_layer()

  @tf.function
  def serve_tf_examples_fn(serialized_tf_examples):
    """Returns the output to be used in the serving signature."""
    feature_spec = tf_transform_output.raw_feature_spec()
    feature_spec.pop(features.LABEL_KEY)
    parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)

    transformed_features = model.tft_layer(parsed_features)

    return model(transformed_features)

  return serve_tf_examples_fn


def _input_fn(file_pattern, data_accessor, tf_transform_output, batch_size=200):
  """Generates features and label for tuning/training.

  Args:
    file_pattern: List of paths or patterns of input tfrecord files.
    data_accessor: DataAccessor for converting input to RecordBatch.
    tf_transform_output: A TFTransformOutput.
    batch_size: representing the number of consecutive elements of returned
      dataset to combine in a single batch

  Returns:
    A dataset that contains (features, indices) tuple where features is a
      dictionary of Tensors, and indices is a single Tensor of label indices.
  """
  return data_accessor.tf_dataset_factory(
      file_pattern,
      dataset_options.TensorFlowDatasetOptions(
          batch_size=batch_size,
          label_key=features.transformed_name(features.LABEL_KEY)),
      tf_transform_output.transformed_metadata.schema)

def _get_hyperparameters() -> kerastuner.HyperParameters:
  """Returns hyperparameters for building Keras model."""
  hp = kerastuner.HyperParameters()
  # Defines search space.
  hp.Choice('learning_rate', [1e-2, 3e-3, 1e-3], default=constants.LEARNING_RATE)
  hp.Choice('hidden_units', ["[16, 8]", "[32, 16]", "[16]"], default=constants.HIDDEN_UNITS)
  #hp.Float('dropout_rate', 0.1, 0.5, default=0.2)
  return hp


#def _build_keras_model(hidden_units, learning_rate):
def _build_keras_model(hparams: kerastuner.HyperParameters = None) -> tf.keras.Model:
  """Creates a DNN Keras model for classifying taxi data.

  Args:
    hidden_units: [int], the layer sizes of the DNN (input layer first).
    learning_rate: [float], learning rate of the Adam optimizer.

  Returns:
    A keras Model.
  """
  real_keys =  features.DENSE_FLOAT_FEATURE_KEYS
  sparse_keys = features.VOCAB_FEATURE_KEYS + features.BUCKET_FEATURE_KEYS + features.CATEGORICAL_FEATURE_KEYS

  real_valued_columns = [
      tf.feature_column.numeric_column(key, shape=())
      for key in features.transformed_names(features.DENSE_FLOAT_FEATURE_KEYS)
  ]
  categorical_columns = [
      tf.feature_column.categorical_column_with_identity(  # pylint: disable=g-complex-comprehension
          key,
          num_buckets=features.VOCAB_SIZE + features.OOV_SIZE,
          default_value=0)
      for key in features.transformed_names(features.VOCAB_FEATURE_KEYS)
  ]
  categorical_columns += [
      tf.feature_column.categorical_column_with_identity(  # pylint: disable=g-complex-comprehension
          key,
          num_buckets=num_buckets,
          default_value=0) for key, num_buckets in zip(
              features.transformed_names(features.BUCKET_FEATURE_KEYS),
              features.BUCKET_FEATURE_BUCKET_COUNT)
  ]
  categorical_columns += [
      tf.feature_column.categorical_column_with_identity(  # pylint: disable=g-complex-comprehension
          key,
          num_buckets=num_buckets,
          default_value=0) for key, num_buckets in zip(
              features.transformed_names(features.CATEGORICAL_FEATURE_KEYS),
              features.CATEGORICAL_FEATURE_MAX_VALUES)
  ]
  indicator_column = [
      tf.feature_column.indicator_column(categorical_column)
      for categorical_column in categorical_columns
  ]

  learning_rate = float(hparams.get('learning_rate'))
  hidden_units = ast.literal_eval(hparams.get('hidden_units'))
    
  #hidden_units= hparams.get('hidden_units')

  model = _wide_and_deep_classifier(
      # TODO(b/140320729) Replace with premade wide_and_deep keras model
      wide_columns=indicator_column,
      deep_columns=real_valued_columns,
      hidden_units=hidden_units,
      learning_rate=learning_rate)
  return model


def _wide_and_deep_classifier(wide_columns, deep_columns, hidden_units,
                              learning_rate):
  """Build a simple keras wide and deep model.

  Args:
    wide_columns: Feature columns wrapped in indicator_column for wide (linear)
      part of the model.
    deep_columns: Feature columns for deep part of the model.
    dnn_hidden_units: [int], the layer sizes of the hidden DNN.
    learning_rate: [float], learning rate of the Adam optimizer.

  Returns:
    A Wide and Deep Keras model
  """
  # Keras needs the feature definitions at compile time.
  # TODO(b/139081439): Automate generation of input layers from FeatureColumn.
  input_layers = {
      colname: tf.keras.layers.Input(name=colname, shape=(), dtype=tf.float32)
      for colname in features.transformed_names(
          features.DENSE_FLOAT_FEATURE_KEYS)
  }
  input_layers.update({
      colname: tf.keras.layers.Input(name=colname, shape=(), dtype='int32')
      for colname in features.transformed_names(features.VOCAB_FEATURE_KEYS)
  })
  input_layers.update({
      colname: tf.keras.layers.Input(name=colname, shape=(), dtype='int32')
      for colname in features.transformed_names(features.BUCKET_FEATURE_KEYS)
  })
  input_layers.update({
      colname: tf.keras.layers.Input(name=colname, shape=(), dtype='int32') for
      colname in features.transformed_names(features.CATEGORICAL_FEATURE_KEYS)
  })

  # TODO(b/161952382): Replace with Keras premade models and
  # Keras preprocessing layers.
  deep = tf.keras.layers.DenseFeatures(deep_columns)(input_layers)
  for numnodes in hidden_units:
    deep = tf.keras.layers.Dense(numnodes)(deep)
  wide = tf.keras.layers.DenseFeatures(wide_columns)(input_layers)

  output = tf.keras.layers.Dense(
      1, activation='sigmoid')(
          tf.keras.layers.concatenate([deep, wide]))
  output = tf.squeeze(output, -1)

  model = tf.keras.Model(input_layers, output)

  model.compile(
      loss='binary_crossentropy',
      optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
      metrics=['binary_accuracy'])
      #metrics=[tf.keras.metrics.BinaryAccuracy(), 'accuracy'])
  model.summary(print_fn=logging.info)
  return model


# TFX Trainer will call this function.
def run_fn(fn_args):
  """Train the model based on given args.

  Args:
    fn_args: Holds args used to train the model as name/value pairs.
  """

  if fn_args.hyperparameters:
    hparams = kerastuner.HyperParameters.from_config(fn_args.hyperparameters)
  else:
    # This is a shown case when hyperparameters is decided and Tuner is removed
    # from the pipeline. User can also inline the hyperparameters directly in
    # _build_keras_model.
    hparams = _get_hyperparameters()
  absl.logging.info('HyperParameters for training: %s' % hparams.get_config())

  tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

  train_dataset = _input_fn(fn_args.train_files, fn_args.data_accessor,
                            tf_transform_output, constants.TRAIN_BATCH_SIZE)
  eval_dataset = _input_fn(fn_args.eval_files, fn_args.data_accessor,
                           tf_transform_output, constants.EVAL_BATCH_SIZE)

  mirrored_strategy = tf.distribute.MirroredStrategy()
  with mirrored_strategy.scope():
    model = _build_keras_model(hparams)
    #model = _build_keras_model(hidden_units=constants.HIDDEN_UNITS, learning_rate=constants.LEARNING_RATE)

  # Write logs to path
  log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), 'logs')
  #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=fn_args.model_run_dir, update_freq='batch')
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq='batch')
  early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

  model.fit(
      train_dataset,
      steps_per_epoch=fn_args.train_steps,
      validation_data=eval_dataset,
      validation_steps=fn_args.eval_steps,
      callbacks=[tensorboard_callback, early_stopping_callback])

  signatures = {
      'serving_default':
          _get_serve_tf_examples_fn(model,
                                    tf_transform_output).get_concrete_function(
                                        tf.TensorSpec(
                                            shape=[None],
                                            dtype=tf.string,
                                            name='examples')),
  }
  model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)


# TFX Tuner will call this function.
def tuner_fn(fn_args: FnArgs) -> TunerFnResult:
  """Build the tuner using the KerasTuner API.
  Args:
    fn_args: Holds args as name/value pairs.
      - working_dir: working dir for tuning.
      - train_files: List of file paths containing training tf.Example data.
      - eval_files: List of file paths containing eval tf.Example data.
      - train_steps: number of train steps.
      - eval_steps: number of eval steps.
      - schema_path: optional schema of the input data.
      - transform_graph_path: optional transform graph produced by TFT.
  Returns:
    A namedtuple contains the following:
      - tuner: A BaseTuner that will be used for tuning.
      - fit_kwargs: Args to pass to tuner's run_trial function for fitting the
                    model , e.g., the training and validation dataset. Required
                    args depend on the above tuner's implementation.
  """
  # RandomSearch is a subclass of kerastuner.Tuner which inherits from
  # BaseTuner.
  tuner = kerastuner.RandomSearch(
      _build_keras_model,
      max_trials=10,
      hyperparameters=_get_hyperparameters(),
      allow_new_entries=False,
      objective=kerastuner.Objective('val_binary_accuracy', 'max'),
      directory=fn_args.working_dir,
      project_name='titanic_tuning')
  #sparse_categorical_accuracy
  transform_graph = tft.TFTransformOutput(fn_args.transform_graph_path)

  train_dataset = _input_fn(
      fn_args.train_files,
      fn_args.data_accessor,
      transform_graph,
      batch_size=constants.TRAIN_BATCH_SIZE)
  eval_dataset = _input_fn(
      fn_args.eval_files,
      fn_args.data_accessor,
      transform_graph,
      batch_size=constants.EVAL_BATCH_SIZE)

  return TunerFnResult(
      tuner=tuner,
      fit_kwargs={
          'x': train_dataset,
          'validation_data': eval_dataset,
          'steps_per_epoch': fn_args.train_steps,
          'validation_steps': fn_args.eval_steps
      })