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
"""TFX taxi template pipeline definition.

This file defines TFX pipeline and various components in the pipeline.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Dict, List, Optional, Text

from . import configs
import tensorflow_model_analysis as tfma
from tfx.components import CsvExampleGen
from tfx.components import Evaluator
from tfx.components import ExampleValidator
from tfx.components import Pusher
from tfx.components import ResolverNode
from tfx.components import SchemaGen
from tfx.components import StatisticsGen
from tfx.components import Trainer
from tfx.components import Transform
from tfx.components import Tuner
from tfx.components.trainer import executor as trainer_executor
from tfx.dsl.components.base import executor_spec
from tfx.dsl.experimental import latest_blessed_model_resolver
from tfx.extensions.google_cloud_ai_platform.pusher import executor as ai_platform_pusher_executor
from tfx.extensions.google_cloud_ai_platform.trainer import executor as ai_platform_trainer_executor
from tfx.extensions.google_cloud_big_query.example_gen import component as big_query_example_gen_component  # pylint: disable=unused-import
from tfx.orchestration import pipeline
from tfx.proto import example_gen_pb2
from tfx.proto import pusher_pb2
from tfx.proto import trainer_pb2
from tfx.utils.dsl_utils import external_input
from tfx.types import Channel
from tfx.types.standard_artifacts import Model
from tfx.types.standard_artifacts import ModelBlessing


from ml_metadata.proto import metadata_store_pb2


def create_pipeline(
    pipeline_name: Text,
    pipeline_root: Text,
    data_path: Text,
    # TODO(step 7): (Optional) Uncomment here to use BigQuery as a data source.
    # query: Text,
    preprocessing_fn: Text,
    run_fn: Text,
    tuner_fn: Text,
    train_args: trainer_pb2.TrainArgs,
    eval_args: trainer_pb2.EvalArgs,
    tuner_train_args: trainer_pb2.TrainArgs,
    tuner_eval_args: trainer_pb2.EvalArgs, 
    eval_accuracy_threshold: float,
    serving_model_dir: Text,
    metadata_connection_config: Optional[
        metadata_store_pb2.ConnectionConfig] = None,
    beam_pipeline_args: Optional[List[Text]] = None,
    ai_platform_training_args: Optional[Dict[Text, Text]] = None,
    ai_platform_tuner_args: Optional[Dict[Text, Text]] = None,
    ai_platform_serving_args: Optional[Dict[Text, Any]] = None,
    train_ratio_percent: float=0.8,
    enable_tuning: bool=True,
) -> pipeline.Pipeline:
  """Implements the titanic taxi pipeline with TFX."""
  
  train_ratio = int(train_ratio_percent*100)
  eval_ratio  = 100-train_ratio

  components = []

  output = example_gen_pb2.Output(
             split_config=example_gen_pb2.SplitConfig(splits=[
                 example_gen_pb2.SplitConfig.Split(name='train', hash_buckets=train_ratio),
                 example_gen_pb2.SplitConfig.Split(name='eval', hash_buckets=eval_ratio)
             ]))
  examples = external_input(data_path)
  # Brings data into the pipeline or otherwise joins/converts training data.
  #example_gen = CsvExampleGen(input=external_input(data_path))
  example_gen = CsvExampleGen(input=examples, output_config=output)
  # TODO(step 7): (Optional) Uncomment here to use BigQuery as a data source.
  # example_gen = big_query_example_gen_component.BigQueryExampleGen(
  #     query=query)
  components.append(example_gen)

  # Computes statistics over data for visualization and example validation.
  statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])
  components.append(statistics_gen)

  # Generates schema based on statistics files.
  schema_gen = SchemaGen(
      statistics=statistics_gen.outputs['statistics'],
      infer_feature_shape=True)
  components.append(schema_gen)

  # Performs anomaly detection based on statistics and data schema.
  example_validator = ExampleValidator(  # pylint: disable=unused-variable
      statistics=statistics_gen.outputs['statistics'],
      schema=schema_gen.outputs['schema'])
  components.append(example_validator)

  # Performs transformations and feature engineering in training and serving.
  transform = Transform(
      examples=example_gen.outputs['examples'],
      schema=schema_gen.outputs['schema'],
      preprocessing_fn=preprocessing_fn)
  components.append(transform)

  tuner_args = {
      'tuner_fn': tuner_fn,
      'examples': transform.outputs['transformed_examples'], # transformed_examples field is deprecated, see for example https://github.com/tensorflow/tfx/blob/master/tfx/components/trainer/component.py
      'schema': schema_gen.outputs['schema'],
      'transform_graph': transform.outputs['transform_graph'],
      'train_args': tuner_train_args,
      'eval_args': tuner_eval_args,
  }
  
  # see https://github.com/tensorflow/tfx/blob/master/docs/guide/tuner.md  
  #if ai_platform_tuner_args is not None:
  #  tuner_args.update({
  #      'custom_executor_spec':
  #          executor_spec.ExecutorClassSpec(
  #              ai_platform_trainer_executor.GenericExecutor
  #          ),
  #      'custom_config': {
  #          ai_platform_trainer_executor.TRAINING_ARGS_KEY:
  #              ai_platform_tuner_args,
  #      }
  #  })
    
  if enable_tuning:
    # Hyperparameter tuning based on the tuner_fn .
    tuner = Tuner(**tuner_args)

  trainer_args = {
      'run_fn': run_fn,
      'examples': transform.outputs['transformed_examples'],
      'schema': schema_gen.outputs['schema'],
      'transform_graph': transform.outputs['transform_graph'],
      'train_args': train_args,
      'eval_args': eval_args,
      # If Tuner is in the pipeline, Trainer can take Tuner's output
      # best_hyperparameters artifact as input and utilize it in the user module
      # code.
      #
      # If there isn't Tuner in the pipeline, either use ImporterNode to import
      # a previous Tuner's output to feed to Trainer, or directly use the tuned
      # hyperparameters in user module code and set hyperparameters to None
      # here.
      #
      # Example of ImporterNode,
      #   hparams_importer = ImporterNode(
      #     instance_name='import_hparams',
      #     source_uri='path/to/best_hyperparameters.txt',
      #     artifact_type=HyperParameters)
      #   ...
      #   hyperparameters = hparams_importer.outputs['result'],
      'hyperparameters': (tuner.outputs['best_hyperparameters']
                       if enable_tuning else None), 
      'custom_executor_spec': executor_spec.ExecutorClassSpec(trainer_executor.GenericExecutor),
  }

  if ai_platform_training_args is not None:
    trainer_args.update({
        'custom_executor_spec':
            executor_spec.ExecutorClassSpec(
                ai_platform_trainer_executor.GenericExecutor
            ),
        'custom_config': {
            ai_platform_trainer_executor.TRAINING_ARGS_KEY:
                ai_platform_training_args,
        }
    })
  
    
  trainer = Trainer(**trainer_args)
  # TODO(step 6): Uncomment here to add Trainer to the pipeline.
  components.append(trainer)
  
  
  # Uses user-provided Python function that implements a model using TF-Learn.
  
  # Get the latest blessed model for model validation.
  model_resolver = ResolverNode(
      instance_name='latest_blessed_model_resolver',
      resolver_class=latest_blessed_model_resolver.LatestBlessedModelResolver,
      model=Channel(type=Model),
      model_blessing=Channel(type=ModelBlessing))
  components.append(model_resolver)

  # Uses TFMA to compute a evaluation statistics over features of a model and
  # perform quality validation of a candidate model (compared to a baseline).
  eval_config = tfma.EvalConfig(
    model_specs=[
        # Using signature 'eval' implies the use of an EvalSavedModel. To use
        # a serving model remove the signature to defaults to 'serving_default'
        # and add a label_key.
        #tfma.ModelSpec(signature_name='eval')
        tfma.ModelSpec(signature_name='serving_default',
                       label_key=configs.LABEL_KEY)
    ],
    metrics_specs=[
        tfma.MetricsSpec(
            # The metrics added here are in addition to those saved with the
            # model (assuming either a keras model or EvalSavedModel is used).
            # Any metrics added into the saved model (for example using
            # model.compile(..., metrics=[...]), etc) will be computed
            # automatically.
            metrics=[
                tfma.MetricConfig(class_name='ExampleCount')
            ],
            # To add validation thresholds for metrics saved with the model,
            # add them keyed by metric name to the thresholds map.
            thresholds = {
                'accuracy': tfma.MetricThreshold(
                    value_threshold=tfma.GenericValueThreshold(
                        lower_bound={'value': eval_accuracy_threshold}),
                    change_threshold=tfma.GenericChangeThreshold(
                       direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                       absolute={'value': -1e-10}))
            }
        )
    ],
    slicing_specs=[
        # An empty slice spec means the overall slice, i.e. the whole dataset.
        tfma.SlicingSpec(),
        # Data can be sliced along a feature column. In this case, data is
        # sliced along feature column Sex.
        tfma.SlicingSpec(feature_keys=['Sex']),
        tfma.SlicingSpec(feature_keys=['Age']),
        tfma.SlicingSpec(feature_keys=['Age_xf']),
        tfma.SlicingSpec(feature_keys=['Fare']),
        tfma.SlicingSpec(feature_keys=['Parch']),
        tfma.SlicingSpec(feature_keys=['Parch_xf']),
        tfma.SlicingSpec(feature_keys=['SibSp']),
        tfma.SlicingSpec(feature_keys=['SibSp_xf'])
    ])
  evaluator = Evaluator(
      examples=example_gen.outputs['examples'],
      model=trainer.outputs['model'],
      baseline_model=model_resolver.outputs['model'],
      # Change threshold will be ignored if there is no baseline (first run).
      eval_config=eval_config)
  components.append(evaluator)

  # Checks whether the model passed the validation steps and pushes the model
  # to a file destination if check passed.
  pusher_args = {
      'model':
          trainer.outputs['model'],
      'model_blessing':
          evaluator.outputs['blessing'],
      'push_destination':
          pusher_pb2.PushDestination(
              filesystem=pusher_pb2.PushDestination.Filesystem(
                  base_directory=serving_model_dir)),
  }
  if ai_platform_serving_args is not None:
    pusher_args.update({
        'custom_executor_spec':
            executor_spec.ExecutorClassSpec(ai_platform_pusher_executor.Executor
                                           ),
        'custom_config': {
            ai_platform_pusher_executor.SERVING_ARGS_KEY:
                ai_platform_serving_args
        },
    })
  pusher = Pusher(**pusher_args)  # pylint: disable=unused-variable
  components.append(pusher)
    
  if enable_tuning:
    components.append(tuner)

  return pipeline.Pipeline(
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      components=components,
      # Change this value to control caching of execution results. Default value
      # is `False`.
      # enable_cache=True,
      metadata_connection_config=metadata_connection_config,
      beam_pipeline_args=beam_pipeline_args,
  )
