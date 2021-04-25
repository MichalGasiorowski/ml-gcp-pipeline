# Copyright 2021 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#            http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Covertype training pipeline DSL."""

from __future__ import absolute_import

from typing import Any, Dict, List, Optional, Text

import absl
import tensorflow_model_analysis as tfma
from ml_metadata.proto import metadata_store_pb2
from tfx.components import CsvExampleGen
from tfx.components import Evaluator
from tfx.components import ExampleValidator
from tfx.components import ImporterNode
from tfx.components import InfraValidator
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
from tfx.orchestration import pipeline
from tfx.proto import example_gen_pb2
from tfx.proto import infra_validator_pb2
from tfx.proto import pusher_pb2
from tfx.proto import tuner_pb2
from tfx.types import Channel
from tfx.types.standard_artifacts import Model
from tfx.types.standard_artifacts import ModelBlessing
from tfx.types.standard_artifacts import Schema
from tfx.types.standard_artifacts import HyperParameters

from pipeline_args import TrainerConfig
from pipeline_args import TunerConfig
from pipeline_args import PusherConfig

SCHEMA_FOLDER = 'schema'
TRANSFORM_MODULE_FILE = 'preprocessing.py'
TRAIN_MODULE_FILE = 'model.py'


def create_pipeline(pipeline_name: Text,
                    pipeline_root: Text,
                    data_root_uri,
                    trainerConfig: TrainerConfig,
                    tunerConfig: TunerConfig,
                    pusherConfig: PusherConfig,
                    local_run: bool = False,
                    beam_pipeline_args: Optional[List[Text]] = None,
                    enable_cache: Optional[bool] = True,
                    metadata_connection_config: Optional[metadata_store_pb2.ConnectionConfig] = None,
                    ) -> pipeline.Pipeline:
    """Trains and deploys the Keras Titanic Classifier with TFX and Kubeflow Pipeline on Google Cloud.
  Args:
    pipeline_name: name of the TFX pipeline being created.
    pipeline_root: root directory of the pipeline. Should be a valid GCS path.
    data_root_uri: uri of the dataset.
    train_steps: runtime parameter for number of model training steps for the Trainer component.
    eval_steps: runtime parameter for number of model evaluation steps for the Trainer component.
    enable_tuning: If True, the hyperparameter tuning through CloudTuner is
      enabled.    
    ai_platform_training_args: Args of CAIP training job. Please refer to
      https://cloud.google.com/ml-engine/reference/rest/v1/projects.jobs#Job
      for detailed description.
    ai_platform_serving_args: Args of CAIP model deployment. Please refer to
      https://cloud.google.com/ml-engine/reference/rest/v1/projects.models
      for detailed description.
    beam_pipeline_args: Optional list of beam pipeline options. Please refer to
      https://cloud.google.com/dataflow/docs/guides/specifying-exec-params#setting-other-cloud-dataflow-pipeline-options.
      When this argument is not provided, the default is to use GCP
      DataflowRunner with 50GB disk size as specified in this function. If an
      empty list is passed in, default specified by Beam will be used, which can
      be found at
      https://cloud.google.com/dataflow/docs/guides/specifying-exec-params#setting-other-cloud-dataflow-pipeline-options
    enable_cache: Optional boolean
  Returns:
    A TFX pipeline object.
  """

    absl.logging.info('train_steps for training: %s' % trainerConfig.train_steps)
    absl.logging.info('tuner_steps for tuning: %s' % tunerConfig.tuner_steps)

    absl.logging.info('data_root_uri for training: %s' % data_root_uri)
    absl.logging.info('eval_steps for evaluating: %s' % trainerConfig.eval_steps)

    # Brings data into the pipeline and splits the data into training and eval splits
    output_config = example_gen_pb2.Output(
        split_config=example_gen_pb2.SplitConfig(splits=[
            example_gen_pb2.SplitConfig.Split(name='train', hash_buckets=4),
            example_gen_pb2.SplitConfig.Split(name='eval', hash_buckets=1)
        ]))

    # examples = external_input(data_root_uri)
    examplegen = CsvExampleGen(input_base=data_root_uri, output_config=output_config)

    # example_gen = tfx.components.CsvExampleGen(
    #    input_base=DATA_ROOT,
    #    output_config=output_config)

    # examplegen = CsvExampleGen(input_base=data_root_uri)

    # Computes statistics over data for visualization and example validation.
    statisticsgen = StatisticsGen(examples=examplegen.outputs.examples)

    # Generates schema based on statistics files. Even though, we use user-provided schema
    # we still want to generate the schema of the newest data for tracking and comparison
    schemagen = SchemaGen(statistics=statisticsgen.outputs.statistics)

    # Import a user-provided schema
    import_schema = ImporterNode(
        instance_name='import_user_schema',
        source_uri=SCHEMA_FOLDER,
        artifact_type=Schema)

    # Performs anomaly detection based on statistics and data schema.
    examplevalidator = ExampleValidator(
        statistics=statisticsgen.outputs.statistics,
        schema=import_schema.outputs.result)

    # Performs transformations and feature engineering in training and serving.
    transform = Transform(
        examples=examplegen.outputs.examples,
        schema=import_schema.outputs.result,
        module_file=TRANSFORM_MODULE_FILE)

    # Tunes the hyperparameters for model training based on user-provided Python
    # function. Note that once the hyperparameters are tuned, you can drop the
    # Tuner component from pipeline and feed Trainer with tuned hyperparameters.

    hparams_importer = ImporterNode(
        instance_name='import_hparams',
        source_uri='hyperparameters',
        artifact_type=HyperParameters)

    if tunerConfig.enable_tuning:
        tuner_args = {
            'module_file': TRAIN_MODULE_FILE,
            'examples': transform.outputs.transformed_examples,
            'transform_graph': transform.outputs.transform_graph,
            'train_args': {'num_steps': tunerConfig.tuner_steps},
            'eval_args': {'num_steps': trainerConfig.eval_steps},
            'custom_config': {'max_trials': tunerConfig.max_trials}
            # 'tune_args': tuner_pb2.TuneArgs(num_parallel_trials=3),
        }

        if tunerConfig.ai_platform_tuner_args is not None:
            tuner_args.update({
                'custom_config': {
                    ai_platform_trainer_executor.TRAINING_ARGS_KEY: tunerConfig.ai_platform_tuner_args
                },
                'tune_args': tuner_pb2.TuneArgs(num_parallel_trials=3)
            })

        absl.logging.info("tuner_args: " + str(tuner_args))
        tuner = Tuner(**tuner_args)

    hyperparameters = tuner.outputs.best_hyperparameters if tunerConfig.enable_tuning else hparams_importer.outputs['result']

    # Trains the model using a user provided trainer function.

    trainer_args = {
        'module_file': TRAIN_MODULE_FILE,
        'transformed_examples': transform.outputs.transformed_examples,
        'schema': import_schema.outputs.result,
        'transform_graph': transform.outputs.transform_graph,
        'train_args': {'num_steps': trainerConfig.train_steps},
        'eval_args': {'num_steps': trainerConfig.eval_steps},
        #'hyperparameters': tuner.outputs.best_hyperparameters if tunerConfig.enable_tuning else None,
        'hyperparameters': hyperparameters,
        'custom_config': {'epochs': trainerConfig.epochs, 'train_batch_size': trainerConfig.train_batch_size,
                          'eval_batch_size': trainerConfig.eval_batch_size}
    }

    if trainerConfig.ai_platform_training_args is not None:
        trainer_args['custom_config'].update({
            ai_platform_trainer_executor.TRAINING_ARGS_KEY:
                trainerConfig.ai_platform_training_args,
        })
        trainer_args.update({
            'custom_executor_spec':
                executor_spec.ExecutorClassSpec(ai_platform_trainer_executor.GenericExecutor),
            # 'custom_config': {
            #    ai_platform_trainer_executor.TRAINING_ARGS_KEY:
            #        ai_platform_training_args,
            # }
        })
    else:
        trainer_args.update({
            'custom_executor_spec':
                executor_spec.ExecutorClassSpec(trainer_executor.GenericExecutor),
        })

    trainer = Trainer(**trainer_args)

    # Get the latest blessed model for model validation.
    resolver = ResolverNode(
        instance_name='latest_blessed_model_resolver',
        resolver_class=latest_blessed_model_resolver.LatestBlessedModelResolver,
        model=Channel(type=Model),
        model_blessing=Channel(type=ModelBlessing))

    # Uses TFMA to compute a evaluation statistics over features of a model.
    accuracy_threshold = tfma.MetricThreshold(
        value_threshold=tfma.GenericValueThreshold(
            lower_bound={'value': 0.5},
            upper_bound={'value': 0.995}),
    )

    metrics_specs = tfma.MetricsSpec(
        metrics=[
            tfma.MetricConfig(class_name='BinaryAccuracy',
                              threshold=accuracy_threshold),
            tfma.MetricConfig(class_name='ExampleCount')])

    eval_config = tfma.EvalConfig(
        model_specs=[
            tfma.ModelSpec(label_key='Survived')
        ],
        metrics_specs=[metrics_specs],
        slicing_specs=[
            tfma.SlicingSpec()
            ,tfma.SlicingSpec(feature_keys=['Sex'])
            ,tfma.SlicingSpec(feature_keys=['Age'])
            ,tfma.SlicingSpec(feature_keys=['Parch'])
        ]
    )

    evaluator = Evaluator(
        examples=examplegen.outputs.examples,
        model=trainer.outputs.model,
        baseline_model=resolver.outputs.model,
        eval_config=eval_config
    )

    # Validate model can be loaded and queried in sand-boxed environment
    # mirroring production.

    serving_config = None

    if local_run:
        serving_config = infra_validator_pb2.ServingSpec(
            tensorflow_serving=infra_validator_pb2.TensorFlowServing(tags=['latest']),
            local_docker=infra_validator_pb2.LocalDockerConfig()  # Running on local docker.
        )
    else:
        serving_config = infra_validator_pb2.ServingSpec(
            tensorflow_serving=infra_validator_pb2.TensorFlowServing(tags=['latest']),
            kubernetes=infra_validator_pb2.KubernetesConfig()  # Running on K8s.
        )

    validation_config = infra_validator_pb2.ValidationSpec(
        max_loading_time_seconds=180,
        num_tries=3,
    )

    request_config = infra_validator_pb2.RequestSpec(
        tensorflow_serving=infra_validator_pb2.TensorFlowServingRequestSpec(),
        num_examples=3,
    )

    infravalidator = InfraValidator(
        model=trainer.outputs.model,
        examples=examplegen.outputs.examples,
        serving_spec=serving_config,
        validation_spec=validation_config,
        request_spec=request_config,
    )

    # Checks whether the model passed the validation steps and pushes the model
    # to CAIP Prediction if checks are passed.

    pusher_args = {
        'model': trainer.outputs.model,
        'model_blessing': evaluator.outputs.blessing,
        'infra_blessing': infravalidator.outputs.blessing
    }

    if local_run:
        pusher_args.update({'push_destination':
            pusher_pb2.PushDestination(
                filesystem=pusher_pb2.PushDestination.Filesystem(
                    base_directory=pusherConfig.serving_model_dir))})

    if pusherConfig.ai_platform_serving_args is not None:
        pusher_args.update({
            'custom_executor_spec':
                executor_spec.ExecutorClassSpec(ai_platform_pusher_executor.Executor
                                                ),
            'custom_config': {
                ai_platform_pusher_executor.SERVING_ARGS_KEY:
                    pusherConfig.ai_platform_serving_args
            },
        })

    pusher = Pusher(**pusher_args)  # pylint: disable=unused-variable

    components = [
        examplegen,
        statisticsgen,
        schemagen,
        import_schema,
        examplevalidator,
        transform,
        trainer,
        resolver,
        evaluator,
        infravalidator,
        pusher
    ]

    if tunerConfig.enable_tuning:
        components.append(tuner)
    else:
        components.append(hparams_importer)

    return pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components,
        enable_cache=enable_cache,
        metadata_connection_config=metadata_connection_config,
        beam_pipeline_args=beam_pipeline_args
    )
