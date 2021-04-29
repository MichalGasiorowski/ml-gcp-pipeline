# Lint as: python2, python3
# Copyright 2019 Google LLC. All Rights Reserved.
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
"""KFP runner configuration"""

import kfp

from tfx.orchestration import data_types
from tfx.orchestration.kubeflow import kubeflow_dag_runner
from tfx.orchestration import metadata

from typing import Optional, Dict, List, Text
from distutils.util import strtobool
import os
import shutil
import time

from absl import logging


from config import Config
from pipeline_args import TrainerConfig
from pipeline_args import TunerConfig
from pipeline_args import PusherConfig
from pipeline_args import RuntimeParametersConfig

# from pipeline import create_pipeline
import pipelines as pipeline

class KubeFlowRunner():
    """Class for Local Runner encapsulation"""

    def __init__(self):
        self.env_config = Config()
        self._setup_pipeline_parameters_from_env()

    def _set_additional_cloud_properties(self):
        self.USE_GS = self.env_config.USE_GS
        self.USE_AI_PLATFORM = self.env_config.USE_AI_PLATFORM
        if strtobool(self.USE_GS):
            self.ARTIFACT_STORE_URI = self.env_config.ARTIFACT_STORE_URI
            self.PIPELINE_ROOT = '{}/{}/{}'.format(
                self.ARTIFACT_STORE_URI,
                self.env_config.PIPELINE_NAME,
                kfp.dsl.RUN_ID_PLACEHOLDER)
            self.beam_tmp_folder = '{}/beam/tmp'.format(self.env_config.ARTIFACT_STORE_URI)
            self.beam_pipeline_args = [
                '--runner=DataflowRunner',
                '--experiments=shuffle_mode=auto',
                '--project=' + self.env_config.PROJECT_ID,
                '--temp_location=' + self.beam_tmp_folder,
                '--region=' + self.env_config.GCP_REGION,
            ]
        else:
            self.beam_tmp_folder = '{}/beam/tmp'.format(self.env_config.LOCAL_ARTIFACT_STORE)
            self.beam_pipeline_args = [
                '--runner=DirectRunner',
                '--experiments=shuffle_mode=auto',
                '--project=' + self.env_config.PROJECT_ID,
                '--temp_location=' + self.beam_tmp_folder,
                '--region=' + self.env_config.GCP_REGION,
            ]
        if strtobool(self.USE_AI_PLATFORM):
            self.ai_platform_training_args = {
                'project': self.env_config.PROJECT_ID,
                'region': self.env_config.GCP_REGION,
                'serviceAccount': self.env_config.CUSTOM_SERVICE_ACCOUNT,
                'masterConfig': {
                    'imageUri': self.env_config.TFX_IMAGE,
                }
            }
            self.ai_platform_serving_args = {
                'project_id': self.env_config.PROJECT_ID,
                'model_name': self.env_config.MODEL_NAME,
                'runtimeVersion': self.env_config.RUNTIME_VERSION,
                'pythonVersion': self.env_config.PYTHON_VERSION,
                'regions': [self.env_config.GCP_REGION]
            }

        else:
            self.ai_platform_training_args = None
            self.ai_platform_serving_args = None


    def _setup_pipeline_parameters_from_env(self):
        self.LOCAL_LOG_DIR = self.env_config.LOCAL_LOG_DIR
        self.PIPELINE_NAME = self.env_config.PIPELINE_NAME
        self.ENABLE_CACHE = self.env_config.ENABLE_CACHE

        self.TFX_IMAGE = self.env_config.TFX_IMAGE
        self.RUNTIME_VERSION = self.env_config.RUNTIME_VERSION
        self.PYTHON_VERSION = self.env_config.PYTHON_VERSION
        self.USE_KFP_SA = self.env_config.USE_KFP_SA

        self.DATA_ROOT_URI = self.env_config.DATA_ROOT_URI

        # properties applicable for local run
        self.HOME = self.env_config.HOME
        self.LOCAL_ARTIFACT_STORE = self.env_config.LOCAL_ARTIFACT_STORE
        self.LOCAL_SERVING_MODEL_DIR = self.env_config.LOCAL_SERVING_MODEL_DIR
        self.LOCAL_PIPELINE_ROOT = self.env_config.LOCAL_PIPELINE_ROOT
        self.LOCAL_METADATA_PATH = self.env_config.LOCAL_METADATA_PATH

        self._set_additional_cloud_properties()

        self.trainer_config = TrainerConfig.from_config(config=self.env_config, ai_platform_training_args=self.ai_platform_training_args)
        self.tuner_config = TunerConfig.from_config(config=self.env_config, ai_platform_tuner_args=None)
        self.pusher_config = PusherConfig.from_config(config=self.env_config, serving_model_dir=self.LOCAL_SERVING_MODEL_DIR,
                                                     ai_platform_serving_args=self.ai_platform_serving_args)
        # Set the default values for the pipeline runtime parameters
        self.runtime_parameters_config = RuntimeParametersConfig.from_config(config=self.env_config)

        metadata_config = kubeflow_dag_runner.get_default_kubeflow_metadata_config()

        self.runner_config = kubeflow_dag_runner.KubeflowDagRunnerConfig(
            kubeflow_metadata_config=metadata_config,
            pipeline_operator_funcs=kubeflow_dag_runner.get_default_pipeline_operator_funcs(
                strtobool(self.USE_KFP_SA)),
            tfx_image=self.env_config.TFX_IMAGE)

    def run(self):
        """Define a pipeline and run it using KubeFlow."""

        kubeflow_dag_runner.KubeflowDagRunner(config=self.runner_config).run(
            pipeline.create_pipeline(
                pipeline_name=self.PIPELINE_NAME,
                pipeline_root=self.PIPELINE_ROOT,
                data_root_uri=self.DATA_ROOT_URI,
                trainer_config=self.trainer_config,
                tuner_config=self.tuner_config,
                pusher_config=self.pusher_config,
                runtime_parameters_config=self.runtime_parameters_config,
                enable_cache=self.ENABLE_CACHE,
                local_run=False,
                beam_pipeline_args=self.beam_pipeline_args
                ))

        return self

if __name__ == '__main__':

    logging.set_verbosity(logging.INFO)
    kubeflowRunner = KubeFlowRunner()

    logging.info("PIPELINE_ROOT=" + kubeflowRunner.PIPELINE_ROOT)

    kubeflowRunner.run()