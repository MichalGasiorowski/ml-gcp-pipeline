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
"""Define LocalDagRunner to run the pipeline locally."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import time
from distutils.util import strtobool

from absl import logging
from tfx.orchestration import metadata
from tfx.orchestration.local.local_dag_runner import LocalDagRunner

from config import Config
from pipeline_args import TrainerConfig

# from pipeline import create_pipeline
import pipelines as pipeline


# import features as features

# TFX pipeline produces many output files and metadata. All output data will be
# stored under this OUTPUT_DIR.
# NOTE: It is recommended to have a separated OUTPUT_DIR which is *outside* of
#       the source code structure. Please change OUTPUT_DIR to other location
#       where we can store outputs of the pipeline.


# TFX produces two types of outputs, files and metadata.
# - Files will be created under PIPELINE_ROOT directory.
# - Metadata will be written to SQLite database in METADATA_PATH.

# The last component of the pipeline, "Pusher" will produce serving model under
# SERVING_MODEL_DIR.

# Specifies data file directory. DATA_PATH should be a directory containing CSV
# files for CsvExampleGen in this example. By default, data files are in the
# `data` directory.
# NOTE: If you upload data files to GCS(which is recommended if you use
#       Kubeflow), you can use a path starting "gs://YOUR_BUCKET_NAME/path" for
#       DATA_PATH. For example,
#       DATA_PATH = 'gs://bucket/chicago_taxi_trips/csv/'.

# DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'train')

class LocalRunner():
    """Class for Local Runner encapsulation"""

    def __init__(self):
        self.env_config = Config()
        self._setup_pipeline_parameters_from_env()

    def _setup_pipeline_parameters_from_env(self):
        self.HOME = self.env_config.HOME
        self.LOCAL_LOG_DIR = self.env_config.LOCAL_LOG_DIR
        self.PIPELINE_NAME = self.env_config.PIPELINE_NAME
        self.ENABLE_CACHE = self.env_config.ENABLE_CACHE
        self.ENABLE_TUNING = self.env_config.ENABLE_TUNING
        self.data_root_uri = self.env_config.DATA_ROOT_URI
        self.TRAIN_STEPS = self.env_config.TRAIN_STEPS
        self.TUNER_STEPS = self.env_config.TUNER_STEPS
        self.EVAL_STEPS = self.env_config.EVAL_STEPS
        self.EPOCHS = self.env_config.EPOCHS
        self.MAX_TRIALS = self.env_config.MAX_TRIALS
        self.TRAIN_BATCH_SIZE = self.env_config.TRAIN_BATCH_SIZE
        self.EVAL_BATCH_SIZE = self.env_config.EVAL_BATCH_SIZE

        self.ARTIFACT_STORE = os.path.join(os.sep, self.HOME, 'artifact-store')
        self.SERVING_MODEL_DIR = os.path.join(os.sep, self.HOME, 'serving_model')
        self.PIPELINE_ROOT = os.path.join(self.ARTIFACT_STORE, self.PIPELINE_NAME, time.strftime("%Y%m%d_%H%M%S"))
        self.METADATA_PATH = os.path.join(self.PIPELINE_ROOT, 'tfx_metadata', self.PIPELINE_NAME, 'metadata.db')

        self.trainerConfig = TrainerConfig.from_config(config=self.env_config, ai_platform_training_args = None)

    def create_pipeline_root_folders_paths(self):
        os.makedirs(self.PIPELINE_ROOT, exist_ok=True)

    def remove_folders(self, folder):
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

    def run(self):
        # clear local log folder
        logging.info('Cleaning local log folder : %s' % self.LOCAL_LOG_DIR)
        os.makedirs(self.LOCAL_LOG_DIR, exist_ok=True)
        self.remove_folders(self.LOCAL_LOG_DIR)

        """Define a local pipeline and run it."""

        LocalDagRunner().run(
            pipeline.create_pipeline(
                pipeline_name=self.PIPELINE_NAME,
                pipeline_root=self.PIPELINE_ROOT,
                data_root_uri=self.data_root_uri,
                tuner_steps=int(self.TUNER_STEPS),
                trainerConfig=self.trainerConfig,
                enable_tuning=strtobool(self.ENABLE_TUNING),
                max_trials=int(self.MAX_TRIALS),
                enable_cache=self.ENABLE_CACHE,
                local_run=True,
                serving_model_dir=self.SERVING_MODEL_DIR,
                metadata_connection_config=metadata.sqlite_metadata_connection_config(
                    self.METADATA_PATH)))
        return self


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    localRunner = LocalRunner()
    localRunner.create_pipeline_root_folders_paths()

    logging.info("PIPELINE_ROOT=" + localRunner.PIPELINE_ROOT)

    localRunner.run()
