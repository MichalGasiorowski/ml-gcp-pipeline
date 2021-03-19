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
import time
from absl import logging
from tfx.orchestration import data_types

from config import Config
from pipeline import create_pipeline

from typing import Optional, Dict, List, Text
from distutils.util import strtobool


from tfx.orchestration import metadata
from tfx.orchestration.local.local_dag_runner import LocalDagRunner
from tfx.proto import trainer_pb2

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

#DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'train')

        
ARTIFACT_STORE = os.path.join(os.sep, 'home', 'jupyter', 'artifact-store')
SERVING_MODEL_DIR=os.path.join(os.sep, 'home', 'jupyter', 'serving_model')
DATA_ROOT_URI = 'gs://cloud-training-281409-kubeflowpipelines-default/tfx-template/data/titanic'
        
PIPELINE_NAME = Config.PIPELINE_NAME
PIPELINE_ROOT = os.path.join(ARTIFACT_STORE, PIPELINE_NAME, time.strftime("%Y%m%d_%H%M%S"))
os.makedirs(PIPELINE_ROOT, exist_ok=True)    

METADATA_PATH = os.path.join(PIPELINE_ROOT, 'tfx_metadata', PIPELINE_NAME, 'metadata.db')
        
def run():
  """Define a local pipeline."""

  beam_tmp_folder = '{}/beam/tmp'.format(Config.ARTIFACT_STORE_URI)
  beam_pipeline_args = [
      '--runner=DataflowRunner',
      '--experiments=shuffle_mode=auto',
      '--project=' + Config.PROJECT_ID,
      '--temp_location=' + beam_tmp_folder,
      '--region=' + Config.GCP_REGION,
  ]

  # Set the default values for the pipeline runtime parameters
  data_root_uri = data_types.RuntimeParameter(
      name='data-root-uri',
      default=Config.DATA_ROOT_URI,
      ptype=Text
  )

  train_steps = data_types.RuntimeParameter(
      name='train-steps',
      default=30000,
      ptype=int
  )

  tuner_steps = data_types.RuntimeParameter(
      name='tuner-steps',
      default=2000,
      ptype=int
  )

  eval_steps = data_types.RuntimeParameter(
      name='eval-steps',
      default=1000,
      ptype=int
  )

  enable_cache = data_types.RuntimeParameter(
      name='enable-cache',
      default=Config.ENABLE_CACHE,
      ptype=bool
  )

  LocalDagRunner().run(
      create_pipeline(
          pipeline_name=PIPELINE_NAME,
          pipeline_root=PIPELINE_ROOT,
          data_path=DATA_ROOT_URI,
          tuner_steps=tuner_steps,
          train_steps=train_steps,
          eval_steps=eval_steps,
          enable_tuning=strtobool(Config.ENABLE_TUNING),
          enable_cache=enable_cache,
          serving_model_dir=SERVING_MODEL_DIR,
          metadata_connection_config=metadata.sqlite_metadata_connection_config(
              METADATA_PATH)))


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  run()
