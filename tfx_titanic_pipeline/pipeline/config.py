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
"""The pipeline configurations.
"""

from __future__ import absolute_import

import os


class Config:

    def __init__(self):
        """Sets configuration vars."""
        # Lab user environment resource settings
        self.GCP_REGION = os.getenv("GCP_REGION", "us-central1")
        self.PROJECT_ID = os.getenv("PROJECT_ID", "cloud-training-281409")
        self.ARTIFACT_STORE_URI = os.getenv("ARTIFACT_STORE_URI",
                                            "gs://cloud-training-281409-kubeflowpipelines-default")
        self.CUSTOM_SERVICE_ACCOUNT = os.getenv("CUSTOM_SERVICE_ACCOUNT",
                                                "tfx-tuner-service-account@cloud-training-281409.iam.gserviceaccount.com")

        self.PIPELINE_NAME = os.getenv("PIPELINE_NAME", "tfx-titanic-training")
        self.MODEL_NAME = os.getenv("MODEL_NAME", "covertype_classifier")
        self.DATA_ROOT_URI = os.getenv("DATA_ROOT_URI",
                                       "gs://cloud-training-281409-kubeflowpipelines-default/tfx-template/data/titanic")
        self.TFX_IMAGE = os.getenv("KUBEFLOW_TFX_IMAGE", "tensorflow/tfx:0.25.0")
        self.RUNTIME_VERSION = os.getenv("RUNTIME_VERSION", "2.3")
        self.PYTHON_VERSION = os.getenv("PYTHON_VERSION", "3.7")
        self.USE_KFP_SA = os.getenv("USE_KFP_SA", "False")

        self.ENABLE_TUNING = os.getenv("ENABLE_TUNING", "True")
        self.TUNER_STEPS = os.getenv("TUNER_STEPS", "2000")
        self.MAX_TRIALS = os.getenv("MAX_TRIALS", "30")

        self.ENABLE_CACHE = os.getenv("ENABLE_CACHE", "False")
        self.TRAIN_STEPS = os.getenv("TRAIN_STEPS", "30000")
        self.EVAL_STEPS = os.getenv("EVAL_STEPS", "1000")
        self.EPOCHS = os.getenv("EPOCHS", "10")
        self.TRAIN_BATCH_SIZE = os.getenv("TRAIN_BATCH_SIZE", "64")
        self.EVAL_BATCH_SIZE = os.getenv("EVAL_BATCH_SIZE", "64")


        self.HOME = os.getenv("HOME", os.path.expanduser("~"))
        self.LOCAL_LOG_DIR = os.getenv("LOCAL_LOG_DIR", '/tmp/logs')
