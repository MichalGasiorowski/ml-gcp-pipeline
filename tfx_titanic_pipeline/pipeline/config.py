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

import os

class Config:
    """Sets configuration vars."""
    # Lab user environment resource settings
    GCP_REGION=os.getenv("GCP_REGION", "us-central1")
    PROJECT_ID=os.getenv("PROJECT_ID", "cloud-training-281409")
    ARTIFACT_STORE_URI=os.getenv("ARTIFACT_STORE_URI", "gs://cloud-training-281409-kubeflowpipelines-default")
    CUSTOM_SERVICE_ACCOUNT=os.getenv("CUSTOM_SERVICE_ACCOUNT", "tfx-tuner-service-account@cloud-training-281409.iam.gserviceaccount.com")
    # Lab user runtime environment settings
    PIPELINE_NAME=os.getenv("PIPELINE_NAME", "tfx-titanic-training")
    MODEL_NAME=os.getenv("MODEL_NAME", "covertype_classifier")
    DATA_ROOT_URI=os.getenv("DATA_ROOT_URI", "gs://cloud-training-281409-kubeflowpipelines-default/tfx-template/data/titanic")
    TFX_IMAGE=os.getenv("KUBEFLOW_TFX_IMAGE", "tensorflow/tfx:0.25.0")
    RUNTIME_VERSION=os.getenv("RUNTIME_VERSION", "2.3")
    PYTHON_VERSION=os.getenv("PYTHON_VERSION", "3.7")    
    USE_KFP_SA=os.getenv("USE_KFP_SA", "False")
    ENABLE_TUNING=os.getenv("ENABLE_TUNING", "True")
    ENABLE_CACHE = os.getenv("ENABLE_CACHE", "False")
    TRAIN_STEPS = os.getenv("TRAIN_STEPS", "30000")
    TUNER_STEPS = os.getenv("TUNER_STEPS", "2000")
    EVAL_STEPS = os.getenv("EVAL_STEPS", "1000")
    
    
    