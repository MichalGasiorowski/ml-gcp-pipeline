from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import time
import glob

import tensorflow as tf

from pipeline.local_runner import LocalRunner
from pipeline.config import Config

import pipeline.pipelines as pipelines


from distutils.util import strtobool
from tfx.orchestration import metadata
dir_path = os.path.dirname(os.path.realpath(__file__))
print("dir_path: ", dir_path)

tests_path = os.getcwd()
print("tests_path: ", tests_path)
local_data_dirpath = os.path.join(tests_path, '../', 'data')
local_data_dirpath = os.path.normpath(local_data_dirpath)

local_train_dirpath = os.path.join(local_data_dirpath, "train")
local_train_filepath = os.path.join(local_train_dirpath, "train.csv")
local_test_dirpath = os.path.join(local_data_dirpath, "test")
local_test_filepath = os.path.join(local_test_dirpath, "test.csv")

HOME = os.path.expanduser("~")
LOCAL_LOG_DIR = '/tmp/logs'


class PipelineTest(tf.test.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.setup_environment_variables()

    @classmethod
    def setup_environment_variables(cls):
        print("in setup_environment_variables")
        os.environ["PIPELINE_NAME"] = 'tfx-titanic-training'
        os.environ["MODEL_NAME"] = 'tfx_titanic_classifier'
        os.environ["DATA_ROOT_URI"] = local_train_dirpath
        os.environ["RUNTIME_VERSION"] = '2.3'
        os.environ["PYTHON_VERSION"] = '3.7'
        os.environ["ENABLE_TUNING"] = 'False'
        os.environ["ENABLE_CACHE"] = 'True'
        os.environ["TRAIN_STEPS"] = '1000'
        os.environ["TUNER_STEPS"] = '500'
        os.environ["EVAL_STEPS"] = '500'
        os.environ["EPOCHS"] = '2'
        os.environ["TRAIN_BATCH_SIZE"] = '64'
        os.environ["EVAL_BATCH_SIZE"] = '64'

    def setup_pipeline_arguments(self):
        print("in setup_pipeline_arguments")
        self.PIPELINE_NAME = self.env_config.PIPELINE_NAME
        self.ARTIFACT_STORE = os.path.join(os.sep, HOME, 'artifact-store')
        self.SERVING_MODEL_DIR = os.path.join(os.sep, HOME, 'serving_model')
        self.PIPELINE_ROOT = os.path.join(self.ARTIFACT_STORE, self.PIPELINE_NAME, time.strftime("%Y%m%d_%H%M%S"))
        self.METADATA_PATH = os.path.join(self.PIPELINE_ROOT, 'tfx_metadata', self.PIPELINE_NAME, 'metadata.db')
        self.enable_cache = self.env_config.ENABLE_CACHE
        self.data_root_uri = self.env_config.DATA_ROOT_URI

        os.makedirs(self.PIPELINE_ROOT, exist_ok=True)

    def setUp(self):
        self.env_config = Config()
        self.setup_pipeline_arguments()

    def tearDown(self):
        shutil.rmtree(self.PIPELINE_ROOT)

    def _create_pipeline(self):
        return pipelines.create_pipeline(
            pipeline_name=self.PIPELINE_NAME,
            pipeline_root=self.PIPELINE_ROOT,
            data_root_uri=self.data_root_uri,
            tuner_steps=int(self.env_config.TUNER_STEPS),
            train_steps=int(self.env_config.TRAIN_STEPS),
            eval_steps=int(self.env_config.EVAL_STEPS),
            epochs=int(self.env_config.EPOCHS),
            enable_tuning=strtobool(self.env_config.ENABLE_TUNING),
            enable_cache=self.enable_cache,
            local_run=True,
            serving_model_dir=self.SERVING_MODEL_DIR,
            metadata_connection_config=metadata.sqlite_metadata_connection_config(
                self.METADATA_PATH))

    # test scenarios below

    #def testPipelineCreation(self):
    #    created_pipeline = self._create_pipeline()

    #    self.assertLen(created_pipeline.components, 12, "12 components should be present in the pipeline.")

    def testLocalDagRunnerWithoutTuning(self):
        local_runner = LocalRunner()
        local_runner._setup_pipeline_parameters_from_env()
        local_runner.create_pipeline_root_folders_paths()

        local_runner.run()

        self.assertNotEmpty(glob.glob(os.path.join(self.PIPELINE_ROOT, 'CsvExampleGen/**/train/data_tfrecord*.*'), recursive=True))
        self.assertNotEmpty(glob.glob(os.path.join(self.PIPELINE_ROOT, 'CsvExampleGen/**/eval/data_tfrecord*.*'), recursive=True))

        #tfx_metadata/ tfx - titanic - training

        self.assertTrue(True)

if __name__ == '__main__':
    tf.test.main()
