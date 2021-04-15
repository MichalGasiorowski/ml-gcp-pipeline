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

tests_path = os.getcwd()
local_data_dirpath = os.path.join(tests_path, '../', 'data')
local_data_dirpath = os.path.normpath(local_data_dirpath)

local_train_dirpath = os.path.join(local_data_dirpath, "train")
local_train_filepath = os.path.join(local_train_dirpath, "train.csv")
local_test_dirpath = os.path.join(local_data_dirpath, "test")
local_test_filepath = os.path.join(local_test_dirpath, "test.csv")

HOME = os.path.expanduser("~")
LOCAL_LOG_DIR = '/tmp/logs'


class PipelineTest(tf.test.TestCase):
    component_output_directories = ["CsvExampleGen", "StatisticsGen", "SchemaGen", "ExampleValidator", "Transform",
                                    "Trainer", "Evaluator", "InfraValidator", "Pusher"]
    component_output_directories += ["tfx_metadata"]

    component_output_directories_wth_tuning = component_output_directories + ["Tuner"]

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
        os.environ["TRAIN_STEPS"] = '100'
        os.environ["TUNER_STEPS"] = '50'
        os.environ["EVAL_STEPS"] = '50'
        os.environ["EPOCHS"] = '2'
        os.environ["TRAIN_BATCH_SIZE"] = '64'
        os.environ["EVAL_BATCH_SIZE"] = '64'
        os.environ["MAX_TRIALS"] = '5'

    def setUp(self):
        self.env_config = Config()
        self.local_runner = None

    def tearDown(self):
        shutil.rmtree(self.local_runner.PIPELINE_ROOT)
        pass

    # test scenarios below

    def testLocalDagRunnerWithoutTuning(self):
        self.local_runner = LocalRunner()
        self.local_runner.create_pipeline_root_folders_paths()

        self.local_runner.run()

        for comp in PipelineTest.component_output_directories:
            self.assertTrue(os.path.exists(os.path.join(self.local_runner.PIPELINE_ROOT, comp)))

        self.assertNotEmpty(glob.glob(os.path.join(self.local_runner.PIPELINE_ROOT, 'CsvExampleGen/**/train/data_tfrecord*.*'), recursive=True))
        self.assertNotEmpty(glob.glob(os.path.join(self.local_runner.PIPELINE_ROOT, 'CsvExampleGen/**/eval/data_tfrecord*.*'), recursive=True))

    def testLocalDagRunnerWithTuning(self):
        os.environ["ENABLE_TUNING"] = 'True'
        os.environ["MAX_TRIALS"] = '5'
        self.local_runner = LocalRunner()
        self.local_runner.create_pipeline_root_folders_paths()

        self.local_runner.run()

        for comp in PipelineTest.component_output_directories_wth_tuning:
            self.assertTrue(os.path.exists(os.path.join(self.local_runner.PIPELINE_ROOT, comp)),
                            msg=f'{comp} component directory doesnt exist in PIPELINE_ROOT ( {self.local_runner.PIPELINE_ROOT} ) ')

        self.assertNotEmpty(
            glob.glob(os.path.join(self.local_runner.PIPELINE_ROOT, 'CsvExampleGen/**/train/data_tfrecord*.*'), recursive=True))
        self.assertNotEmpty(
            glob.glob(os.path.join(self.local_runner.PIPELINE_ROOT, 'CsvExampleGen/**/eval/data_tfrecord*.*'), recursive=True))

if __name__ == '__main__':
    tf.test.main()
