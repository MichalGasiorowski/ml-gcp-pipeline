from __future__ import absolute_import

from config import Config
from distutils.util import strtobool
from typing import Any, Dict, List, Optional, Text
from tfx.orchestration import data_types


class TrainerConfig:

    def __init__(self, train_steps: int, eval_steps: int, epochs: int, train_batch_size: int, eval_batch_size: int,
                 ai_platform_training_args: Optional[Dict[Text, Text]] = None):
        self.train_steps = train_steps
        self.eval_steps = eval_steps
        self.epochs = epochs
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.ai_platform_training_args = ai_platform_training_args

    @classmethod
    def from_config(cls, config: Config, ai_platform_training_args):
        return cls(train_steps=int(config.TRAIN_STEPS), eval_steps=int(config.EVAL_STEPS), epochs=int(config.EPOCHS),
                   train_batch_size=int(config.TRAIN_BATCH_SIZE), eval_batch_size=int(config.EVAL_BATCH_SIZE),
                   ai_platform_training_args=ai_platform_training_args)


class TunerConfig:

    def __init__(self, enable_tuning: bool, tuner_steps: int, max_trials: int,
                 ai_platform_tuner_args: Optional[Dict[Text, Text]] = None
                 ):
        self.enable_tuning = enable_tuning
        self.tuner_steps = tuner_steps
        self.max_trials = max_trials
        self.ai_platform_tuner_args = ai_platform_tuner_args

    @classmethod
    def from_config(cls, config: Config, ai_platform_tuner_args: Optional[Dict[Text, Text]] = None):
        return cls(enable_tuning=strtobool(config.ENABLE_TUNING), tuner_steps=int(config.TUNER_STEPS),
                   max_trials=int(config.MAX_TRIALS), ai_platform_tuner_args=ai_platform_tuner_args)


class PusherConfig:

    def __init__(self, serving_model_dir: Optional[Text],
                 ai_platform_serving_args: Optional[Dict[Text, Text]] = None
                 ):
        self.serving_model_dir = serving_model_dir
        self.ai_platform_serving_args = ai_platform_serving_args

    @classmethod
    def from_config(cls, config: Config, serving_model_dir, ai_platform_serving_args):
        return cls(serving_model_dir=serving_model_dir, ai_platform_serving_args=ai_platform_serving_args)


class RuntimeParametersConfig:
    def __init__(self, data_root_uri: data_types.RuntimeParameter, train_steps: data_types.RuntimeParameter,
                 tuner_steps: data_types.RuntimeParameter, eval_steps: data_types.RuntimeParameter):
        self.data_root_uri = data_root_uri
        self.train_steps = train_steps
        self.tuner_steps = tuner_steps
        self.eval_steps = eval_steps

    @classmethod
    def from_config(cls, config: Config):
        data_root_uri = data_types.RuntimeParameter(
            name='data-root-uri',
            default=config.DATA_ROOT_URI,
            ptype=Text
        )

        train_steps = data_types.RuntimeParameter(
            name='train-steps',
            default=int(config.TRAIN_STEPS),
            ptype=int
        )

        tuner_steps = data_types.RuntimeParameter(
            name='tuner-steps',
            default=int(config.TUNER_STEPS),
            ptype=int
        )

        eval_steps = data_types.RuntimeParameter(
            name='eval-steps',
            default=int(config.EVAL_STEPS),
            ptype=int
        )
        return cls(data_root_uri=data_root_uri, train_steps=train_steps, tuner_steps=tuner_steps, eval_steps=eval_steps)
