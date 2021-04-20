from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from numpy import nan

import client.prod_client as prod_client
import client.client_util as client_util

train_survived_examples_data = [
    {'PassengerId': 2,
  'Pclass': 1,
  'Name': 'Cumings, Mrs. John Bradley (Florence Briggs Thayer)',
  'Sex': 'female',
  'Age': 38.0,
  'SibSp': 1,
  'Parch': 0,
  'Ticket': 'PC 17599',
  'Fare': 71.2833,
  'Cabin': 'C85',
  'Embarked': 'C'},
 {'PassengerId': 3,
  'Pclass': 3,
  'Name': 'Heikkinen, Miss. Laina',
  'Sex': 'female',
  'Age': 26.0,
  'SibSp': 0,
  'Parch': 0,
  'Ticket': 'STON/O2. 3101282',
  'Fare': 7.925,
  'Cabin': '',
  'Embarked': 'S'},
 {'PassengerId': 4,
  'Pclass': 1,
  'Name': 'Futrelle, Mrs. Jacques Heath (Lily May Peel)',
  'Sex': 'female',
  'Age': 35.0,
  'SibSp': 1,
  'Parch': 0,
  'Ticket': '113803',
  'Fare': 53.1,
  'Cabin': 'C123',
  'Embarked': 'S'},
 {'PassengerId': 9,
  'Pclass': 3,
  'Name': 'Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)',
  'Sex': 'female',
  'Age': 27.0,
  'SibSp': 0,
  'Parch': 2,
  'Ticket': '347742',
  'Fare': 11.1333,
  'Cabin': '',
  'Embarked': 'S'},
 {'PassengerId': 10,
  'Pclass': 2,
  'Name': 'Nasser, Mrs. Nicholas (Adele Achem)',
  'Sex': 'female',
  'Age': 14.0,
  'SibSp': 1,
  'Parch': 0,
  'Ticket': '237736',
  'Fare': 30.0708,
  'Cabin': '',
  'Embarked': 'C'},
 {'PassengerId': 11,
  'Pclass': 3,
  'Name': 'Sandstrom, Miss. Marguerite Rut',
  'Sex': 'female',
  'Age': 4.0,
  'SibSp': 1,
  'Parch': 1,
  'Ticket': 'PP 9549',
  'Fare': 16.7,
  'Cabin': 'G6',
  'Embarked': 'S'},
 {'PassengerId': 12,
  'Pclass': 1,
  'Name': 'Bonnell, Miss. Elizabeth',
  'Sex': 'female',
  'Age': 58.0,
  'SibSp': 0,
  'Parch': 0,
  'Ticket': '113783',
  'Fare': 26.55,
  'Cabin': 'C103',
  'Embarked': 'S'},
 {'PassengerId': 16,
  'Pclass': 2,
  'Name': 'Hewlett, Mrs. (Mary D Kingcome) ',
  'Sex': 'female',
  'Age': 55.0,
  'SibSp': 0,
  'Parch': 0,
  'Ticket': '248706',
  'Fare': 16.0,
  'Cabin': '',
  'Embarked': 'S'},
 {'PassengerId': 18,
  'Pclass': 2,
  'Name': 'Williams, Mr. Charles Eugene',
  'Sex': 'male',
  'Age': nan,
  'SibSp': 0,
  'Parch': 0,
  'Ticket': '244373',
  'Fare': 13.0,
  'Cabin': '',
  'Embarked': 'S'},
 {'PassengerId': 20,
  'Pclass': 3,
  'Name': 'Masselmani, Mrs. Fatima',
  'Sex': 'female',
  'Age': nan,
  'SibSp': 0,
  'Parch': 0,
  'Ticket': '2649',
  'Fare': 7.225,
  'Cabin': '',
  'Embarked': 'C'}]


class ProdClientTest(tf.test.TestCase):

  def testProdClientWorks(self):
      prodclient = prod_client.ProdClient(host='localhost:8500', model_name='tfx_titanic_classifier', \
                                          in_tensor_name='examples', signature_name='serving_default')

      prediction = prodclient.predict(request_data=train_survived_examples_data, request_timeout=30)
      self.assertIsNotNone(prediction)
      first_key = list(prediction.keys())[0]
      print(prediction[first_key])
      self.assertTrue(len(prediction[first_key]) == len(train_survived_examples_data))

      self.assertTrue(np.isfinite(prediction[first_key]).all(), msg='All predictions should be finite numbers, +inf, -inf, nan are prohibited.')

      self.assertAllInRange(target=prediction[first_key], lower_bound=0.0, upper_bound=1.0)






