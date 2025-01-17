# Copyright 2022 The MediaPipe Authors. All Rights Reserved.s
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

import collections
import os
import shutil
from typing import NamedTuple
import unittest

from absl import flags
from absl.testing import parameterized
import tensorflow as tf

from mediapipe.model_maker.python.vision.gesture_recognizer import dataset
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.tasks.python.test import test_utils

FLAGS = flags.FLAGS

_TEST_DATA_DIRNAME = 'raw_data'


class DatasetTest(tf.test.TestCase, parameterized.TestCase):

  def test_split(self):
    input_data_dir = test_utils.get_test_data_path(_TEST_DATA_DIRNAME)
    data = dataset.Dataset.from_folder(
        dirname=input_data_dir, hparams=dataset.HandDataPreprocessingParams())
    train_data, test_data = data.split(0.5)

    self.assertLen(train_data, 17)
    for _, elem in enumerate(train_data.gen_tf_dataset(is_training=True)):
      self.assertEqual(elem[0].shape, (1, 128))
      self.assertEqual(elem[1].shape, ([1, 4]))
    self.assertEqual(train_data.num_classes, 4)
    self.assertEqual(train_data.label_names, ['none', 'call', 'four', 'rock'])

    self.assertLen(test_data, 18)
    for _, elem in enumerate(test_data.gen_tf_dataset(is_training=True)):
      self.assertEqual(elem[0].shape, (1, 128))
      self.assertEqual(elem[1].shape, ([1, 4]))
    self.assertEqual(test_data.num_classes, 4)
    self.assertEqual(test_data.label_names, ['none', 'call', 'four', 'rock'])

  def test_from_folder(self):
    input_data_dir = test_utils.get_test_data_path(_TEST_DATA_DIRNAME)
    data = dataset.Dataset.from_folder(
        dirname=input_data_dir, hparams=dataset.HandDataPreprocessingParams())
    for _, elem in enumerate(data.gen_tf_dataset(is_training=True)):
      self.assertEqual(elem[0].shape, (1, 128))
      self.assertEqual(elem[1].shape, ([1, 4]))
    self.assertLen(data, 35)
    self.assertEqual(data.num_classes, 4)
    self.assertEqual(data.label_names, ['none', 'call', 'four', 'rock'])

  def test_create_dataset_from_empty_folder_raise_value_error(self):
    with self.assertRaisesRegex(ValueError, 'Image dataset directory is empty'):
      dataset.Dataset.from_folder(
          dirname=self.get_temp_dir(),
          hparams=dataset.HandDataPreprocessingParams())

  def test_create_dataset_from_folder_without_none_raise_value_error(self):
    input_data_dir = test_utils.get_test_data_path(_TEST_DATA_DIRNAME)
    tmp_dir = self.create_tempdir()
    # Copy input dataset to a temporary directory and skip 'None' directory.
    for name in os.listdir(input_data_dir):
      if name == 'none':
        continue
      src_dir = os.path.join(input_data_dir, name)
      dst_dir = os.path.join(tmp_dir, name)
      shutil.copytree(src_dir, dst_dir)

    with self.assertRaisesRegex(ValueError,
                                'Label set does not contain label "None"'):
      dataset.Dataset.from_folder(
          dirname=tmp_dir, hparams=dataset.HandDataPreprocessingParams())

  def test_create_dataset_from_folder_with_capital_letter_in_folder_name(self):
    input_data_dir = test_utils.get_test_data_path(_TEST_DATA_DIRNAME)
    tmp_dir = self.create_tempdir()
    # Copy input dataset to a temporary directory and change the base folder
    # name to upper case letter, e.g. 'none' -> 'NONE'
    for name in os.listdir(input_data_dir):
      src_dir = os.path.join(input_data_dir, name)
      dst_dir = os.path.join(tmp_dir, name.upper())
      shutil.copytree(src_dir, dst_dir)

    upper_base_folder_name = list(os.listdir(tmp_dir))
    self.assertCountEqual(upper_base_folder_name,
                          ['CALL', 'FOUR', 'NONE', 'ROCK'])

    data = dataset.Dataset.from_folder(
        dirname=tmp_dir, hparams=dataset.HandDataPreprocessingParams())
    for _, elem in enumerate(data.gen_tf_dataset(is_training=True)):
      self.assertEqual(elem[0].shape, (1, 128))
      self.assertEqual(elem[1].shape, ([1, 4]))
    self.assertLen(data, 35)
    self.assertEqual(data.num_classes, 4)
    self.assertEqual(data.label_names, ['NONE', 'CALL', 'FOUR', 'ROCK'])

  @parameterized.named_parameters(
      dict(
          testcase_name='invalid_field_name_multi_hand_landmark',
          hand=collections.namedtuple('Hand', [
              'multi_hand_landmark', 'multi_hand_world_landmarks',
              'multi_handedness'
          ])(1, 2, 3)),
      dict(
          testcase_name='invalid_field_name_multi_hand_world_landmarks',
          hand=collections.namedtuple('Hand', [
              'multi_hand_landmarks', 'multi_hand_world_landmark',
              'multi_handedness'
          ])(1, 2, 3)),
      dict(
          testcase_name='invalid_field_name_multi_handed',
          hand=collections.namedtuple('Hand', [
              'multi_hand_landmarks', 'multi_hand_world_landmarks',
              'multi_handed'
          ])(1, 2, 3)),
      dict(
          testcase_name='multi_hand_landmarks_is_none',
          hand=collections.namedtuple('Hand', [
              'multi_hand_landmarks', 'multi_hand_world_landmarks',
              'multi_handedness'
          ])(None, 2, 3)),
      dict(
          testcase_name='multi_hand_world_landmarks_is_none',
          hand=collections.namedtuple('Hand', [
              'multi_hand_landmarks', 'multi_hand_world_landmarks',
              'multi_handedness'
          ])(1, None, 3)),
      dict(
          testcase_name='multi_handedness_is_none',
          hand=collections.namedtuple('Hand', [
              'multi_hand_landmarks', 'multi_hand_world_landmarks',
              'multi_handedness'
          ])(1, 2, None)),
  )
  def test_create_dataset_from_invalid_hand_data(self, hand: NamedTuple):
    with unittest.mock.patch.object(
        mp_hands.Hands, 'process', return_value=hand):
      input_data_dir = test_utils.get_test_data_path(_TEST_DATA_DIRNAME)
      with self.assertRaisesRegex(ValueError, 'No valid hand is detected'):
        dataset.Dataset.from_folder(
            dirname=input_data_dir,
            hparams=dataset.HandDataPreprocessingParams())


if __name__ == '__main__':
  tf.test.main()
