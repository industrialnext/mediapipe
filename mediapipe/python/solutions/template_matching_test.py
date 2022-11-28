# Copyright 2021 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for mediapipe.python.solutions.template_matching."""

import os

from absl.testing import absltest
from absl.testing import parameterized
import cv2
import numpy as np

# resources dependency
# undeclared dependency
from mediapipe.python.solutions import template_matching as mp_template_matching

TEST_IMAGE_PATH = "mediapipe/python/solutions/testdata"


class TemplateMatchingTest(parameterized.TestCase):
    def _draw(self, frame: np.ndarray, mask: np.ndarray):
        frame = np.minimum(frame, np.stack((mask,) * 3, axis=-1))
        path = os.path.join(tempfile.gettempdir(), self.id().split(".")[-1] + ".png")
        cv2.imwrite(path, frame)

    def test_invalid_image_shape(self):
        with mp_template_matching.TemplateMatching() as template_matching:
            with self.assertRaisesRegex(
                ValueError, "Input image must contain three channel rgb data."
            ):
                template_matching.process(
                    np.arange(36, dtype=np.uint8).reshape(3, 3, 4)
                )

    def test_blank_image(self):
        with mp_template_matching.SelfieSegmentation() as template_matching:
            image = np.zeros([100, 100, 3], dtype=np.uint8)
            image.fill(255)
            results = template_matching.process(image)
            # normalized_segmentation_mask = (results.segmentation_mask * 255).astype(int)
            # self.assertLess(np.amax(normalized_segmentation_mask), 1)


if __name__ == "__main__":
    absltest.main()
