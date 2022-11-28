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
"""MediaPipe Template Matching."""

from typing import NamedTuple

import numpy as np

# The following imports are needed because python pb2 silently discards
# unknown protobuf fields.
# pylint: disable=unused-import
from mediapipe.calculators.core import constant_side_packet_calculator_pb2
from mediapipe.calculators.tensor import image_to_tensor_calculator_pb2
from mediapipe.calculators.tensor import inference_calculator_pb2
from mediapipe.calculators.tensor import tensors_to_segmentation_calculator_pb2
from mediapipe.calculators.util import local_file_contents_calculator_pb2
from mediapipe.framework.tool import switch_container_pb2

# pylint: enable=unused-import

from mediapipe.python.solution_base import SolutionBase

_BINARYPB_FILE_PATH = (
    "mediapipe/modules/template_matching/template_matching_mobile_cpu.binarypb"
)


class TemplateMatching(SolutionBase):
    """MediaPipe Template Matching.

    MediaPipe Template Matching processes an RGB image and returns a
    segmentation mask.

    Please refer to
    https://solutions.mediapipe.dev/template_matching#python-solution-api for
    usage examples. (TO BE ADDED)
    """

    def __init__(self, model_selection=0):
        """Initializes a MediaPipe Template Matching object."""
        super().__init__(
            binary_graph_path=_BINARYPB_FILE_PATH,
            side_inputs={},
            outputs=["landmarks_render_data"],
        )

    def process(self, image: np.ndarray) -> NamedTuple:
        """Processes an RGB image and returns a segmentation mask.

        Args:
          image: An RGB image represented as a numpy ndarray.

        Raises:
          RuntimeError: If the underlying graph throws any error.
          ValueError: If the input image is not three channel RGB.

        Returns:
          A NamedTuple object with a "segmentation_mask" field that contains a float
          type 2d np array representing the mask.
        """

        return super().process(input_data={"image": image})
