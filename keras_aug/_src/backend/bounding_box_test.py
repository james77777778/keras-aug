import numpy as np
from absl.testing import parameterized
from keras.src import testing
from keras.src.testing.test_utils import named_product

from keras_aug._src.backend.bounding_box import BoundingBoxBackend


class BoundingBoxBackendTest(testing.TestCase, parameterized.TestCase):
    size = 1000.0
    xyxy_box = np.array([[[10, 20, 110, 120], [20, 30, 120, 130]]], "float32")
    yxyx_box = np.array([[[20, 10, 120, 110], [30, 20, 130, 120]]], "float32")
    xywh_box = np.array([[[10, 20, 100, 100], [20, 30, 100, 100]]], "float32")
    center_xywh_box = np.array(
        [[[60, 70, 100, 100], [70, 80, 100, 100]]], "float32"
    )

    def get_box(self, name):
        box_dict = {
            "xyxy": self.xyxy_box,
            "yxyx": self.yxyx_box,
            "xywh": self.xywh_box,
            "center_xywh": self.center_xywh_box,
            "rel_xyxy": self.xyxy_box / self.size,
            "rel_yxyx": self.yxyx_box / self.size,
            "rel_xywh": self.xywh_box / self.size,
            "rel_center_xywh": self.center_xywh_box / self.size,
        }
        return box_dict[name]

    @parameterized.named_parameters(
        named_product(
            source=[
                "xyxy",
                "yxyx",
                "xywh",
                "center_xywh",
                "rel_xyxy",
                "rel_yxyx",
                "rel_xywh",
                "rel_center_xywh",
            ],
            target=[
                "xyxy",
                "yxyx",
                "xywh",
                "center_xywh",
                "rel_xyxy",
                "rel_yxyx",
                "rel_xywh",
                "rel_center_xywh",
            ],
        )
    )
    def test_convert_format(self, source, target):
        bbox_backend = BoundingBoxBackend()
        boxes = self.get_box(source)
        ref_boxes = self.get_box(target)

        # Test batched
        result = bbox_backend.convert_format(boxes, source, target, 1000, 1000)
        self.assertAllClose(result, ref_boxes)

        # Test unbatched
        boxes = boxes[0]
        ref_boxes = ref_boxes[0]
        result = bbox_backend.convert_format(boxes, source, target, 1000, 1000)
        self.assertAllClose(result, ref_boxes)
