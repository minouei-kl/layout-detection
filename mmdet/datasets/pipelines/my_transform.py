from mmdet.datasets import PIPELINES
import cv2
import numpy as np
@PIPELINES.register_module()
class MyTransform:

    def __call__(self, results):
        img = results['img']
        kernal = np.ones((2, 2), np.uint8)
        mask = 255 - img
        dst = cv2.dilate(mask, kernal, iterations=1)
        dst = cv2.medianBlur(dst, 3)
        results['img']=dst
        return results
