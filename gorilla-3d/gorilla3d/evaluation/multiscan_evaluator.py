# Modified from https://github.com/Gorilla-Lab-SCUT/gorilla-3d/blob/dev/gorilla3d/evaluation/scannet_evaluator.py
import os
import os.path as osp
from typing import List, Union

import numpy as np

from gorilla.evaluation import DatasetEvaluators

from .pattern import SemanticEvaluator, InstanceEvaluator

# CLASS_LABELS = ['floor', 'ceiling', 'wall', 'door', 'table', 'chair', 'cabinet', 'window', 'sofa', 'microwave', 'pillow',
# 'tv_monitor', 'curtain', 'trash_can', 'suitcase', 'sink', 'backpack', 'bed', 'refrigerator','toilet']
# CLASS_IDS = range(1, 21)
CLASS_LABELS = ['cabinet:door', 'cabinet:cabinet', 'table:table', 'cabinet:drawer', 'door:door', 'door:frame', 'window:window', 'table:drawer', 'window:frame', 'bed:bed', 'refrigerator:door', 
'toilet:toilet', "refrigerator:refrigerator", "bed:drawer", "microwave:microwave", "microwave:door", "toilet:lid", "table:door"]
CLASS_IDS = range(1, 19)

class MultiScanSemanticEvaluator(SemanticEvaluator):
    def __init__(self,
                 dataset_root,
                 class_labels: List[str] = CLASS_LABELS,
                 class_ids: Union[np.ndarray, List[int]] = CLASS_IDS,
                 **kwargs):
        super().__init__(class_labels=class_labels,
                         class_ids=class_ids,
                         **kwargs)
        self.dataset_root = dataset_root

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model.
            outputs: the outputs of a model. It is either list of semantic segmentation predictions
                or list of dicts with key "sem_seg" that contains semantic
                segmentation prediction in the same format.
        """
        if not isinstance(inputs, List):
            inputs = [inputs]
        if not isinstance(outputs, List):
            outputs = [outputs]
        for input, output in zip(inputs, outputs):
            scene_name = input["scene_name"]
            semantic_gt = self.read_gt(self.dataset_root, scene_name)
            semantic_pred = output["semantic_pred"].cpu().clone().numpy()
            semantic_pred = self.class_ids[semantic_pred]
            self.fill_confusion(semantic_pred, semantic_gt)

    @staticmethod
    def read_gt(origin_root, scene_name):
        label = np.loadtxt(os.path.join(origin_root, scene_name + "_sem.txt"))
        label = label.astype(np.int32)
        
        return label


# ---------- Label info ---------- #
# FOREGROUND_CLASS_LABELS = ['door', 'table', 'chair', 'cabinet', 'window', 'sofa', 'microwave', 'pillow',
# 'tv_monitor', 'curtain', 'trash_can', 'suitcase', 'sink', 'backpack', 'bed', 'refrigerator','toilet']
# FOREGROUND_CLASS_IDS = np.array(range(4, 21))

FOREGROUND_CLASS_LABELS = ['cabinet:door', 'cabinet:cabinet', 'table:table', 'cabinet:drawer', 'door:door', 'door:frame', 'window:window', 'table:drawer', 'window:frame', 'bed:bed', 'refrigerator:door', 
'toilet:toilet', "refrigerator:refrigerator", "bed:drawer", "microwave:microwave", "microwave:door", "toilet:lid", "table:door"]
FOREGROUND_CLASS_IDS = range(1, 19)


class MultiScanInstanceEvaluator(InstanceEvaluator):
    """
    Evaluate semantic segmentation metrics.
    """
    def __init__(self,
                 dataset_root: str,
                 class_labels: List[str] = FOREGROUND_CLASS_LABELS,
                 class_ids: List[int] = FOREGROUND_CLASS_IDS,
                 **kwargs):
        """
        Args:
            ignore_label: deprecated argument
        """
        super().__init__(class_labels=class_labels,
                         class_ids=class_ids,
                         **kwargs)
        self._dataset_root = dataset_root

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dicts.
            outputs: the outputs of a model. It is either list of semantic segmentation predictions
                or list of dicts with key "sem_seg" that contains semantic
                segmentation prediction in the same format.
        """
        if not isinstance(inputs, List):
            inputs = [inputs]
        if not isinstance(outputs, List):
            outputs = [outputs]
        for input, output in zip(inputs, outputs):
            scene_name = input["scene_name"]
            gt_file = osp.join(self._dataset_root, scene_name + ".txt")
            gt_ids = np.loadtxt(gt_file)
            self.assign(scene_name, output, gt_ids)


MultiScanEvaluator = DatasetEvaluators(
    [MultiScanSemanticEvaluator, MultiScanInstanceEvaluator])