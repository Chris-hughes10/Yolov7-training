import contextlib
import json
from typing import Set, List, Dict
import sys

import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pytorch_accelerated.callbacks import TrainerCallback


XMIN_COL = "xmin"
YMIN_COL = "ymin"
XMAX_COL = "xmax"
YMAX_COL = "ymax"
CLASS_ID_COL = "class_id"
SCORE_COL = "score"
BOX_WIDTH_COL = "w"
BOX_HEIGHT_COL = "h"
IMAGE_ID_COL = "image_id"


class DummyFile(object):
    def write(self, x):
        pass


@contextlib.contextmanager
def silencer():
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    yield
    sys.stdout = save_stdout


class COCOMeanAveragePrecision:
    """Mean Average Precision for single IoU threshold based on pycocotools

    :param foreground_threshold: IoU threshold for which we want to calculate mAP.
    """

    # Box area range is a concept needed for benchmarking models, we do not need that.
    # Thus, we pick one that can just fit any prediction.
    AREA_RANGE = np.array([0**2, 1e5**2])
    AREA_RANGE_LABEL = "all"
    # Maximum number of predictions we account for for each image.
    MAX_PREDS = 100

    def __init__(self, foreground_threshold: float):
        self.foreground_threshold = foreground_threshold

    def __call__(
        self, targets_df: pd.DataFrame, preds_df: pd.DataFrame, image_ids: Set[str]
    ) -> float:
        """Calculate mAP

        :param targets_df: DF w/ cols: ["image_id", "xmin", "ymin", "xmax", "ymax", "class_id"]
        :param preds_df: DF w/ cols: ["image_id", "xmin", "ymin", "xmax", "ymax", "class_id", "score"]
        :param image_ids: Set of image IDs in same dtype they are present in the DataFrames
        """
        if len(preds_df) == 0:
            # If there are no predictions (sometimes on error impact), return -1.
            return -1
        targets, preds = self._format_inputs(targets_df, preds_df, image_ids)
        # Silence all the garbage prints that cocotools spits out
        with silencer():
            coco_eval = self._build_coco_eval(targets, preds)
            coco_eval.evaluate()
            coco_eval.accumulate()
            map = self._compute(coco_eval)
        return map

    def _format_inputs(self, targets_df, preds_df, image_ids):
        preds = self._format_box_df_for_cocotools(preds_df, is_preds=True)
        # Targets are expected to carry extra information
        target_anns = self._format_box_df_for_cocotools(targets_df)
        targets = {
            "images": [{"id": id_} for id_ in set(image_ids)],
            "categories": [{"id": cat} for cat in targets_df[CLASS_ID_COL].unique()],
            "annotations": target_anns,
        }
        return targets, preds

    @staticmethod
    def _format_box_df_for_cocotools(box_df: pd.DataFrame, is_preds: bool = False) -> List[Dict]:
        # `box_df` is either a `targets_df` or a `preds_df`
        box_df = box_df.copy()  # Ensure no side effects
        box_df[BOX_WIDTH_COL] = box_df[XMAX_COL] - box_df[XMIN_COL]
        box_df[BOX_HEIGHT_COL] = box_df[YMAX_COL] - box_df[YMIN_COL]

        ann_records = json.loads(box_df.to_json(orient="records"))

        formatted = [
            {
                "id": i,
                "image_id": r[IMAGE_ID_COL],
                "category_id": r[CLASS_ID_COL],
                "bbox": [r[XMIN_COL], r[YMIN_COL], r[BOX_WIDTH_COL], r[BOX_HEIGHT_COL]],
                "iscrowd": False,
                "area": r[BOX_WIDTH_COL] * r[BOX_HEIGHT_COL],
            }
            for i, r in enumerate(ann_records, start=1)
        ]
        if is_preds:
            # preds need a "score" field
            for r, a in zip(ann_records, formatted):
                a["score"] = r[SCORE_COL]
                a.pop("id")
        return formatted

    def _build_coco_eval(self, targets, preds):
        """Build the COCOeval object we need to leverage pycocotools computation"""
        coco_targets = COCO()
        coco_targets.dataset = targets
        coco_targets.createIndex()
        coco_preds = coco_targets.loadRes(preds)
        coco_eval = COCOeval(cocoGt=coco_targets, cocoDt=coco_preds, iouType="bbox")
        coco_eval.params.iouThrs = np.array([self.foreground_threshold])  # Single IoU threshold
        coco_eval.params.areaRng = np.array([self.AREA_RANGE])
        coco_eval.params.areaRngLbl = [self.AREA_RANGE_LABEL]
        # Single maximum number of predictions to account for
        coco_eval.params.maxDets = np.array([self.MAX_PREDS])
        return coco_eval

    def _compute(self, coco_eval):
        """Actual computation of mAP; extracted from non-flexible `COCOeval.summarize` method."""
        p = coco_eval.params

        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == self.AREA_RANGE_LABEL]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == self.MAX_PREDS]
        # dimension of precision: [TxRxKxAxM]
        s = coco_eval.eval["precision"]
        # IoU
        t = np.where(self.foreground_threshold == p.iouThrs)[0]
        s = s[t]
        s = s[:, :, :, aind, mind]
        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])
        return mean_s

class CalculateMetricsCallback(TrainerCallback):

    def __init__(self, targets_df, eval_image_ids):
        self.evaulator = COCOMeanAveragePrecision(0.3)
        self.targets_df = targets_df
        self.image_ids = eval_image_ids

    def on_eval_epoch_end(self, trainer, **kwargs):
        map = self.evaulator(targets_df=trainer.targets_df, preds_df=trainer.preds_df, image_ids=self.image_ids)

        trainer.run_history.update_metric(f"map", map)
