import contextlib
import json
import sys
from pathlib import Path
from typing import List, Dict
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pytorch_accelerated.callbacks import TrainerCallback

from yolov7.models.yolo import filter_predictions

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


class Silencer:
    def __init__(self):
        self.save_stdout = sys.stdout

    def start(self):
        sys.stdout = MagicMock()
        # sys.stdout = DummyFile()

    def stop(self):
        sys.stdout = self.save_stdout

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class COCOMeanAveragePrecision:
    """Mean Average Precision for single IoU threshold based on pycocotools

    :param iou_threshold: IoU threshold for which we want to calculate mAP.
    """

    # Box area range is a concept needed for benchmarking models, we do not need that.
    # Thus, we pick one that can just fit any prediction.
    AREA_RANGE = np.array([0**2, 1e5**2])
    AREA_RANGE_LABEL = "all"
    # Maximum number of predictions we account for each image.
    MAX_PREDS = 100

    def __init__(self, iou_threshold: float = None, verbose=False):
        self.foreground_threshold = iou_threshold
        self.verbose = verbose
        self.silencer = Silencer()

    def __call__(self, targets_json: pd.DataFrame, preds_json) -> float:
        """Calculate mAP from json files"""
        mAP = self.compute(targets_json, preds_json)
        return mAP

    def compute_from_dfs(
        self,
        targets_df: pd.DataFrame,
        preds_df: pd.DataFrame,
    ) -> float:
        """Calculate mAP

        :param targets_df: DF w/ cols: ["image_id", "xmin", "ymin", "xmax", "ymax", "class_id"]
        :param preds_df: DF w/ cols: ["image_id", "xmin", "ymin", "xmax", "ymax", "class_id", "score"]
        :param image_ids: Set of image IDs in same dtype they are present in the DataFrames
        """
        if len(preds_df) == 0:
            # If there are no predictions (sometimes on error impact), return -1.
            return -1

        image_ids = set(targets_df[IMAGE_ID_COL].unique())
        image_ids.update(preds_df[IMAGE_ID_COL].unique())

        targets, preds = self._format_inputs(targets_df, preds_df, image_ids)
        # Silence all the garbage prints that cocotools spits out
        # with silencer():
        map = self.compute(targets, preds)
        return map

    def _format_inputs(self, targets_df, preds_df, image_ids):
        preds = self.format_box_df_for_cocotools(preds_df, is_preds=True)
        # Targets are expected to carry extra information
        target_anns = self.format_box_df_for_cocotools(targets_df)
        targets = {
            "images": [{"id": id_} for id_ in set(image_ids)],
            "categories": [{"id": cat} for cat in targets_df[CLASS_ID_COL].unique()],
            "annotations": target_anns,
        }

        return targets, preds

    @staticmethod
    def format_box_df_for_cocotools(
        box_df: pd.DataFrame, is_preds: bool = False
    ) -> List[Dict]:
        # `box_df` is either a `targets_df` or a `preds_df`
        box_df = box_df.copy()  # Ensure no side effects
        box_df[BOX_WIDTH_COL] = box_df[XMAX_COL] - box_df[XMIN_COL]
        box_df[BOX_HEIGHT_COL] = box_df[YMAX_COL] - box_df[YMIN_COL]
        box_df = box_df.sort_values(
            [IMAGE_ID_COL, CLASS_ID_COL], ascending=[True, True]
        )

        ann_records = json.loads(box_df.to_json(orient="records"))

        formatted = [
            {
                "id": i,
                "image_id": r[IMAGE_ID_COL],
                "category_id": int(r[CLASS_ID_COL]),
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

        if self.foreground_threshold is not None:
            coco_eval.params.iouThrs = np.array(
                [self.foreground_threshold]
            )  # Single IoU threshold
            coco_eval.params.areaRng = np.array([self.AREA_RANGE])
            coco_eval.params.areaRngLbl = [self.AREA_RANGE_LABEL]
            # Single maximum number of predictions to account for
            coco_eval.params.maxDets = np.array([self.MAX_PREDS])
        return coco_eval

    def compute(self, targets_json, preds_json):
        if len(preds_json) == 0:
            # If there are no predictions (sometimes on error impact), return -1.
            return -1
        # Silence all the garbage prints that cocotools spits out
        # with silencer():

        if not self.verbose:
            self.silencer.start()

        coco_eval = self._build_coco_eval(targets_json, preds_json)
        coco_eval.evaluate()
        coco_eval.accumulate()
        if self.foreground_threshold is None:
            coco_eval.summarize()
            mAP = coco_eval.stats[0]
        else:
            mAP = self._compute(coco_eval)

        if not self.verbose:
            self.silencer.stop()

        return mAP

    def _compute(self, coco_eval):
        """Actual computation of mAP; extracted from non-flexible `COCOeval.summarize` method."""
        p = coco_eval.params

        aind = [
            i for i, aRng in enumerate(p.areaRngLbl) if aRng == self.AREA_RANGE_LABEL
        ]
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
    def __init__(
        self,
        targets_json,
        iou_threshold=None,
        save_predictions_output_dir_path=None,
        verbose=False,
        nms_iou_threshold=0.65,
        prediction_confidence_threshold=0.001,
    ):

        self.evaluator = COCOMeanAveragePrecision(iou_threshold)
        self.targets_json = targets_json
        self.verbose = verbose
        self.save_predictions_path = (
            Path(save_predictions_output_dir_path)
            if save_predictions_output_dir_path is not None
            else None
        )
        self.nms_iou_thres = nms_iou_threshold
        self.confidence_threshold = prediction_confidence_threshold

        self.eval_predictions = []
        self.image_ids = set()

    def on_eval_step_end(self, trainer, batch, batch_output, **kwargs):
        predictions = batch_output["predictions"]
        self.update(predictions)

    def remove_seen(self, labels):
        image_ids = labels[:, -1].tolist()

        # remove any image_idx that has already been seen
        # this can arise from distributed training where batch size does not evenly divide dataset
        seen_idx_mask = torch.as_tensor(
            [False if idx not in self.image_ids else True for idx in image_ids]
        )

        if seen_idx_mask.all():
            # no update required as all ids already seen this pass
            return []
        elif seen_idx_mask.any():  # at least one True
            # remove predictions for images already seen this pass
            labels = labels[~seen_idx_mask]

        return labels

    def update(self, predictions):
        predictions = self.remove_seen(predictions)
        filtered_predictions = filter_predictions(
            predictions,
            confidence_threshold=self.confidence_threshold,
            nms_iou_threshold=self.nms_iou_thres,
        )

        if len(filtered_predictions) > 0:
            self.eval_predictions.extend(filtered_predictions.tolist())
            updated_ids = filtered_predictions[:, -1].unique().tolist()
            self.image_ids.update(updated_ids)

    def reset(self):
        self.image_ids = set()
        self.eval_predictions = []

    def on_eval_epoch_end(self, trainer, **kwargs):
        preds_df = pd.DataFrame(
            torch.as_tensor(self.eval_predictions),
            columns=["xmin", "ymin", "xmax", "ymax", "score", "class_id", "image_id"],
        )

        predictions_json = self.evaluator.format_box_df_for_cocotools(
            preds_df, is_preds=True
        )

        if (
            self.save_predictions_path is not None
            and trainer.run_config.is_world_process_zero
        ):
            with open(self.save_predictions_path / "predictions.json", "w") as f:
                json.dump(predictions_json, f)

        if self.verbose and trainer.run_config.is_local_process_zero:
            self.evaluator.verbose = True

        map = self.evaluator.compute(self.targets_json, predictions_json)

        trainer.run_history.update_metric(f"map", map)

        self.reset()
