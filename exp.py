from pathlib import Path
import yaml

from torch import nn
from yolov7.migrated.models.common import *
from yolov7.migrated.models.experimental import *

from itertools import chain

from yolov7.migrated.models.yolo import parse_model
# from yolov7.migrated_refactor.models.yolo import parse_model

if __name__ == "__main__":

    layers = set()

    for model_cfg in Path("model_configs").iterdir():
        with open(model_cfg) as f:
            model_yaml = yaml.load(f, Loader=yaml.SafeLoader)

            print(model_cfg)

        layers.update(
            [
                e
                for e in chain.from_iterable(model_yaml["backbone"])
                if isinstance(e, str)
            ]
        )
        layers.update(
            [e for e in chain.from_iterable(model_yaml["head"]) if isinstance(e, str)]
        )

    print(layers)

    # cfg_path = Path("model_configs/yolov7.yaml")
    # layers = set()
    #
    # with open(cfg_path) as f:
    #     model_yaml = yaml.load(f, Loader=yaml.SafeLoader)  # model dict
    #
    #     model, _ = parse_model(model_yaml, [3])
    #
    #     model.load_state_dict(torch.load('yolov7_training_state_dict.pt'), strict=True)
    #
    #     print("here")
