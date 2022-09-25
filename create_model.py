from pathlib import Path
import yaml

from yolov7.migrated.models.yolo import Model

if __name__ == "__main__":
    yolov7_config_path = Path("model_configs/yolov7.yaml")

    with open(yolov7_config_path, "r") as file:
        model_cfg = yaml.safe_load(file)

    model = Model(model_cfg, ch=3, nc=80, anchors=None)

    print("here")
