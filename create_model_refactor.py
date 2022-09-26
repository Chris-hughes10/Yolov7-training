from yolov7.models.yolo import Yolov7Model
import torch

if __name__ == "__main__":
    # model_config = get_yolov7_config()

    # model, _ = parse_model(model_config, [3])

    model = Yolov7Model()

    model.load_state_dict(torch.load("yolov7_training_state_dict.pt"), strict=True)

    print("done")
