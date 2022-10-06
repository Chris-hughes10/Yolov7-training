from pathlib import Path

from pytorch_accelerated.callbacks import TrainerCallback
from pytorch_accelerated.utils import local_process_zero_only

from yolov7.plotting import annotate_image


def intersect_dicts(da, db, exclude=()):
    # Dictionary intersection of matching keys and shapes, omitting 'exclude' keys, using da values
    return {
        k: v
        for k, v in da.items()
        if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape
    }


class SaveFirstBatchCallback(TrainerCallback):
    def __init__(self, output_path, num_images_per_batch=1, num_epochs=1):
        self.output_path = Path(output_path)
        self.num_images_per_batch = num_images_per_batch
        self.num_epochs = num_epochs

    @local_process_zero_only
    def on_eval_step_end(self, trainer, batch, batch_output, **kwargs):
        if trainer.run_history.current_epoch == 1:
            images, labels, image_idxs, original_image_sizes = (
                batch[0],
                batch[1],
                batch[2],
                batch[3].cpu(),
            )

            for idx in range(self.num_images_per_batch):
                image = images[idx].permute(1, 2, 0).clone().detach().cpu()

                boxes = labels[labels[:, 0] == idx][:, 2:].clone().detach().cpu()
                boxes[:, [0, 2]] *= image.shape[0]
                boxes[:, [1, 3]] *= image.shape[1]

                fig = annotate_image(image, boxes.tolist(), "cxcywh")
                out_path = (
                    self.output_path
                    / f"epoch_{trainer.run_history.current_epoch}/eval/"
                )
                out_path.mkdir(exist_ok=True, parents=True)

                fig.savefig(out_path / f"image_{idx}.jpg")
