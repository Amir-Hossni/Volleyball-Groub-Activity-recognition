import io
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from torch.utils.tensorboard import SummaryWriter


def create_writer(log_dir):
    """
    Create TensorBoard writer

    Args:
        log_dir (str):
            folder where TensorBoard logs will be saved

    Returns:
        SummaryWriter
    """

    # Resolve to an absolute path so event files are written to a fixed,
    # discoverable location regardless of the current working directory.
    # A bare relative "runs/..." path is relative to the process CWD, which
    # can differ between training and later inspection (e.g. on Kaggle).
    log_dir = str(Path(log_dir).resolve())

    writer = SummaryWriter(
        log_dir=log_dir
    )

    # Print the exact directory so TensorBoard can be pointed at it directly.
    print(f"TensorBoard event files will be written to: {writer.log_dir}")

    return writer


def log_metrics(writer, metrics, epoch):
    """
    Log training metrics to TensorBoard

    Args:
        writer:
            TensorBoard writer

        metrics (dict):
            dictionary containing metric names and values

        epoch (int):
            current epoch number
    """

    for name, value in metrics.items():

        writer.add_scalar(
            name,
            value,
            epoch
        )

    writer.flush()


def log_confusion_matrix(
    writer,
    figure,
    tag,
    epoch,
    close_figure=True,
):
    """
    Log a matplotlib confusion-matrix figure to TensorBoard as an image.

    Args:
        writer:
            TensorBoard writer

        figure:
            matplotlib Figure returned by save_confusion_matrix(save_path=None)

        tag (str):
            image tag, e.g. "ConfusionMatrix/val"

        epoch (int):
            current epoch number

        close_figure (bool):
            close the figure after logging to free memory

    Note:
        matplotlib and numpy are imported lazily here so that the core
        scalar logging workflow (create_writer/log_metrics) keeps working
        even when matplotlib is not installed.
    """


    matplotlib.use("Agg")


    buf = io.BytesIO()
    figure.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)

    # Convert PNG bytes to a numpy HxWxC uint8 array for add_image.
    image = plt.imread(buf, format="png")

    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)

    # add_image expects CHW
    image = np.transpose(image, (2, 0, 1))

    writer.add_image(
        tag,
        image,
        global_step=epoch,
        dataformats="CHW",
    )
    writer.flush()
    buf.close()

    if close_figure:
        plt.close(figure)