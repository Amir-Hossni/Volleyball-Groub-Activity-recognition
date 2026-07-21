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

    writer = SummaryWriter(
        log_dir=log_dir
    )

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