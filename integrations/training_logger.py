import json
import uuid
from pathlib import Path
from datetime import datetime


def log_epoch_results(epoch, training_uuid, results, log_dir, learning_rate=None, epoch_time=None):
    """
    Log training and validation results for a specific epoch.

    Args:
        epoch (int): Current epoch number
        training_uuid (str): UUID for the entire training run
        results (dict): Dictionary containing training and validation results per class
                       Expected structure: {
                           "train": {"class_name": {"metric": value, ...}, ...},
                           "val": {"class_name": {"metric": value, ...}, ...}
                       }
        log_dir (str or Path): Directory to save the log files
        learning_rate (float, optional): Learning rate for this epoch
        epoch_time (float, optional): Time taken for this epoch in seconds
    """
    epoch_uuid = str(uuid.uuid4())

    data = {
        "training_uuid": training_uuid,
        "epoch_uuid": epoch_uuid,
        "epoch": epoch,
        "timestamp": datetime.now().isoformat(),
        "results": results
    }

    if learning_rate is not None:
        data["learning_rate"] = learning_rate
    if epoch_time is not None:
        data["epoch_time"] = epoch_time

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    filename = f"epoch_{epoch}_{epoch_uuid}.json"
    filepath = log_dir / filename

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Epoch {epoch} results logged to {filepath}")


def generate_training_uuid():
    """Generate a unique UUID for a training run."""
    return str(uuid.uuid4())