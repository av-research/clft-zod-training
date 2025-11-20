import json
import uuid
from pathlib import Path
from datetime import datetime
import math


def clean_nan_values(obj):
    """
    Recursively replace NaN values with empty strings in nested dictionaries/lists.

    Args:
        obj: The object to clean (dict, list, or primitive)

    Returns:
        The cleaned object with NaN values replaced
    """
    if isinstance(obj, dict):
        return {key: clean_nan_values(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan_values(item) for item in obj]
    elif isinstance(obj, float) and math.isnan(obj):
        return 0.0
    else:
        return obj


def log_epoch_results(epoch, training_uuid, results, log_dir, learning_rate=None, epoch_time=None, system_info=None, vision_training_id=None):
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
        system_info (dict, optional): System resource usage snapshot
        vision_training_id (str, optional): Vision service training ID
    
    Returns:
        str: Path to the logged file
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
    if system_info is not None:
        # Store system_info in results to ensure it's sent to vision service
        data["results"]["system_info"] = system_info

    # Clean NaN values from the data before writing
    data = clean_nan_values(data)

    # Create epochs subfolder
    log_dir = Path(log_dir)
    epochs_dir = log_dir / "epochs"
    epochs_dir.mkdir(parents=True, exist_ok=True)

    filename = f"epoch_{epoch}_{epoch_uuid}.json"
    filepath = epochs_dir / filename

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Epoch {epoch} results logged to {filepath}")
    
    # Send to vision service if available
    if vision_training_id:
        from integrations.vision_service import send_epoch_results_from_file
        success = send_epoch_results_from_file(vision_training_id, epoch, str(filepath))
        if success:
            print(f"Sent epoch {epoch} results to vision service")
        else:
            print(f"Failed to send epoch {epoch} results to vision service")
    
    return str(filepath)


def generate_training_uuid():
    """Generate a unique UUID for a training run."""
    return str(uuid.uuid4())


def log_epoch(epoch, train_loss, val_loss, val_metrics, log_dir, training_uuid, 
              epoch_uuid, vision_training_id=None):
    """
    Simplified epoch logging for DeepLabV3+ training.
    
    Args:
        epoch (int): Current epoch number
        train_loss (float): Training loss
        val_loss (float): Validation loss
        val_metrics (dict): Validation metrics (miou, accuracy, etc.)
        log_dir (str): Directory to save logs
        training_uuid (str): Training run UUID
        epoch_uuid (str): This epoch's UUID
        vision_training_id (str, optional): Vision service training ID
    
    Returns:
        str: Path to the logged file
    """
    data = {
        "training_uuid": training_uuid,
        "epoch_uuid": epoch_uuid,
        "epoch": epoch,
        "timestamp": datetime.now().isoformat(),
        "train_loss": train_loss,
        "validation_loss": val_loss,
        "validation_metrics": val_metrics
    }
    
    if vision_training_id:
        data["vision_training_id"] = vision_training_id
    
    # Clean NaN values
    data = clean_nan_values(data)
    
    # Create epochs subfolder
    log_dir = Path(log_dir)
    epochs_dir = log_dir / "epochs"
    epochs_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"epoch_{epoch}_{epoch_uuid}.json"
    filepath = epochs_dir / filename
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    # Send to vision service if available
    if vision_training_id:
        from integrations.vision_service import send_epoch_results_from_file
        send_epoch_results_from_file(vision_training_id, epoch, str(filepath))
    
    return str(filepath)