import requests
import json
from datetime import datetime

VISION_API_BASE_URL = "https://vision-api.tumbaland.eu/api"

def create_training(uuid, name, model, dataset, description=None, status="running"):
    """
    Create a new training run in the vision service.
    
    Args:
        uuid (str): Unique identifier for the training
        name (str): Name of the training run
        model (str): Model type (e.g., 'clft', 'clfcn')
        dataset (str): Dataset name (e.g., 'zod', 'waymo')
        description (str, optional): Description of the training
        status (str, optional): Initial status, defaults to 'pending'
    
    Returns:
        str or None: Training ID if successful, None if failed
    """
    url = f"{VISION_API_BASE_URL}/trainings"
    
    payload = {
        "uuid": uuid,
        "status": status,
        "name": name,
        "model": model,
        "dataset": dataset
    }
    
    if description:
        payload["description"] = description
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get("success"):
            return data.get("data", {}).get("_id")
        else:
            print(f"Failed to create training: {data.get('message')}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Request failed when creating training: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Failed to parse response JSON: {e}")
        return None

def send_epoch_results(training_id, epoch_num, results_json):
    """
    Send epoch results to the vision service.
    
    Args:
        training_id (str): ID of the training run
        epoch_num (int): Epoch number
        results_json (dict): Dictionary containing epoch results/metrics
    
    Returns:
        bool: True if successful, False if failed
    """
    url = f"{VISION_API_BASE_URL}/epochs/upload"
    
    # For the upload endpoint, send the full results_json with trainingId and epoch added
    payload = results_json.copy()
    payload["trainingId"] = training_id
    payload["epoch"] = epoch_num
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get("success"):
            return True
        else:
            print(f"Failed to send epoch results: {data.get('message')}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"Request failed when sending epoch results: {e}")
        return False
    except json.JSONDecodeError as e:
        print(f"Failed to parse response JSON: {e}")
        return False
