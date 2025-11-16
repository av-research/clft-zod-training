import requests
import json
from datetime import datetime
import uuid

VISION_API_BASE_URL = "https://vision-api.tumbaland.eu/api"

def create_training(uuid, name, model, dataset, description=None, status="running", tags=None, config_id=None):
    """
    Create a new training run in the vision service.
    
    Args:
        uuid (str): Unique identifier for the training
        name (str): Name of the training run
        model (str): Model type (e.g., 'clft', 'clfcn')
        dataset (str): Dataset name (e.g., 'zod', 'waymo')
        description (str, optional): Description of the training
        status (str, optional): Initial status, defaults to 'pending'
        tags (list, optional): List of tags for the training
        config_id (str, optional): ID of the associated config
    
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
    
    if config_id:
        payload["configId"] = config_id
    
    if tags:
        payload["tags"] = tags
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get("success"):
            training_id = data.get("data", {}).get("_id")
            return training_id
        else:
            print(f"Failed to create training: {data.get('message')}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Request failed when creating training: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Failed to parse response JSON: {e}")
        return None

def send_epoch_results_from_file(training_id, epoch_num, results_file_path):
    """
    Send epoch results from a logged JSON file to the vision service.
    
    Args:
        training_id (str): ID of the training run
        epoch_num (int): Epoch number
        results_file_path (str): Path to the JSON file containing epoch results
    
    Returns:
        bool: True if successful, False if failed
    """
    try:
        # Read the logged JSON file
        with open(results_file_path, 'r') as f:
            payload = json.load(f)
        
        # Update the training_uuid to match the training_id
        payload["training_uuid"] = training_id
        payload["trainingId"] = training_id
        
        url = f"{VISION_API_BASE_URL}/epochs/upload"
        
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get("success"):
            return True
        else:
            print(f"Failed to send epoch results: {data.get('message')}")
            return False
    except FileNotFoundError:
        print(f"Results file not found: {results_file_path}")
        return False
    except json.JSONDecodeError as e:
        print(f"Failed to parse results JSON file: {e}")
        return False
    except requests.exceptions.RequestException as e:
        print(f"Request failed when sending epoch results: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error when sending epoch results: {e}")
        return False

def create_config(name, config_data, config_uuid=None):
    """
    Create a new configuration in the vision service.
    
    Args:
        name (str): Name of the configuration
        config_data (dict): The configuration data
        config_uuid (str, optional): UUID for the config, auto-generated if not provided
    
    Returns:
        str or None: Config ID if successful, None if failed
    """
    if config_uuid is None:
        config_uuid = str(uuid.uuid4())
    
    # Use the upload endpoint for config data
    url = f"{VISION_API_BASE_URL}/configs/upload"
    
    # Extract summary from config
    summary = config_data.get('Summary', f"Training config for {name}")
    
    # Extract config name from the name parameter (remove extra parts)
    config_name = name.replace(' Config', '').split(' - ')[-1]  # Extract just the config part
    
    payload = {
        "config_data": config_data,
        "Summary": summary,
        "config_name": config_name
    }
    
    try:
        print(f"Creating config with config_name={config_name}, Summary={summary}")
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        if data.get("success"):
            config_id = data.get("data", {}).get("_id")
            return config_id
        else:
            print(f"Failed to create config: {data.get('message')}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Request failed when creating config: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status: {e.response.status_code}")
            print(f"Response body: {e.response.text}")
        return None
    except json.JSONDecodeError as e:
        print(f"Failed to parse response JSON: {e}")
        return None

def send_test_results_from_file(results_file_path):
    """
    Send test results from a logged JSON file to the vision service.
    
    Args:
        results_file_path (str): Path to the JSON file containing test results
    
    Returns:
        bool: True if successful, False if failed
    """
    try:
        # Read the logged JSON file
        with open(results_file_path, 'r') as f:
            payload = json.load(f)
        
        url = f"{VISION_API_BASE_URL}/test-results/upload"
        
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        if data.get("success"):
            print(f"Successfully uploaded test results to vision service")
            return True
        else:
            print(f"Failed to upload test results: {data.get('message')}")
            return False
    except FileNotFoundError:
        print(f"Test results file not found: {results_file_path}")
        return False
    except json.JSONDecodeError as e:
        print(f"Failed to parse test results JSON file: {e}")
        return False
    except requests.exceptions.RequestException as e:
        print(f"Request failed when uploading test results: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error when uploading test results: {e}")
        return False
