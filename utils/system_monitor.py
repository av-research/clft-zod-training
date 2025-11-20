#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
System monitoring utilities for tracking resource usage during training.
"""
import torch
import psutil
import platform
from datetime import datetime


def get_system_info():
    """
    Get comprehensive system information including CPU, GPU, and memory usage.
    
    Returns:
        dict: System information including hardware specs and current usage
    """
    system_info = {
        'timestamp': datetime.now().isoformat(),
        'platform': {
            'system': platform.system(),
            'release': platform.release(),
            'machine': platform.machine(),
            'python_version': platform.python_version()
        },
        'cpu': {
            'count': psutil.cpu_count(logical=True),
            'count_physical': psutil.cpu_count(logical=False),
            'usage_percent': psutil.cpu_percent(interval=1),
            'per_cpu_percent': psutil.cpu_percent(interval=1, percpu=True)
        },
        'memory': {
            'total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
            'available_gb': round(psutil.virtual_memory().available / (1024**3), 2),
            'used_gb': round(psutil.virtual_memory().used / (1024**3), 2),
            'percent': psutil.virtual_memory().percent
        },
        'gpu': {}
    }
    
    # Add GPU information if CUDA is available
    if torch.cuda.is_available():
        gpu_info = {}
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_props = torch.cuda.get_device_properties(i)
            
            # Get memory info
            total_memory = gpu_props.total_memory / (1024**3)  # Convert to GB
            reserved_memory = torch.cuda.memory_reserved(i) / (1024**3)
            allocated_memory = torch.cuda.memory_allocated(i) / (1024**3)
            free_memory = total_memory - allocated_memory
            
            gpu_info[f'gpu_{i}'] = {
                'name': gpu_name,
                'total_memory_gb': round(total_memory, 2),
                'allocated_memory_gb': round(allocated_memory, 2),
                'reserved_memory_gb': round(reserved_memory, 2),
                'free_memory_gb': round(free_memory, 2),
                'utilization_percent': round((allocated_memory / total_memory) * 100, 2),
                'compute_capability': f"{gpu_props.major}.{gpu_props.minor}",
                'multi_processor_count': gpu_props.multi_processor_count
            }
        
        system_info['gpu'] = gpu_info
        system_info['cuda_version'] = torch.version.cuda
    else:
        system_info['gpu'] = {'available': False, 'message': 'CUDA not available'}
    
    return system_info


def get_epoch_system_snapshot():
    """
    Get a lightweight snapshot of system usage for epoch logging.
    
    Returns:
        dict: Current system resource usage
    """
    mem = psutil.virtual_memory()
    
    # Get current process info
    process = psutil.Process()
    
    snapshot = {
        'cpu_percent': psutil.cpu_percent(interval=0.1),
        'cpu_count': psutil.cpu_count(logical=True),
        'cpu_count_physical': psutil.cpu_count(logical=False),
        'memory_percent': mem.percent,
        'memory_used_gb': round(mem.used / (1024**3), 2),
        'memory_max_gb': round(mem.total / (1024**3), 2),
        'process': {
            'cpu_percent': process.cpu_percent(interval=0.1),
            'memory_gb': round(process.memory_info().rss / (1024**3), 2),
            'threads': process.num_threads()
        }
    }
    
    # Add GPU info if available
    if torch.cuda.is_available():
        gpu_info = {}
        
        # Try to get GPU stats using nvidia-ml-py if available
        try:
            import pynvml
            pynvml.nvmlInit()
            use_nvml = True
        except (ImportError, Exception):
            use_nvml = False
        
        for i in range(torch.cuda.device_count()):
            total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            allocated_memory = torch.cuda.memory_allocated(i) / (1024**3)
            reserved_memory = torch.cuda.memory_reserved(i) / (1024**3)
            
            gpu_data = {
                'memory_used_gb': round(allocated_memory, 2),
                'memory_max_gb': round(total_memory, 2),
                'memory_reserved_gb': round(reserved_memory, 2),
                'memory_utilization_percent': round((allocated_memory / total_memory) * 100, 2)
            }
            
            # Add additional GPU metrics if pynvml is available
            if use_nvml:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    
                    # GPU utilization
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_data['gpu_utilization_percent'] = util.gpu
                    gpu_data['memory_bandwidth_percent'] = util.memory
                    
                    # Temperature
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    gpu_data['temperature_celsius'] = temp
                    
                    # Power usage
                    try:
                        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
                        power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
                        gpu_data['power_watts'] = round(power, 1)
                        gpu_data['power_limit_watts'] = round(power_limit, 1)
                        gpu_data['power_percent'] = round((power / power_limit) * 100, 1)
                    except:
                        pass
                    
                    # Clock speeds
                    try:
                        sm_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
                        mem_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
                        gpu_data['clock_sm_mhz'] = sm_clock
                        gpu_data['clock_memory_mhz'] = mem_clock
                    except:
                        pass
                    
                    # Fan speed
                    try:
                        fan = pynvml.nvmlDeviceGetFanSpeed(handle)
                        gpu_data['fan_speed_percent'] = fan
                    except:
                        pass
                        
                except Exception as e:
                    gpu_data['nvml_error'] = str(e)
            
            gpu_info[f'gpu_{i}'] = gpu_data
        
        if use_nvml:
            try:
                pynvml.nvmlShutdown()
            except:
                pass
        
        snapshot['gpu'] = gpu_info
    
    return snapshot


def print_system_info(system_info=None):
    """
    Print formatted system information.
    
    Args:
        system_info (dict, optional): System info dict. If None, will call get_system_info()
    """
    if system_info is None:
        system_info = get_system_info()
    
    print("\n" + "="*60)
    print("SYSTEM INFORMATION")
    print("="*60)
    
    # Platform
    print(f"\nPlatform: {system_info['platform']['system']} {system_info['platform']['release']}")
    print(f"Machine: {system_info['platform']['machine']}")
    print(f"Python: {system_info['platform']['python_version']}")
    
    # CPU
    cpu = system_info['cpu']
    print(f"\nCPU:")
    print(f"  Physical cores: {cpu['count_physical']}")
    print(f"  Logical cores: {cpu['count']}")
    print(f"  Usage: {cpu['usage_percent']:.1f}%")
    
    # Memory
    mem = system_info['memory']
    print(f"\nMemory:")
    print(f"  Total: {mem['total_gb']:.2f} GB")
    print(f"  Used: {mem['used_gb']:.2f} GB ({mem['percent']:.1f}%)")
    print(f"  Available: {mem['available_gb']:.2f} GB")
    
    # GPU
    if system_info['gpu'] and system_info['gpu'].get('available') != False:
        print(f"\nGPU (CUDA {system_info.get('cuda_version', 'N/A')}):")
        for gpu_id, gpu_data in system_info['gpu'].items():
            print(f"  {gpu_id}: {gpu_data['name']}")
            print(f"    Memory: {gpu_data['allocated_memory_gb']:.2f}/{gpu_data['total_memory_gb']:.2f} GB "
                  f"({gpu_data['utilization_percent']:.1f}% used)")
            print(f"    Compute: {gpu_data['compute_capability']}, "
                  f"MPs: {gpu_data['multi_processor_count']}")
    else:
        print("\nGPU: Not available")
    
    print("="*60 + "\n")


if __name__ == '__main__':
    # Test the functions
    print("Testing system monitoring utilities...")
    
    # Get and print full system info
    sys_info = get_system_info()
    print_system_info(sys_info)
    
    # Get epoch snapshot
    print("\nEpoch snapshot:")
    snapshot = get_epoch_system_snapshot()
    import json
    print(json.dumps(snapshot, indent=2))
