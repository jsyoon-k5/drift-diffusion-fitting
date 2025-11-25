"""
Device information logging utility
Code written by June-Seop Yoon
"""

import os
import psutil
import platform
from datetime import datetime


def save_device_info(log_path: str, verbose=True):
    """
    Save device information (CPU, RAM, GPU) to a text file.
    
    Parameters
    ----------
    log_path : str
        Path to save the device information text file.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    with open(log_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("Device Information Log\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        
        # System info
        f.write("[System Information]\n")
        f.write(f"OS: {platform.system()} {platform.release()}\n")
        f.write(f"Platform: {platform.platform()}\n")
        f.write(f"Processor: {platform.processor()}\n")
        f.write(f"Architecture: {platform.machine()}\n\n")
        
        # CPU info
        f.write("[CPU Information]\n")
        f.write(f"Physical cores: {psutil.cpu_count(logical=False)}\n")
        f.write(f"Logical cores: {psutil.cpu_count(logical=True)}\n")
        f.write(f"CPU frequency: {psutil.cpu_freq().current:.2f} MHz (max: {psutil.cpu_freq().max:.2f} MHz)\n")
        f.write(f"CPU usage: {psutil.cpu_percent(interval=1)}%\n\n")
        
        # RAM info
        ram = psutil.virtual_memory()
        f.write("[RAM Information]\n")
        f.write(f"Total: {ram.total / (1024**3):.2f} GB\n")
        f.write(f"Available: {ram.available / (1024**3):.2f} GB\n")
        f.write(f"Used: {ram.used / (1024**3):.2f} GB ({ram.percent}%)\n\n")
        
        # GPU info (using nvidia-smi if available)
        f.write("[GPU Information]\n")
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total,memory.used,memory.free,driver_version,compute_cap', 
                 '--format=csv,noheader'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                gpu_info = result.stdout.strip().split('\n')
                for i, info in enumerate(gpu_info):
                    parts = [p.strip() for p in info.split(',')]
                    f.write(f"GPU {i}: {parts[0]}\n")
                    f.write(f"  Memory Total: {parts[1]}\n")
                    f.write(f"  Memory Used: {parts[2]}\n")
                    f.write(f"  Memory Free: {parts[3]}\n")
                    f.write(f"  Driver Version: {parts[4]}\n")
                    f.write(f"  Compute Capability: {parts[5]}\n")
            else:
                f.write("No NVIDIA GPU detected or nvidia-smi not available.\n")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            f.write("nvidia-smi not found. No GPU information available.\n")
        
        f.write("\n" + "=" * 60 + "\n")
    
    if verbose:
        print(f"Device information saved to: {log_path}")


if __name__ == "__main__":
    save_device_info("./device_info_log.txt")