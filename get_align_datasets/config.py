# filename: config.py
import os
import json
from types import SimpleNamespace

# --- Default configuration values ---
DEFAULT_PREFIX1 = "./data/xenium"
DEFAULT_PREFIX2 = "kidney-Xenium_V1_hKidney_cancer_section"
DEFAULT_CODE_DIR = "./data_processing"
DEFAULT_HE_IMG_NAME = "he_image.tif"
DEFAULT_HOVERNET_DIR = "./external/hover_net"
DEFAULT_GPU_ID = 0
DEFAULT_N_PROCESSES_DEFAULT = 32 # Default value for pre2, pre3, pre4
DEFAULT_N_PROCESSES_PRE1 = 24  # Default value for pre1
DEFAULT_THRESHOLD = 0.8 # Default value for pre4

def load_prefix_data(file_path):
    """
    Reads prefix1 and prefix2 from a JSON file at the specified path.
    (Note: The original script used 'config.txt' and assumed JSON format)
    """
    
    # Path defaults
    prefix1 = DEFAULT_PREFIX1
    prefix2 = DEFAULT_PREFIX2
    
    try:
        with open(file_path, 'r') as f:
            paths = json.load(f)
        prefix1 = paths.get("prefix1", DEFAULT_PREFIX1)
        prefix2 = paths.get("prefix2", DEFAULT_PREFIX2)
    except FileNotFoundError:
        print(f"Configuration file not found ({file_path}), using default paths") 
    except json.JSONDecodeError:
        print(f"Failed to parse configuration file '{file_path}', please check JSON format. Using default paths.")
    
    return prefix1, prefix2

def get_config(script_name, file_path="../config.txt"):
    """
    Loads configuration based on script name and config file path to build a SimpleNamespace object.
    
    Args:
        file_path (str): Relative path to the configuration file (e.g., 'config.txt').
        script_name (str): Script name used to differentiate n_processes and threshold values.
    
    Returns:
        tuple: (combined_prefix, config SimpleNamespace)
    """
    # 1. Read prefix1 and prefix2
    prefix1, prefix2 = load_prefix_data(file_path)
    combined_prefix = os.path.join(prefix1, prefix2)
    
    # 2. Determine n_processes
    n_processes = DEFAULT_N_PROCESSES_PRE1 if script_name == 'pre1-hover_cut.py' else DEFAULT_N_PROCESSES_DEFAULT
        
    # 3. Build config SimpleNamespace
    config = SimpleNamespace(
        code_dir=DEFAULT_CODE_DIR,
        dir_output=os.path.join(combined_prefix, "processed"),
        dir_xenium_outs=combined_prefix, # Nuclei boundaries
        fp_he_img=os.path.join(combined_prefix, DEFAULT_HE_IMG_NAME),
        dir_hovernet=DEFAULT_HOVERNET_DIR,
        gpu_id=DEFAULT_GPU_ID,
        n_processes=n_processes
    )
    
    # 4. pre4 script requires a threshold; add it
    if script_name == 'pre4-align.py':
        config.threshold = DEFAULT_THRESHOLD
        
    return combined_prefix, config