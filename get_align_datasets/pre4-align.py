import os
import subprocess
from types import SimpleNamespace
import tarfile

from config import get_config 


combined_prefix, config = get_config(script_name='pre4-align.py')

# Change working directory
os.chdir(config.code_dir)

# !pip install opencv-python
# !pip install natsort

command = f"""
python 4_get_corresponding_cells.py \
    --dir_output {config.dir_output} \
    --n_processes {config.n_processes} \
    --changepath {combined_prefix} \
    --min_overlap_fraction {config.threshold}
"""
os.system(command)