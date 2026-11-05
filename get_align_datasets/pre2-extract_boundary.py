import os
import subprocess
from types import SimpleNamespace
import tarfile

from config import get_config 


combined_prefix, config = get_config(file_path='../config.txt', script_name='pre2-extract_boundary.py')


os.chdir(config.code_dir)

# !pip install opencv-python
# !pip install natsort

command = f"""
python 1_get_xenium_nuclei_seg_image.py \
    --fp_boundaries {config.dir_xenium_outs}/nucleus_boundaries.csv.gz \
    --dir_output {config.dir_output} \
    --fp_he_img {config.fp_he_img} \
    --n_processes {config.n_processes} \
    --changepath {combined_prefix}
"""
os.system(command)