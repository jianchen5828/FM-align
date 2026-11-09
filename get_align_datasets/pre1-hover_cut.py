import os
import subprocess
from types import SimpleNamespace
import tarfile

from config import get_config 

combined_prefix, config = get_config(file_path='../config.txt', script_name='pre1-hover_cut.py')

os.chdir(config.code_dir)


# 1
command = f"""
python 3_segment_nuclei_he_image.py \
    --dir_output {config.dir_output} \
    --fp_he_img {config.fp_he_img} \
    --dir_hovernet {config.dir_hovernet} \
    --gpu_id {config.gpu_id} \
    --step 1
"""
os.system(command)

# 2
conda_base = subprocess.check_output("conda info --base", shell=True).decode().strip()

cmd = f"""
source {conda_base}/etc/profile.d/conda.sh && \
conda activate hovernet && \
python 3_segment_nuclei_he_image.py \
  --dir_output {config.dir_output} \
  --fp_he_img {config.fp_he_img} \
  --dir_hovernet {config.dir_hovernet} \
  --gpu_id {config.gpu_id} \
  --step 2
"""

subprocess.run(cmd, shell=True, executable="/bin/bash")

# 3
conda_base = subprocess.check_output("conda info --base", shell=True).decode().strip()

cmd = f"""
source {conda_base}/etc/profile.d/conda.sh && \
conda activate GHIST && \
python 3_segment_nuclei_he_image.py \
  --dir_output {config.dir_output} \
  --fp_he_img {config.fp_he_img} \
  --dir_hovernet {config.dir_hovernet} \
  --gpu_id {config.gpu_id} \
  --step 3
"""

subprocess.run(cmd, shell=True, executable=f"/bin/bash")