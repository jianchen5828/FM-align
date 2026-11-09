import os
import subprocess
from types import SimpleNamespace
import tarfile

from config import get_config 


combined_prefix, config = get_config(file_path='../config.txt', script_name='pre3-genes_matrix.py')


os.chdir(config.code_dir)

# cell_feature_matrix.tar.gz
tar_path = os.path.join(combined_prefix, "cell_feature_matrix.tar.gz")
extracted_folder = os.path.join(combined_prefix, "cell_feature_matrix")

if os.path.exists(extracted_folder):
    print(f"cell_feature_matrix folder already exists")
elif os.path.exists(tar_path):
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=combined_prefix)
    print("cell_feature_matrix extracted Done！！")
else:
    print("Neither 'cell_feature_matrix.tar.gz' nor the extracted folder were found")

# !pip install opencv-python
# !pip install natsort

command = f"""
python 2_get_xenium_cell_gene_matrix.py \
    --dir_feature_matrix {config.dir_xenium_outs}/cell_feature_matrix \
    --dir_output {config.dir_output}
"""
os.system(command)