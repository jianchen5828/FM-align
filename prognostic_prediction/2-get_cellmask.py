import os
import subprocess
import glob
from pathlib import Path
import argparse
import sys

# ==============================================================================
# 0. Configuration Class
# ==============================================================================
class Config:
    def __init__(self):
        self.dir_processed = "."
        self.dir_output = "./cell_mask0"
        self.dir_hovernet = "./external/hovernet" 
        self.gpu_id = 0
        self.crop_size = 1024

SCRIPT_NAME = "./external/hovernet_cell.py" 

def run_step(step, fp_he_img, fp_tumour_mask=None, fp_final_seg=None):
    config = Config()
    conda_env = "hovernet" 
    slide_name = Path(fp_he_img).stem 
    dir_intermediate_output = os.path.join(config.dir_output, slide_name)
    os.makedirs(dir_intermediate_output, exist_ok=True)
    
    print(f"  - Running step {step} for {slide_name}")

    if step == 1:
        tumour_mask_arg = f"--fp_tumour_mask {fp_tumour_mask}" if fp_tumour_mask else ""
        command = f"""
        python {SCRIPT_NAME} \
            --dir_output {dir_intermediate_output} \
            --fp_he_img {fp_he_img} \
            --dir_hovernet {config.dir_hovernet} \
            --gpu_id {config.gpu_id} \
            --crop_size {config.crop_size} \
            {tumour_mask_arg} \
            --step 1
        """
        os.system(command)
        return
    
    try:
        conda_base = subprocess.check_output("conda info --base", shell=True).decode().strip()
    except subprocess.CalledProcessError:
        print("Error: Unable to find Conda path.")
        sys.exit(1)
        
    final_output_arg = f"--fp_out_seg {fp_final_seg}" if step == 3 else ""
    cmd = f"""
    source {conda_base}/etc/profile.d/conda.sh && \
    conda activate {conda_env} && \
    python {SCRIPT_NAME} \
        --dir_output {dir_intermediate_output} \
        --fp_he_img {fp_he_img} \
        --dir_hovernet {config.dir_hovernet} \
        --gpu_id {config.gpu_id} \
        {final_output_arg} \
        --step {step}
    """
    subprocess.run(cmd, shell=True, executable="/bin/bash")

def main():
    config = Config()
    os.makedirs(config.dir_output, exist_ok=True)
    
    he_dir = "./he0/sp1"
    mask_dir = os.path.join(config.dir_processed, "mask_d4")
    
    he_files = sorted(glob.glob(os.path.join(he_dir, "*.tif")))
    
    if not he_files:
        print(f"Warning: No .tif images found in {he_dir}.")
        return

    print(f"[SP1 - GPU 0] Found {len(he_files)} images, starting processing...")

    for fp_he_img in he_files:
        slide_name = Path(fp_he_img).stem
        final_seg_filename = f"{slide_name}_he_image_nuclei_seg.tif"
        fp_final_seg = os.path.abspath(os.path.join(config.dir_output, final_seg_filename))

        if os.path.exists(fp_final_seg) and os.path.getsize(fp_final_seg) > 1024:
            print(f"Skipping completed task: {slide_name}")
            continue

        print(f"\n" + "="*60)
        print(f"Processing task: {slide_name}")
        
        mask_filename = slide_name + ".tif" 
        fp_tumour_mask = os.path.join(mask_dir, mask_filename)
        
        if not os.path.exists(fp_tumour_mask):
            fp_tumour_mask = None 
        
        fp_he_img = os.path.abspath(fp_he_img)
        if fp_tumour_mask:
            fp_tumour_mask = os.path.abspath(fp_tumour_mask)

        try:
            run_step(1, fp_he_img, fp_tumour_mask)
            run_step(2, fp_he_img)
            run_step(3, fp_he_img, fp_final_seg=fp_final_seg)
        except Exception as e:
            print(f"Error: {slide_name} crashed: {e}")
            continue

if __name__ == "__main__":
    main()