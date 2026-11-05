import numpy as np
import tifffile
import os
from PIL import Image
import argparse
import sys
import glob
import natsort
import scipy.io as sio
import re
import csv
import shutil
import cv2
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None


def crop_patches(img, dir_output, crop_size, height_whole, width_whole):

    # Determine top left coords of crops
    height = crop_size
    width = crop_size

    h_starts = list(np.arange(0, height_whole - height, height))
    w_starts = list(np.arange(0, width_whole - width, width))
    coords_starts = [(x, y) for x in h_starts for y in w_starts]
    # print("Num crops:", len(coords_starts))

    for h, w in coords_starts:
        fp_output_patch = dir_output + "/%d_%d.tif" % (h, w)
        patch = img[h : h + height, w : w + width, :]
        tifffile.imwrite(fp_output_patch, patch)

    # Remainder patches vertical strip on the right
    vs_num = height_whole // height
    vs_h_starts = list(np.arange(0, vs_num * height, height))
    vs_w_start = w_starts[-1] + width
    vs_w_starts = vs_num * [vs_w_start]

    coords_starts = list(zip(vs_h_starts, vs_w_starts))
    # print("Num crops right edge:", len(coords_starts))

    for h, w in coords_starts:
        fp_output_patch = dir_output + "/%d_%d.tif" % (h, w)
        patch = img[h : h + height, w:]
        tifffile.imwrite(fp_output_patch, patch)

    # Remainder patches horizontal strip on the bottom
    hs_num = width_whole // width
    hs_w_starts = list(np.arange(0, hs_num * width, width))
    hs_h_start = h_starts[-1] + height
    hs_h_starts = hs_num * [hs_h_start]

    coords_starts = list(zip(hs_h_starts, hs_w_starts))
    # print("Num crops bottom edge:", len(coords_starts))

    for h, w in coords_starts:
        fp_output_patch = dir_output + "/%d_%d.tif" % (h, w)
        patch = img[h:, w : w + width]
        tifffile.imwrite(fp_output_patch, patch)

    # last patch in the bottom right corner
    last_h = hs_h_starts[-1]
    last_w = vs_w_starts[-1]
    fp_output_patch = dir_output + "/%d_%d.tif" % (last_h, last_w)
    patch = img[last_h:, last_w:]
    tifffile.imwrite(fp_output_patch, patch)

    print("Done cropping")

    return height_whole, width_whole


def run_hovernet(dir_hovernet, gpu_id, dir_crops, dir_out_hovernet):
    dir_crops = os.path.abspath(dir_crops)
    dir_out_hovernet = os.path.abspath(dir_out_hovernet)

    print(dir_crops)
    print(dir_out_hovernet)

    dir_crops_temp = dir_crops + "_temp"
    os.makedirs(dir_crops_temp, exist_ok=True)

    fps_crops = glob.glob(dir_crops + "/*.tif")
    print("Num crops found:", len(fps_crops))

    os.chdir(dir_hovernet)

    # 1 crop at a time due to too many files opened error
    for fp in fps_crops:
        dst_file = fp.replace(dir_crops, dir_crops_temp)
        shutil.copy2(fp, dst_file)

        dir_out_hovernet_crop = f"{dir_out_hovernet}/{os.path.basename(fp).replace('.tif','')}"

        test_command = (
            "python run_infer.py --gpu="
            "%d"
            " --nr_types=0 --batch_size=64 --model_mode=original --model_path=hovernet_original_consep_notype_tf2pytorch.tar --nr_inference_workers=1 --nr_post_proc_workers=8 tile --input_dir=%s --output_dir=%s --mem_usage=0.1"
            % (gpu_id, dir_crops_temp, dir_out_hovernet_crop)
        )

        os.system(test_command)

        os.remove(dst_file)

    shutil.rmtree(dir_crops_temp)


def combine_crops(hist_h, hist_w, dir_output, dir_hovernet_output):

    output = np.zeros((hist_h, hist_w), dtype=np.uint32)

    files = glob.glob(dir_hovernet_output + "/**/mat/*.mat")
    files = natsort.natsorted(files)

    fp_out = f"{dir_output}/he_image_nuclei_seg.tif"

    hs_all = []
    ws_all = []

    total_n = 0

    region_ids = []

    # patch_ids = []
    # fp_out_idx = dir_output + "/nuclei_seg_ids.csv"

    for mat_fname in files:
        coords = re.findall(r"\d+", os.path.basename(mat_fname))
        hs = int(coords[0])
        ws = int(coords[1])

        hs_all.append(hs)
        ws_all.append(ws)

        mat_contents = sio.loadmat(mat_fname)
        inst_map = mat_contents["inst_map"]

        he = hs + inst_map.shape[0]
        we = ws + inst_map.shape[1]

        # place into whole segmentation and make nuclei ID unique
        nuclei_mask = np.where(inst_map > 0, 1, 0)
        output[hs:he, ws:we] = inst_map + total_n * nuclei_mask

        unique_ids = np.unique(inst_map)[1:]

        # patch_ids_patch = [f"{hs}_{ws}_{x}" for x in unique_ids]
        # patch_ids.extend(patch_ids_patch)
        region_ids.extend(list(unique_ids + total_n))

        total_n += len(unique_ids)

    print(f"Num IDs before combining: {total_n}")

    region_ids = np.array(region_ids)

    # first set of border pixels
    hs_all_lower = natsort.natsorted(list(set(hs_all)))
    ws_all_right = natsort.natsorted(list(set(ws_all)))

    if 0 in hs_all_lower:
        hs_all_lower.remove(0)
    if 0 in ws_all_right:
        ws_all_right.remove(0)

    # second set of border pixels
    hs_all_upper = [x - 1 for x in hs_all_lower]
    ws_all_left = [x - 1 for x in ws_all_right]

    output2 = np.copy(output)

    print("Combining crops...")

    # borders that run horizontally
    for hsu, hsl in tqdm(zip(hs_all_upper, hs_all_lower), total=len(hs_all_upper)):

        border_a = output[hsu, :].copy()
        border_b = output[hsl, :].copy()

        border_a[border_a > 0] = 1
        border_b[border_b > 0] = 1

        # locations where there is overlap
        overlap = border_a * border_b
        overlap_locs = np.where(overlap > 0)

        # nuclei IDs where there is overlap from a and b - pairs
        overlap_ids_a = output[hsu, overlap_locs[0]]
        overlap_ids_b = output[hsl, overlap_locs[0]]
        # print(list(set(overlap_ids_a) & set(overlap_ids_b)))

        # combine overlapping nuclei
        d = dict(zip(overlap_ids_b, overlap_ids_a))

        for old, new in d.items():
            # check within a distance along the edge
            output2[hsl : hsl + 150, :][output[hsl : hsl + 150, :] == old] = new
            region_ids[region_ids == old] = new

    print(f"Num nuclei intermediate step {len(np.unique(region_ids))}")

    del output

    output3 = np.copy(output2)

    # borders that run vertically
    for hsu, hsl in tqdm(zip(ws_all_left, ws_all_right), total=len(ws_all_left)):

        border_a = output2[:, hsu].copy()
        border_b = output2[:, hsl].copy()

        border_a[border_a > 0] = 1
        border_b[border_b > 0] = 1

        # locations where there is overlap
        overlap = border_a * border_b
        overlap_locs = np.where(overlap > 0)

        # nuclei IDs where there is overlap from a and b - pairs
        overlap_ids_a = output2[overlap_locs[0], hsu]
        overlap_ids_b = output2[overlap_locs[0], hsl]
        # print(list(set(overlap_ids_a) & set(overlap_ids_b)))

        # combine overlapping nuclei
        d = dict(zip(overlap_ids_b, overlap_ids_a))

        for old, new in d.items():
            output3[:, hsl : hsl + 150][output2[:, hsl : hsl + 150] == old] = new
            region_ids[region_ids == old] = new

    print(f"Num nuclei final: {len(np.unique(region_ids))}")
    tifffile.imwrite(fp_out, output3, photometric="minisblack")
    print("Saved", fp_out)

    # also save a downsized copy
    resized_h = int(round(hist_h * 0.2125))
    resized_w = int(round(hist_w * 0.2125))
    output_microns = cv2.resize(
        output3.astype(np.float32),
        (resized_w, resized_h),
        interpolation=cv2.INTER_NEAREST,
    )
    output_microns = output_microns.astype(np.uint32)
    fp_out_microns = fp_out.replace(".tif", "_microns.tif")
    tifffile.imwrite(fp_out_microns, output_microns, photometric="minisblack")

    print(f"Num nuclei final: {len(np.unique(output_microns)) - 1}")


def main(config):
    """
    Segment H&E image with HoverNet:
        Divide H&E image (full resolution) into patches
        Run HoverNet on the patches
        Combine the patches together
    """

    dir_output = os.path.join(config.dir_output, "he_image_nuclei_seg_crops")
    os.makedirs(dir_output, exist_ok=True)

    dir_output_hovernet = dir_output + "_hovernet"

    fp_he_img = config.fp_he_img
    he_img = tifffile.imread(fp_he_img)

    if he_img.shape[-1] == 3:
        pass
    elif he_img.shape[0] == 3:
        he_img = np.moveaxis(he_img, 0, -1)
    else:
        sys.exit("RGB channel not 0 or 2")

    height_whole = he_img.shape[0]
    width_whole = he_img.shape[1]

    if config.step == 1:
        crop_patches(he_img, dir_output, config.crop_size, height_whole, width_whole)

    elif config.step == 2:
        print("Make sure you are using the correct env for hovernet")
        # input("Press Enter to continue or CTRL+C to quit...")
        os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
        run_hovernet(
            config.dir_hovernet, config.gpu_id, dir_output, dir_output_hovernet
        )

    elif config.step == 3:

        combine_crops(height_whole, width_whole, config.dir_output, dir_output_hovernet)

        if config.del_intm_files:
            print("Deleting intermediate files")
            # input("Press Enter to continue or CTRL+C to quit...")

            shutil.rmtree(dir_output)
            shutil.rmtree(dir_output_hovernet)

    else:
        sys.exit(
            "Invalid --step specified, (1=crop H&E, 2=run Hover-Net, 3=combine crops)"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fp_he_img",
        default="he_image.ome.tif",
        type=str,
        help="corresponding full resolution H&E image file path",
    )
    parser.add_argument(
        "--dir_hovernet",
        default="hover_net",
        type=str,
        help="dir of hover_net code",
    )
    parser.add_argument(
        "--crop_size",
        default=2000,
        type=int,
        help="patch size of H&E image",
    )
    parser.add_argument(
        "--gpu_id",
        default=0,
        type=int,
        help="which GPU to use",
    )
    parser.add_argument(
        "--step",
        default=1,
        type=int,
        help="which step (1=crop H&E, 2=run Hover-Net, 3=combine crops)",
    )
    parser.add_argument("--dir_output", default="data_processing", type=str)
    parser.add_argument("--fp_out_seg", default="he_image_nuclei.tif", type=str)
    parser.add_argument(
        "--del_intm_files",
        default=True,
        type=bool,
        help="delete intermediate saved files",
    )
    config = parser.parse_args()
    main(config)
