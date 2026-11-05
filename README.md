# FM-align

Code for **FM-align: an Interpretable Functional–Morphological Cross-Modal Alignment Framework for Cancer Prognosis**.

## Before running

Prepare the input data, required checkpoints, and external segmentation tools. Configure input/output paths, compatible checkpoint paths, and `DEVICE` in the scripts before execution.

Each command block below starts from the repository root.

## Data preprocessing

Prepare `config.txt` in the repository root for the preprocessing scripts.

```bash
cd get_align_datasets
python pre1-hover_cut.py
python pre2-extract_boundary.py
python pre3-genes_matrix.py
python pre4-align.py
cd ..
```

## Cell preparation and alignment

Prepare `config.txt` in `align_and_cellular_context/` for the alignment data.

```bash
cd align_and_cellular_context
python genes-get_all_maskAndcell.py
python he-all_get_cell.py
python align.py
cd ..
```

## Cell features and context training

Set the current dataset paths in `align_and_cellular_context/config.txt` and the checkpoint path in the feature-extraction script. Extract features for each dataset, then configure the dataset paths in the context-training script.

```bash
cd align_and_cellular_context
python context1-getcellandvec.py
python context2-context.py
cd ..
```

## Prognostic feature preparation

Configure the slide, segmentation, cell-image, checkpoint, and feature-output paths in the corresponding scripts.

```bash
cd prognostic_prediction
python 1-get_he.py
python 2-get_cellmask.py
python 3-get_cell.py
python 4-getHevec.py
python 5-batch_extract_cls.py
cd ..
```

## Prognostic prediction

Prepare the patch-feature files and cohort CSV, then set `csv_path`, `feat_root`, and `DEVICE` in the corresponding script.

```bash
cd prognostic_prediction

# Classification
python 6-Classification.py

# Survival regression
python 7-Regression.py
cd ..
```
