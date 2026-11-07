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

### Patient-level evaluation protocol

Both prognostic scripts use **outer stratified K-fold cross-validation** with a
separate internal validation split. Classification uses 6 outer folds and 100
epochs per fold; regression uses 10 outer folds and 150 epochs per fold. Neither
script uses early stopping. The existing five seeds for each task are retained.

For each outer fold, its test patients are held aside. The remaining patients
are stratified by the binary label into 80% training and 20% internal validation.
Only internal validation selects the checkpoint (classification: ACC by default,
configurable through `MONITOR_MODE`; regression: C-index). After all epochs have
finished, the selected checkpoint is loaded and evaluated on the outer test fold.
Test metrics must not be used to tune the protocol or choose epochs/seeds.
All slides/patches from one patient must belong to that patient's single bag.
Duplicate patient IDs are rejected, and all outer test folds are retained.

Each seed saves three-way patient split manifests, validation histories, actual
and selected epoch numbers, outer-test predictions, and result JSON files. A
patient receives exactly one outer-test prediction per seed. New `outercv_v3`
checkpoint directories are separate from historical validation-selected runs;
old checkpoint files are not a substitute for training under the new splits.
Regression fails explicitly if internal validation has no comparable survival
pairs; it does not silently replace its selection metric with 0.5. An undefined
outer-fold C-index is recorded as null, with the number of defined folds reported.

The primary seed score is the arithmetic mean of its defined outer-fold metrics.
Across-seed summaries use sample standard deviation (`ddof=1`). Pooled OOF AUROC
and pooled within-fold-percentile C-index are also retained as separately named
secondary statistics; they are not interchangeable with the fold means.

Run the unit and synthetic CPU integration tests from the repository root:

```bash
python -m unittest discover -s tests -v
```
