# Reproducing the experiments

Everything in the paper was produced by the two notebooks in `notebooks/`, run on Kaggle with the **GPU T4 × 2** accelerator. Colab works with the same notebooks. Every fold is saved the moment it finishes, so a long stage can be spread over several sessions.

## What you need

| Input | Where it comes from |
|---|---|
| Gleason 2019 data | [Challenge site](https://gleason2019.grand-challenge.org/). Keep the folders `imagesTr`, `imagesVal`, `Generated_Labels_Train`, `labelsVal` and `annotations` (with `Map1` to `Map6`). |
| `mobile_sam.pt` | [MobileSAM repository](https://github.com/ChaoningZhang/MobileSAM) |
| The `frozense` package | `python scripts/make_package.py` writes `FrozenSE_SAM_Revision_Package.zip` |
| Hugging Face token (UNI only) | Request access to [MahmoodLab/UNI](https://huggingface.co/MahmoodLab/UNI), then create a read token |

## Setting up on Kaggle

1. Create one dataset holding the Gleason 2019 folders (zipped or not) and `mobile_sam.pt`.
2. Create a second dataset holding `FrozenSE_SAM_Revision_Package.zip`.
3. Import `notebooks/FrozenSE_SAM_Revision.ipynb` (File → Import Notebook) and attach both datasets under **Add Input**.
4. Set the accelerator to **GPU T4 × 2** and turn internet **on**.
5. For UNI: Add-ons → Secrets → add a secret with the label **`HF_TOKEN`** and tick it for the notebook. Without it, UNI is skipped and everything else still runs.

## Running the main notebook

The first code cell holds one setting, `STAGE`. Set it, then use **Save Version → Save & Run All**. Everything outside the chosen stage is skipped.

| `STAGE` | What it runs | GPU hours (T4), approx. |
|---|---|---|
| `'setup'` | Data checks, label audit, consensus analysis, split preview | 0 |
| `'smoke'` | A short fold-0 run that checks the pipeline | under 0.5 |
| `'baselines'` | Fully frozen design, both U-Nets, zero-shot MobileSAM; 6 folds each | about 15 |
| `'improve'` | The final FrozenSE-SAM (chosen on validation) and both U-Nets under the same protocol; 6 folds | about 33 |
| `'ablations'` | One-change ablations on folds 0, 2 and 4 | about 30 |
| `'data_efficiency'` | Validation learning curve (exploratory) | about 4 |
| `'extra'` | 5-core experiment (3 draws × 2 models × 6 folds), FrozenSE-Phikon and FrozenSE-UNI | about 50 |
| `'extra'` with `UNI_ABLATION = True` | Only the two UNI component ablations (linear head; fully frozen encoder), six folds each | about 18 |
| `'report'` | Every table from the saved results; trains nothing | 0 |

Run the stages in that order. With two GPUs, session time is about half the GPU hours.

**Continuing across sessions.** A session stops by itself before Kaggle's 12-hour limit and packages its results as `frozense_runs.zip`. For the next session, attach the previous run's output (Add Input → Your Work → the notebook), keep the same `STAGE`, and run again. Saved folds are loaded, not retrained. The log line `N saved fold results available from earlier sessions` confirms the results were found.

## Running nnU-Net

Import `notebooks/FrozenSE_SAM_nnUNet.ipynb`, attach the same two datasets, and use **Save Version → Save & Run All**. There is no stage to set. The notebook:

- exports, for each fold, exactly the patches the other models use;
- fixes nnU-Net's internal split to the same training and validation patches, so the test slide never enters nnU-Net;
- trains the self-configured 2D nnU-Net for 50 epochs (`NNUNET_EPOCHS`) and predicts the test patches without test-time augmentation;
- scores the predictions with the same metric code as every other model.

All six folds take about 8.7 hours on two T4 GPUs. The log prints one line per finished fold and the pooled Dice at the end.

## From runs to tables and figures

Download `frozense_runs.zip` from each notebook's Output tab, unpack both into one folder, and run:

```bash
python analysis/compute_statistics.py RUNS_DIR results/tables
python analysis/make_figures.py RUNS_DIR figures
```

`results/raw_runs.zip` in this repository is exactly such a folder, holding the runs reported in the paper.

## Settings behind the reported results

| Setting | Value |
|---|---|
| Split | Leave-one-slide-out; test slide *k*, validation slide *k* + 1, 35 training cores drawn (seed 42) from the other four slides |
| Patches | 512 × 512 at 0.25 µm per pixel; 35 per training core (1,225 per fold), 20 per validation or test core |
| Loss | Tversky (α = 0.4, β = 0.6) + focal (γ = 2; class weights 0.2 tumour, 0.8 background) |
| Optimiser | AdamW, weight decay 10⁻⁵, batch size 6, mixed precision |
| Learning rates | 5 × 10⁻⁵ for pretrained weights, 5 × 10⁻⁴ for new weights |
| Schedule | ReduceLROnPlateau (factor 0.2, patience 4) on validation Dice; early stopping patience 10; at most 30 epochs |
| Fine-tuned encoder part | MobileSAM: stage 4 and neck. Phikon and UNI: last two blocks and final norm |
| 5-core experiment | 5 cores × 245 patches (1,225 per epoch); core draws with seeds 42, 1042, 2042 |
| nnU-Net | v2.6.4, 2D, 50 epochs × 250 iterations, no test-time augmentation |
| Software | Python 3.13, PyTorch 2.11, segmentation_models_pytorch 0.6.0, albumentations 2.0.8, timm 1.0.30 |
