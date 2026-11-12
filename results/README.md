# Results

| Path | Content |
|---|---|
| `raw_runs.zip` | One folder per experiment, one subfolder per fold. Each `fold_result.json` holds the configuration fingerprint, the training history, the split manifest and the per-core and per-patch test metrics. `nnunet_2d/fold*/result.json` holds the same for nnU-Net. Dataset paths inside the files are anonymised. |
| `tables/main_results.csv` | Pooled Dice, IoU, HD95 and ASD with bootstrap 95% CIs for every model |
| `tables/paired_tests.csv` | Paired Wilcoxon tests on per-core Dice, with Holm correction within each family |
| `tables/patch_accounting.csv` | Tumour-containing and tumour-free patches: correct, missed and false-positive counts |
| `tables/per_slide_dice.csv`, `tables/per_core_dice.csv` | Dice of every model by held-out slide and by core |
| `tables/lowdata_by_draw.csv`, `tables/lowdata_pooled.csv` | The 5-core experiment |
| `tables/ablation_tests.csv` | One-change ablations of FrozenSE-SAM |
| `tables/efficiency.csv` | Parameters, latency, throughput and memory |
| `fold_manifests/` | The cores in the training, validation and test partition of each fold |

Folder names in `raw_runs.zip`:

| Folder | Model |
|---|---|
| `improved_skip_partial` | FrozenSE-SAM (final) |
| `pfm_phikon`, `pfm_uni` | FrozenSE-Phikon, FrozenSE-UNI |
| `baseline_unet_pretrained_r2`, `baseline_unet_scratch_r2` | U-Nets under the final training protocol |
| `nnunet_2d` | nnU-Net v2 (2D) |
| `baseline_mobilesam_native` | MobileSAM, zero-shot |
| `proposed_frozense_sam`, `improved_skip` | Development steps: fully frozen without skips; with skips |
| `ablation_se_off`, `ablation_aux_sdf`, `loss_focal_tversky`, `loss_dice` | Ablations (folds 0, 2, 4) |
| `lowdata5_frozense_d*`, `lowdata5_unet_d*` | 5-core experiment, one folder per draw |
| `baseline_unet_pretrained`, `baseline_unet_scratch`, `learning_curve`, `loss_tversky_only` | Earlier-protocol and exploratory runs, kept for completeness; not used in the paper's main tables |
