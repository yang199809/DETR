# DEIMv2 SAR Instance Segmentation Stage 1

This branch adds a lightweight query-based SAR ship instance segmentation path on top of DEIMv2. The default Stage-1 setup focuses on a stable instance-segmentation loop: `pred_masks`, mask-aware matching, mask BCE/Dice losses, COCO `bbox`/`segm` evaluation, and mask postprocessing. Optional weak-geometry code remains available for later ablations but is disabled by default.

## Main Configs

- `configs/our/deimv2_hgnetv2_s_sar_ins_stage1.yml`: default stride-4 mask branch for tiny or elongated ship masks.
- `configs/our/deimv2_hgnetv2_s_sar_ins_stage1_mask_stride4.yml`: explicit stride-4 alias for ablation runs.

Key switches:

- `mask_output_stride`: `4` by default; set to `8` only for lower-resolution ablations.
- `use_mask_aux_loss`: adds mask BCE/Dice to decoder auxiliary outputs.
- `use_sparse_mask_train`: optional memory-saving sparse mask path; disabled by default for the stable Stage-1 loop.
- `mask_point_sample_ratio`: enables point-sampled mask BCE/Dice and mask matching. GT masks are sampled at their transformed image resolution rather than being downsampled before loss/matching.
- `mask_gt_sample_ratio` / `mask_gt_fg_sample_ratio`: reserve loss samples for GT-aware foreground/background points so tiny SAR ships remain visible during early training.
- `mask_min_sampled_points` / `mask_max_sampled_points`: bound the point count used by mask loss and matching.
- `use_weak_geometry` / `return_weak_geometry`: optional later ablation path; both are disabled by default.

## Training

```bash
torchrun --nproc_per_node=1 train.py \
  -c configs/our/deimv2_hgnetv2_s_sar_ins_stage1.yml \
  --use-amp --seed=0 \
  -u train_dataloader.dataset.img_folder=/path/to/train \
     train_dataloader.dataset.ann_file=/path/to/instances_train.json \
     val_dataloader.dataset.img_folder=/path/to/val \
     val_dataloader.dataset.ann_file=/path/to/instances_val.json
```

## Evaluation

```bash
python train.py \
  -c configs/our/deimv2_hgnetv2_s_sar_ins_stage1.yml \
  --test-only \
  -r outputs/deimv2_hgnetv2_s_sar_ins_stage1/best_stg1.pth \
  -u val_dataloader.dataset.img_folder=/path/to/val \
     val_dataloader.dataset.ann_file=/path/to/instances_val.json
```

The evaluator reports both `bbox` AP and `segm` AP when `iou_types: ['bbox', 'segm']` is active in `configs/dataset/sar_instance_segmentation.yml`.

## Diagnostics

The matcher reports GT-aware sampled foreground counts and empty-sample fraction. The criterion reports mask-area statistics at the actual `pred_masks` training resolution and prints matched Dice/IoU grouped by small, medium, and large mask area. Masks that disappear after resizing are counted as `missing` and are excluded from the Dice/IoU mean so the diagnostic does not report a false perfect score for empty targets. The postprocessor can save up to `debug_max_images` validation visualizations through:

```yaml
SARInstancePostProcessor:
  debug: True
  debug_dir: ./outputs/deimv2_hgnetv2_s_sar_ins_stage1/debug_val
```

## Smoke Test

```bash
python tools/smoke_test_sar_stage1.py \
  --config configs/our/deimv2_hgnetv2_s_sar_ins_stage1.yml \
  --device cpu
```

The smoke test covers config construction, dummy inference, postprocessing, auxiliary mask losses, detection-only compatibility, and finite loss values.
