# DEIMv2 SAR Instance Segmentation Stage 1

This branch adds a lightweight query-based SAR ship instance segmentation path on top of DEIMv2. The backbone, HybridEncoder, detection decoder, dense positive detection losses, and weak-geometry query branch are kept intact.

## Main Configs

- `configs/our/deimv2_hgnetv2_s_sar_ins_stage1.yml`: default stride-8 mask branch.
- `configs/our/deimv2_hgnetv2_s_sar_ins_stage1_mask_stride4.yml`: stride-4 mask-resolution ablation for tiny or elongated ship masks.

Key switches:

- `mask_output_stride`: `8` by default; set to `4` for higher-resolution mask supervision.
- `use_mask_aux_loss`: adds mask BCE/Dice to decoder auxiliary outputs.
- `use_sparse_mask_train`: returns pixel/query mask embeddings during training so the criterion can rebuild only matched masks.
- `mask_point_sample_ratio`: enables point-sampled mask BCE/Dice and mask matching.
- `return_weak_geometry`: enabled for training, disabled for validation/test.

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

The matcher reports GT mask area before and after matching downsampling, plus empty-mask fraction. The criterion reports the same statistics at the actual `pred_masks` training resolution and prints matched Dice/IoU grouped by small, medium, and large mask area. The postprocessor can save up to `debug_max_images` validation visualizations through:

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

The smoke test covers config construction, dummy inference, postprocessing, auxiliary mask losses, weak geometry targets, and finite loss values.
