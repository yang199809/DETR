"""
Smoke tests for the DEIMv2 SAR Stage-1 instance segmentation path.
"""

import argparse
import os
import sys

import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import engine  # noqa: F401
from engine.core import YAMLConfig
from engine.data.dataset.weak_geometry import masks_to_weak_geometry


def make_elongated_mask(height, width, x0, y0, x1, y1):
    mask = torch.zeros((height, width), dtype=torch.float32)
    mask[y0:y1, x0:x1] = 1.0
    return mask


def boxes_from_masks(masks):
    boxes = []
    height, width = masks.shape[-2:]
    for mask in masks:
        ys, xs = torch.nonzero(mask > 0, as_tuple=True)
        x0, x1 = xs.min().float(), xs.max().float() + 1
        y0, y1 = ys.min().float(), ys.max().float() + 1
        cx = ((x0 + x1) * 0.5) / width
        cy = ((y0 + y1) * 0.5) / height
        w = (x1 - x0) / width
        h = (y1 - y0) / height
        boxes.append(torch.stack([cx, cy, w, h]))
    return torch.stack(boxes, dim=0)


def make_targets(batch_size, height, width, device):
    targets = []
    for batch_idx in range(batch_size):
        mask_a = make_elongated_mask(height, width, 18, 52, 104, 62)
        mask_b = make_elongated_mask(height, width, 54, 20, 66, 104)
        masks = torch.stack([mask_a, mask_b], dim=0)
        target = {
            'boxes': boxes_from_masks(masks),
            'labels': torch.zeros(2, dtype=torch.long),
            'masks': masks,
            'image_id': torch.tensor([batch_idx]),
            'orig_size': torch.tensor([width, height]),
        }
        target.update(masks_to_weak_geometry(masks))
        targets.append({key: value.to(device) for key, value in target.items()})
    return targets


def assert_finite_losses(losses):
    required = {
        'loss_mal',
        'loss_bbox',
        'loss_giou',
        'loss_mask_bce',
        'loss_mask_dice',
        'loss_geo_center',
        'loss_geo_scale',
        'loss_geo_dir',
        'loss_geo_ani',
    }
    missing = required - set(losses.keys())
    assert not missing, f'missing losses: {sorted(missing)}'
    for name, value in losses.items():
        assert torch.isfinite(value).all(), f'{name} is not finite: {value}'


def test_weak_geometry():
    horizontal = make_elongated_mask(96, 96, 10, 42, 86, 52)
    square = make_elongated_mask(96, 96, 35, 35, 61, 61)
    geometry = masks_to_weak_geometry(torch.stack([horizontal, square], dim=0))
    assert geometry['gt_center'].shape == (2, 2)
    assert geometry['gt_scale'].shape == (2, 2)
    assert geometry['gt_axis'].shape == (2, 2)
    assert torch.allclose(geometry['gt_axis'].norm(dim=-1), torch.ones(2), atol=1e-4)
    assert geometry['gt_anisotropy'][0] > geometry['gt_anisotropy'][1]

    pred_axis = -geometry['gt_axis']
    sign_invariant = 1 - (pred_axis * geometry['gt_axis']).sum(-1).abs()
    assert torch.all(sign_invariant < 1e-4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/our/deimv2_hgnetv2_s_sar_ins_stage1.yml')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--size', type=int, default=128)
    args = parser.parse_args()

    device = torch.device(args.device)
    test_weak_geometry()

    cfg = YAMLConfig(
        args.config,
        eval_spatial_size=[args.size, args.size],
        HGNetv2={'pretrained': False},
        DEIMv2_SAR_INS_STAGE1={
            'decoder': {
                'num_queries': 20,
                'num_denoising': 0,
                'num_layers': 2,
                'dim_feedforward': 512,
                'eval_idx': -1,
                'enable_timing': True,
                'mask_output_stride': 4,
            }
        },
        SARStage1Criterion={
            'mask_diagnostics': True,
            'mask_diagnostics_interval': 1,
        },
        SARInstancePostProcessor={
            'num_top_queries_bbox': 20,
            'num_top_queries_segm': 5,
            'enable_timing': True,
            'debug': False,
        },
        train_dataloader={'total_batch_size': 2},
        val_dataloader={'total_batch_size': 2},
        use_amp=False,
        use_ema=False,
    )

    model = cfg.model.to(device)
    criterion = cfg.criterion.to(device)
    postprocessor = cfg.postprocessor.to(device)

    samples = torch.randn(2, 3, args.size, args.size, device=device)
    targets = make_targets(2, args.size, args.size, device)

    model.eval()
    with torch.no_grad():
        outputs = model(samples)
        assert outputs['pred_logits'].shape[:2] == (2, 20)
        assert outputs['pred_boxes'].shape[:2] == (2, 20)
        assert outputs['pred_masks'].shape[:2] == (2, 20)
        assert outputs['pred_masks'].shape[-1] >= args.size // 4
        orig_size = torch.tensor([[args.size * 2, args.size + 16]] * 2, device=device)
        results = postprocessor(outputs, orig_size, targets=targets)
        assert 'masks' in results[0]
        assert results[0]['masks'].ndim == 4
        assert results[0]['scores'].numel() == 20
        assert results[0]['mask_scores'].numel() == 5
        assert results[0]['masks'].shape[0] == 5
        assert results[0]['masks'].shape[-2:] == (args.size + 16, args.size * 2)
        assert postprocessor.last_timing
        assert 'timings' in outputs

    model.train()
    outputs = model(samples, targets=targets)
    assert outputs['aux_outputs'][0]['pred_masks'].shape[:2] == (2, 20)
    losses = criterion(outputs, targets, epoch=0, global_step=0)
    assert 'loss_mask_bce_aux_0' in losses
    assert 'loss_mask_dice_aux_0' in losses
    assert_finite_losses(losses)

    print('SAR Stage-1 smoke test passed.')


if __name__ == '__main__':
    main()
