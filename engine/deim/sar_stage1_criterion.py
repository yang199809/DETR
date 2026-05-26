"""
Stage-1 SAR instance segmentation criterion.

Detection losses and Hungarian matching stay identical to DEIMv2. Mask and weak
geometry losses are added only for final matched positive object queries.
"""

import torch
import torch.nn.functional as F

from ..core import register
from ..data.dataset.weak_geometry import masks_to_weak_geometry
from ..misc.dist_utils import get_world_size, is_dist_available_and_initialized, is_main_process
from .deim_criterion import DEIMCriterion
from .sar_segmentation_head import SARSegmentationHead


@register()
class SARStage1Criterion(DEIMCriterion):
    CUSTOM_LOSSES = {'masks', 'weak_geometry'}

    def __init__(self, *args, mask_diagnostics=False, mask_diagnostics_interval=100,
                 mask_point_sample_ratio=0, oversample_ratio=3.0,
                 importance_sample_ratio=0.75, mask_gt_sample_ratio=0.5,
                 mask_gt_fg_sample_ratio=0.75, mask_min_sampled_points=256,
                 mask_max_sampled_points=1024, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_diagnostics = mask_diagnostics
        self.mask_diagnostics_interval = int(mask_diagnostics_interval)
        self.mask_point_sample_ratio = int(mask_point_sample_ratio)
        self.oversample_ratio = float(oversample_ratio)
        self.importance_sample_ratio = float(importance_sample_ratio)
        self.mask_gt_sample_ratio = float(mask_gt_sample_ratio)
        self.mask_gt_fg_sample_ratio = float(mask_gt_fg_sample_ratio)
        self.mask_min_sampled_points = int(mask_min_sampled_points)
        self.mask_max_sampled_points = int(mask_max_sampled_points)

    def _zero_loss(self, outputs):
        return outputs['pred_logits'].sum() * 0.0

    def _num_matched(self, indices):
        return sum(src.numel() for src, _ in indices)

    def _ensure_geometry_targets(self, targets):
        for target in targets:
            if 'gt_center' not in target and 'masks' in target:
                target.update(masks_to_weak_geometry(target['masks']))

    def _should_report_mask_diagnostics(self, step):
        if not self.mask_diagnostics or not is_main_process():
            return False
        if step is None:
            return True
        return int(step) % max(self.mask_diagnostics_interval, 1) == 0

    def _mask_device_dtype(self, pred_masks, outputs):
        if isinstance(pred_masks, dict):
            tensor = pred_masks['pixel_embed']
        else:
            tensor = pred_masks
        return tensor.device, tensor.dtype

    def _gather_pred_masks(self, pred_masks, src_idx):
        if isinstance(pred_masks, dict):
            return SARSegmentationHead.materialize_matched(pred_masks, src_idx)
        return pred_masks[src_idx]

    @staticmethod
    def _point_sample(masks, point_coords, mode='bilinear'):
        grid = point_coords.mul(2.0).sub(1.0).unsqueeze(2)
        samples = F.grid_sample(
            masks[:, None],
            grid,
            mode=mode,
            align_corners=False,
        )
        return samples[:, 0, :, 0]

    def _num_sample_points(self, height, width):
        num_points = max(1, (height * width) // self.mask_point_sample_ratio)
        if self.mask_min_sampled_points > 0:
            num_points = max(num_points, self.mask_min_sampled_points)
        if self.mask_max_sampled_points > 0:
            num_points = min(num_points, self.mask_max_sampled_points)
        return int(max(1, num_points))

    @staticmethod
    def _sample_indices_as_points(indices, height, width, num_points):
        choice = torch.randint(indices.shape[0], (num_points,), device=indices.device)
        selected = indices[choice]
        y = (selected[:, 0].float() + 0.5) / float(height)
        x = (selected[:, 1].float() + 0.5) / float(width)
        return torch.stack([x, y], dim=-1)

    def _sample_gt_aware_points(self, mask, num_points):
        height, width = mask.shape[-2:]
        foreground = (mask > 0.5).nonzero(as_tuple=False)
        if foreground.numel() == 0:
            return torch.rand(num_points, 2, device=mask.device)

        num_fg = int(round(num_points * self.mask_gt_fg_sample_ratio))
        num_fg = min(num_points, max(1, num_fg))
        num_bg = num_points - num_fg

        points = [self._sample_indices_as_points(foreground, height, width, num_fg)]
        if num_bg > 0:
            background = (mask <= 0.5).nonzero(as_tuple=False)
            if background.numel() > 0:
                points.append(self._sample_indices_as_points(background, height, width, num_bg))
            else:
                points.append(torch.rand(num_bg, 2, device=mask.device))

        points = torch.cat(points, dim=0)
        order = torch.randperm(points.shape[0], device=points.device)
        return points[order]

    def _sample_gt_aware_points_batch(self, target_masks, num_points):
        return torch.stack([
            self._sample_gt_aware_points(mask, num_points) for mask in target_masks
        ], dim=0)

    def _sample_uncertain_points(self, logits, num_points):
        if num_points <= 0:
            return None
        num_masks = logits.shape[0]
        num_sampled = max(num_points, int(num_points * self.oversample_ratio))
        point_coords = torch.rand(num_masks, num_sampled, 2, device=logits.device)
        with torch.no_grad():
            point_logits = self._point_sample(logits.detach(), point_coords)
            num_uncertain = min(num_points, int(num_points * self.importance_sample_ratio))
            num_uncertain = max(1, num_uncertain)
            topk = torch.topk(-point_logits.abs(), k=num_uncertain, dim=1).indices
            uncertain_coords = point_coords.gather(1, topk.unsqueeze(-1).expand(-1, -1, 2))
            num_random = num_points - num_uncertain
            if num_random > 0:
                random_coords = torch.rand(num_masks, num_random, 2, device=logits.device)
                return torch.cat([uncertain_coords, random_coords], dim=1)
            return uncertain_coords

    def _sample_points(self, logits, target_masks=None):
        if self.mask_point_sample_ratio <= 0:
            return None
        num_masks, height, width = logits.shape
        if num_masks == 0:
            return None
        num_points = self._num_sample_points(height, width)

        gt_points = None
        num_gt_points = 0
        if target_masks is not None and self.mask_gt_sample_ratio > 0:
            num_gt_points = min(num_points, max(1, int(round(num_points * self.mask_gt_sample_ratio))))
            # Reserve a stable portion of loss samples for GT foreground/background.
            # This keeps tiny SAR ships visible even when uncertainty sampling would
            # otherwise select mostly background points early in training.
            gt_points = self._sample_gt_aware_points_batch(target_masks, num_gt_points)

        uncertain_points = self._sample_uncertain_points(logits, num_points - num_gt_points)
        if gt_points is None:
            return uncertain_points
        if uncertain_points is None:
            return gt_points
        return torch.cat([gt_points, uncertain_points], dim=1)

    def _report_mask_resize_diagnostics(self, source_masks, resized_masks, resolution, step=None,
                                        branch='final', pred_masks=None):
        if not self._should_report_mask_diagnostics(step) or source_masks.numel() == 0:
            return
        source_areas = source_masks.detach().float().flatten(1).sum(1)
        resized_areas = resized_masks.detach().float().flatten(1).sum(1)
        empty_fraction = (resized_areas <= 0).float().mean().item()
        print(
            f"[SARStage1][criterion][{branch}] GT mask area before resize "
            f"mean={source_areas.mean().item():.2f} "
            f"min={source_areas.min().item():.2f} "
            f"max={source_areas.max().item():.2f}; "
            f"after resize to {resolution[0]}x{resolution[1]} "
            f"mean={resized_areas.mean().item():.2f} "
            f"min={resized_areas.min().item():.2f} "
            f"max={resized_areas.max().item():.2f}; "
            f"empty_fraction={empty_fraction:.4f}"
        )
        if pred_masks is None:
            return

        pred_bin = (pred_masks.detach().sigmoid() > 0.5).float().flatten(1)
        tgt_bin = (resized_masks.detach() > 0.5).float().flatten(1)
        inter = (pred_bin * tgt_bin).sum(1)
        pred_area = pred_bin.sum(1)
        tgt_area = tgt_bin.sum(1)
        dice = (2 * inter + 1) / (pred_area + tgt_area + 1)
        iou = (inter + 1) / (pred_area + tgt_area - inter + 1)
        groups = [
            ('small', source_areas < 32),
            ('medium', (source_areas >= 32) & (source_areas < 256)),
            ('large', source_areas >= 256),
        ]
        parts = []
        for name, keep in groups:
            if keep.any():
                missing = (resized_areas[keep] <= 0).float().mean().item()
                valid = keep & (resized_areas > 0)
                if valid.any():
                    dice_value = f"{dice[valid].mean().item():.4f}"
                    iou_value = f"{iou[valid].mean().item():.4f}"
                    valid_n = int(valid.sum().item())
                else:
                    dice_value = "nan"
                    iou_value = "nan"
                    valid_n = 0
                parts.append(
                    f"{name}:n={int(keep.sum().item())},"
                    f"valid_n={valid_n},"
                    f"dice={dice_value},"
                    f"iou={iou_value},"
                    f"missing={missing:.4f}"
                )
        if parts:
            print(f"[SARStage1][criterion][{branch}] matched mask diagnostics " + "; ".join(parts))

    def loss_masks(self, outputs, targets, indices, num_boxes, step=None, branch='final',
                   enable_diagnostics=True):
        if 'pred_masks' not in outputs or self._num_matched(indices) == 0:
            zero = self._zero_loss(outputs)
            return {'loss_mask_bce': zero, 'loss_mask_dice': zero}

        if not all('masks' in target for target in targets):
            zero = self._zero_loss(outputs)
            return {'loss_mask_bce': zero, 'loss_mask_dice': zero}

        for batch_idx, target in enumerate(targets):
            num_labels = int(target['labels'].shape[0])
            num_masks = int(target['masks'].shape[0])
            if num_labels != num_masks:
                raise ValueError(
                    f"Mask/label count mismatch at batch index {batch_idx}: "
                    f"{num_labels} labels but {num_masks} masks. "
                    "Disable mask-unsafe augmentations or update them to transform masks."
                )

        pred_masks = outputs['pred_masks']
        src_idx = self._get_src_permutation_idx(indices)
        src_masks = self._gather_pred_masks(pred_masks, src_idx)
        device, dtype = self._mask_device_dtype(pred_masks, outputs)
        target_masks = torch.cat([
            target['masks'][j] for target, (_, j) in zip(targets, indices)
        ], dim=0).to(device=device, dtype=dtype)

        source_masks = target_masks
        resized_target_masks = F.interpolate(
            source_masks[:, None],
            size=src_masks.shape[-2:],
            mode='nearest',
        )[:, 0]
        if enable_diagnostics:
            self._report_mask_resize_diagnostics(
                source_masks,
                resized_target_masks,
                src_masks.shape[-2:],
                step=step,
                branch=branch,
                pred_masks=src_masks,
            )

        point_coords = self._sample_points(src_masks, source_masks)
        if point_coords is not None:
            # Sample GT masks at their native training resolution. This avoids erasing
            # tiny SAR ship masks by first resizing them down to pred_masks resolution.
            src_for_loss = self._point_sample(src_masks, point_coords)
            target_for_loss = self._point_sample(source_masks, point_coords, mode='nearest')
        else:
            target_masks = resized_target_masks
            src_for_loss = src_masks.flatten(1)
            target_for_loss = target_masks.flatten(1)

        loss_bce = F.binary_cross_entropy_with_logits(
            src_for_loss, target_for_loss, reduction='none')
        loss_bce = loss_bce.flatten(1).mean(1).sum() / num_boxes

        src_probs = src_for_loss.sigmoid().flatten(1)
        target_flat = target_for_loss.flatten(1)
        numerator = 2 * (src_probs * target_flat).sum(1)
        denominator = src_probs.sum(1) + target_flat.sum(1)
        loss_dice = (1 - (numerator + 1) / (denominator + 1)).sum() / num_boxes

        return {'loss_mask_bce': loss_bce, 'loss_mask_dice': loss_dice}

    def loss_weak_geometry(self, outputs, targets, indices, num_boxes):
        self._ensure_geometry_targets(targets)
        geo_outputs = outputs.get('geo_outputs', {})
        required_preds = ('pred_center', 'pred_scale', 'pred_dir', 'pred_anisotropy')
        required_targets = ('gt_center', 'gt_scale', 'gt_axis', 'gt_anisotropy')

        if (self._num_matched(indices) == 0 or
                not all(key in geo_outputs for key in required_preds) or
                not all(all(key in target for key in required_targets) for target in targets)):
            zero = self._zero_loss(outputs)
            return {
                'loss_geo_center': zero,
                'loss_geo_scale': zero,
                'loss_geo_dir': zero,
                'loss_geo_ani': zero,
            }

        src_idx = self._get_src_permutation_idx(indices)
        pred_center = geo_outputs['pred_center'][src_idx]
        pred_scale = geo_outputs['pred_scale'][src_idx]
        pred_dir = geo_outputs['pred_dir'][src_idx]
        pred_anisotropy = geo_outputs['pred_anisotropy'][src_idx]

        def cat_target(key):
            return torch.cat([
                target[key][j] for target, (_, j) in zip(targets, indices)
            ], dim=0).to(device=pred_center.device, dtype=pred_center.dtype)

        gt_center = cat_target('gt_center')
        gt_scale = cat_target('gt_scale')
        gt_axis = F.normalize(cat_target('gt_axis'), p=2, dim=-1, eps=1e-6)
        gt_anisotropy = cat_target('gt_anisotropy')
        valid = cat_target('gt_geo_valid') if all('gt_geo_valid' in t for t in targets) else torch.ones_like(gt_anisotropy)
        geo_weight = cat_target('gt_geo_weight') if all('gt_geo_weight' in t for t in targets) else valid

        valid = valid.squeeze(-1)
        geo_weight = geo_weight.squeeze(-1)
        loss_center = (F.l1_loss(pred_center, gt_center, reduction='none').sum(-1) * valid).sum() / num_boxes
        loss_scale = (F.l1_loss(pred_scale, gt_scale, reduction='none').sum(-1) * geo_weight).sum() / num_boxes

        dir_dot = (pred_dir * gt_axis).sum(-1).clamp(-1.0, 1.0).abs()
        loss_dir = ((1.0 - dir_dot) * geo_weight).sum() / num_boxes
        loss_ani = (F.l1_loss(pred_anisotropy, gt_anisotropy, reduction='none').squeeze(-1) * geo_weight).sum() / num_boxes

        return {
            'loss_geo_center': loss_center,
            'loss_geo_scale': loss_scale,
            'loss_geo_dir': loss_dir,
            'loss_geo_ani': loss_ani,
        }

    def forward(self, outputs, targets, epoch=0, **kwargs):
        requested_losses = list(self.losses)
        det_losses = [loss for loss in requested_losses if loss not in self.CUSTOM_LOSSES]

        try:
            self.losses = det_losses
            losses = super().forward(outputs, targets, epoch=epoch, **kwargs)
        finally:
            self.losses = requested_losses

        outputs_without_aux = {k: v for k, v in outputs.items() if 'aux' not in k}
        indices = self.matcher(
            outputs_without_aux, targets, epoch=epoch, step=kwargs.get('global_step', None))['indices']

        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=outputs['pred_logits'].device)
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        custom_losses = {}
        global_step = kwargs.get('global_step', None)
        if 'masks' in requested_losses:
            custom_losses.update(self.loss_masks(
                outputs, targets, indices, num_boxes, step=global_step, branch='final'))
            if 'aux_outputs' in outputs:
                for aux_idx, aux_outputs in enumerate(outputs['aux_outputs']):
                    if 'pred_masks' not in aux_outputs:
                        continue
                    aux_indices = self.matcher(
                        aux_outputs, targets, epoch=epoch, step=global_step)['indices']
                    aux_losses = self.loss_masks(
                        aux_outputs,
                        targets,
                        aux_indices,
                        num_boxes,
                        step=global_step,
                        branch=f'aux_{aux_idx}',
                        enable_diagnostics=False,
                    )
                    custom_losses.update({
                        f'{key}_aux_{aux_idx}': value for key, value in aux_losses.items()
                    })
        if 'weak_geometry' in requested_losses:
            custom_losses.update(self.loss_weak_geometry(outputs, targets, indices, num_boxes))

        weighted_custom_losses = {}
        for key, value in custom_losses.items():
            base_key = key.split('_aux_')[0] if '_aux_' in key else key
            weighted_custom_losses[key] = value * self.weight_dict.get(
                key, self.weight_dict.get(base_key, 1.0))
        custom_losses = weighted_custom_losses
        losses.update(custom_losses)
        return {k: torch.nan_to_num(v, nan=0.0) for k, v in losses.items()}
