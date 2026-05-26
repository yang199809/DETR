"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
Modules to compute the matching cost and solve the corresponding LSAP.

Copyright (c) 2024 The D-FINE Authors All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.optimize import linear_sum_assignment
from typing import Dict

from .box_ops import box_cxcywh_to_xyxy, generalized_box_iou, box_iou
from .sar_segmentation_head import SARSegmentationHead

from ..core import register
from ..misc import dist_utils
import numpy as np


@register()
class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    __share__ = ['use_focal_loss', ]

    def __init__(self, weight_dict, use_focal_loss=False, alpha=0.25, gamma=2.0,
                change_matcher=False, iou_order_alpha=1.0, matcher_change_epoch=10000,
                mask_cost_downsample_size=None, mask_cost_valid_area_eps=1e-6,
                mask_point_sample_ratio=0, mask_min_sampled_points=256,
                mask_max_sampled_points=1024, mask_gt_fg_sample_ratio=0.5,
                diagnostics=False, diagnostics_interval=100):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = weight_dict['cost_class']
        self.cost_bbox = weight_dict['cost_bbox']
        self.cost_giou = weight_dict['cost_giou']
        self.cost_mask_bce = weight_dict.get('cost_mask_bce', 0.0)
        self.cost_mask_dice = weight_dict.get('cost_mask_dice', 0.0)
        self.mask_cost_downsample_size = mask_cost_downsample_size
        self.mask_cost_valid_area_eps = float(mask_cost_valid_area_eps)
        self.mask_point_sample_ratio = int(mask_point_sample_ratio)
        self.mask_min_sampled_points = int(mask_min_sampled_points)
        self.mask_max_sampled_points = int(mask_max_sampled_points)
        self.mask_gt_fg_sample_ratio = float(mask_gt_fg_sample_ratio)
        self.diagnostics = diagnostics
        self.diagnostics_interval = int(diagnostics_interval)

        self.change_matcher = change_matcher
        self.iou_order_alpha = iou_order_alpha
        self.matcher_change_epoch = matcher_change_epoch
        if self.change_matcher:
            print(f"Using the new matching cost with iou_order_alpha = {iou_order_alpha} at epoch {matcher_change_epoch}")

        self.use_focal_loss = use_focal_loss
        self.alpha = alpha
        self.gamma = gamma

        assert (self.cost_class != 0 or self.cost_bbox != 0 or self.cost_giou != 0 or
                self.cost_mask_bce != 0 or self.cost_mask_dice != 0), "all costs cant be 0"

    def _should_report(self, step):
        if not self.diagnostics or not dist_utils.is_main_process():
            return False
        if step is None:
            return True
        return int(step) % max(self.diagnostics_interval, 1) == 0

    def _materialize_pred_masks(self, pred_masks):
        if isinstance(pred_masks, dict):
            return SARSegmentationHead.materialize_sparse(pred_masks)
        return pred_masks

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

    def _num_sample_points(self, mask_h, mask_w):
        if self.mask_point_sample_ratio <= 0:
            return 0
        num_points = max(mask_h, (mask_h * mask_w) // self.mask_point_sample_ratio)
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

    def _compute_gt_aware_mask_cost(self, pred_masks, targets, num_points, step=None):
        bs, num_queries = pred_masks.shape[:2]
        target_sizes = [int(target['masks'].shape[0]) for target in targets]
        total_targets = sum(target_sizes)
        total_cost = pred_masks.new_zeros((bs * num_queries, total_targets))

        source_areas = []
        sampled_areas = []
        col_offset = 0
        for batch_idx, (pred_per_image, target) in enumerate(zip(pred_masks, targets)):
            masks = target['masks'].to(device=pred_masks.device, dtype=pred_masks.dtype)
            num_targets = int(masks.shape[0])
            if num_targets == 0:
                continue

            row_start = batch_idx * num_queries
            row_end = row_start + num_queries
            source_areas.append(masks.flatten(1).sum(1).detach())
            cols = []
            for mask in masks:
                if mask.sum() <= self.mask_cost_valid_area_eps:
                    cols.append(pred_per_image.new_zeros(num_queries))
                    sampled_areas.append(mask.new_zeros(()))
                    continue

                # Use target-specific foreground/background points. Full-image random
                # points miss tiny SAR ships too often and make mask matching vanish.
                point_coords = self._sample_gt_aware_points(mask, num_points)
                pred_points = self._point_sample(
                    pred_per_image,
                    point_coords.unsqueeze(0).repeat(num_queries, 1, 1),
                )
                target_points = self._point_sample(
                    mask.unsqueeze(0),
                    point_coords.unsqueeze(0),
                    mode='nearest',
                )[0].clamp_(0, 1)
                sampled_areas.append(target_points.detach().sum())

                cost = pred_per_image.new_zeros(num_queries)
                if self.cost_mask_bce != 0:
                    stable_bce = F.relu(pred_points) + F.softplus(-pred_points.abs())
                    cost_bce = stable_bce.mean(1) - (pred_points * target_points.unsqueeze(0)).mean(1)
                    cost = cost + self.cost_mask_bce * cost_bce

                if self.cost_mask_dice != 0:
                    pred_prob = pred_points.sigmoid()
                    numerator = 2 * (pred_prob * target_points.unsqueeze(0)).sum(1)
                    denominator = pred_prob.sum(1) + target_points.sum()
                    cost_dice = 1 - (numerator + 1) / (denominator + 1)
                    cost = cost + self.cost_mask_dice * cost_dice

                cols.append(cost)

            total_cost[row_start:row_end, col_offset:col_offset + num_targets] = torch.stack(cols, dim=1)
            col_offset += num_targets

        if self._should_report(step) and source_areas:
            source_areas_cat = torch.cat(source_areas).float()
            sampled_areas_cat = torch.stack(sampled_areas).float()
            source_empty_frac = (source_areas_cat <= self.mask_cost_valid_area_eps).float().mean().item()
            sampled_empty_frac = (sampled_areas_cat <= self.mask_cost_valid_area_eps).float().mean().item()
            print(
                "[SARStage1][matcher] GT-aware mask point sampling "
                f"source_area mean={source_areas_cat.mean().item():.2f} "
                f"min={source_areas_cat.min().item():.2f} "
                f"max={source_areas_cat.max().item():.2f}; "
                f"source_empty_fraction={source_empty_frac:.4f}; "
                f"sampled_points={num_points}; "
                f"foreground_ratio={self.mask_gt_fg_sample_ratio:.2f}; "
                f"sampled_positive_count mean={sampled_areas_cat.mean().item():.2f} "
                f"min={sampled_areas_cat.min().item():.2f} "
                f"max={sampled_areas_cat.max().item():.2f}; "
                f"sampled_empty_fraction={sampled_empty_frac:.4f}"
            )

        return total_cost

    def _compute_mask_cost(self, outputs, targets, step=None):
        if ((self.cost_mask_bce == 0 and self.cost_mask_dice == 0) or
                'pred_masks' not in outputs or
                not all('masks' in target for target in targets)):
            return None

        pred_masks = self._materialize_pred_masks(outputs['pred_masks'])
        bs, num_queries, mask_h, mask_w = pred_masks.shape

        if sum(int(target['masks'].shape[0]) for target in targets) == 0:
            return pred_masks.new_zeros((bs * num_queries, 0))

        if self.mask_point_sample_ratio > 0:
            return self._compute_gt_aware_mask_cost(
                pred_masks, targets, self._num_sample_points(mask_h, mask_w), step=step)

        target_masks = torch.cat([
            target['masks'].to(device=pred_masks.device, dtype=pred_masks.dtype)
            for target in targets
        ], dim=0)
        valid_target = target_masks.flatten(1).sum(1) > self.mask_cost_valid_area_eps
        if self.mask_cost_downsample_size is not None:
            size = int(self.mask_cost_downsample_size)
            pred_masks = F.interpolate(
                pred_masks.flatten(0, 1)[:, None],
                size=(size, size),
                mode='bilinear',
                align_corners=False,
            )[:, 0].view(bs, num_queries, size, size)
            mask_h = mask_w = size
            target_masks = F.interpolate(
                target_masks[:, None].float(),
                size=(mask_h, mask_w),
                mode='area',
            )[:, 0].clamp_(0, 1)
        pred_flat = pred_masks.flatten(0, 1).flatten(1)
        target_flat = target_masks.flatten(1)
        num_pixels = float(pred_flat.shape[-1])

        total_cost = pred_flat.new_zeros((pred_flat.shape[0], target_flat.shape[0]))
        if self.cost_mask_bce != 0:
            # Mean BCEWithLogits over pixels without materializing [queries, gt, pixels].
            positive_part = (F.relu(pred_flat) + F.softplus(-pred_flat.abs())).mean(1, keepdim=True)
            cost_bce = positive_part - torch.matmul(pred_flat, target_flat.transpose(0, 1)) / num_pixels
            total_cost = total_cost + self.cost_mask_bce * cost_bce

        if self.cost_mask_dice != 0:
            pred_prob = pred_flat.sigmoid()
            numerator = 2 * torch.matmul(pred_prob, target_flat.transpose(0, 1))
            denominator = pred_prob.sum(1, keepdim=True) + target_flat.sum(1).unsqueeze(0)
            cost_dice = 1 - (numerator + 1) / (denominator + 1)
            total_cost = total_cost + self.cost_mask_dice * cost_dice

        if not valid_target.all():
            total_cost[:, ~valid_target] = 0.0

        return total_cost

    @torch.no_grad()
    def forward(self, outputs: Dict[str, torch.Tensor], targets, return_topk=False, epoch=0, step=None):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]
        sizes = [len(v["boxes"]) for v in targets]
        if sum(sizes) == 0:
            empty = [
                (
                    torch.empty(0, dtype=torch.int64),
                    torch.empty(0, dtype=torch.int64),
                )
                for _ in range(bs)
            ]
            return {'indices_o2m': empty} if return_topk else {'indices': empty}

        # We flatten to compute the cost matrices in a batch
        if self.use_focal_loss:
            out_prob = F.sigmoid(outputs["pred_logits"].flatten(0, 1))
        else:
            out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]

        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        if self.change_matcher and epoch >= self.matcher_change_epoch:
            # Compute the class_score
            class_score = out_prob[:, tgt_ids]  # shape = [batch_size * num_queries, gt num within a batch]

            # # Compute iou
            bbox_iou, _ = box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

            # Final cost matrix
            C = (-1) * (class_score * torch.pow(bbox_iou, self.iou_order_alpha))
        else:
            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            if self.use_focal_loss:
                out_prob = out_prob[:, tgt_ids]
                neg_cost_class = (1 - self.alpha) * (out_prob ** self.gamma) * (-(1 - out_prob + 1e-8).log())
                pos_cost_class = self.alpha * ((1 - out_prob) ** self.gamma) * (-(out_prob + 1e-8).log())
                cost_class = pos_cost_class - neg_cost_class
            else:
                cost_class = -out_prob[:, tgt_ids]

            # Compute the L1 cost between boxes
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

            # Compute the giou cost betwen boxes
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

            # Final cost matrix 3 * self.cost_bbox + 2 * self.cost_class + self.cost_giou
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou

        mask_cost = self._compute_mask_cost(outputs, targets, step=step)
        if mask_cost is not None:
            C = C + mask_cost

        C = C.view(bs, num_queries, -1).cpu()

        C = torch.nan_to_num(C, nan=1.0)
        indices_pre = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices_pre]

        # Compute topk indices
        if return_topk:
            return {'indices_o2m': self.get_top_k_matches(C, sizes=sizes, k=return_topk, initial_indices=indices_pre)}

        return {'indices': indices} # , 'indices_o2m': C.min(-1)[1]}

    def get_top_k_matches(self, C, sizes, k=1, initial_indices=None):
        indices_list = []
        # C_original = C.clone()
        for i in range(k):
            indices_k = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))] if i > 0 else initial_indices
            indices_list.append([
                (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
                for i, j in indices_k
            ])
            for c, idx_k in zip(C.split(sizes, -1), indices_k):
                idx_k = np.stack(idx_k)
                c[:, idx_k] = 1e6
        indices_list = [(torch.cat([indices_list[i][j][0] for i in range(k)], dim=0),
                        torch.cat([indices_list[i][j][1] for i in range(k)], dim=0)) for j in range(len(sizes))]
        # C.copy_(C_original)
        return indices_list
