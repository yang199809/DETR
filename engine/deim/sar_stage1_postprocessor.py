"""
Post-processing for DEIMv2 SAR Stage-1 instance segmentation.
"""

import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image, ImageDraw

from ..core import register
from ..misc import dist_utils
from .postprocessor import mod


@register()
class SARInstancePostProcessor(nn.Module):
    __share__ = [
        'num_classes',
        'use_focal_loss',
        'num_top_queries',
        'num_top_queries_bbox',
        'num_top_queries_segm',
        'remap_mscoco_category'
    ]

    def __init__(self,
                 num_classes=1,
                 use_focal_loss=True,
                 num_top_queries=300,
                 num_top_queries_bbox=None,
                 num_top_queries_segm=100,
                 remap_mscoco_category=False,
                 category_id_offset=0,
                 return_geometry=False,
                 debug=False,
                 debug_dir='outputs/sar_stage1_debug',
                 debug_max_images=5,
                 debug_score_threshold=0.25,
                 mask_threshold=0.5,
                 enable_timing=False,
                 timing_sync_cuda=False):
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_top_queries_bbox = num_top_queries if num_top_queries_bbox is None else num_top_queries_bbox
        self.num_top_queries_segm = num_top_queries_segm
        self.num_classes = int(num_classes)
        self.remap_mscoco_category = remap_mscoco_category
        self.category_id_offset = category_id_offset
        self.return_geometry = return_geometry
        self.debug = debug
        self.debug_dir = debug_dir
        self.debug_max_images = int(debug_max_images)
        self.debug_score_threshold = float(debug_score_threshold)
        self.mask_threshold = float(mask_threshold)
        self.enable_timing = enable_timing
        self.timing_sync_cuda = timing_sync_cuda
        self._debug_saved = 0
        self.last_timing = {}
        self.deploy_mode = False

    def _sync(self, tensor):
        if self.timing_sync_cuda and torch.is_tensor(tensor) and tensor.is_cuda:
            torch.cuda.synchronize(tensor.device)

    def _tic(self, tensor):
        if not self.enable_timing:
            return None
        self._sync(tensor)
        return time.perf_counter()

    def _toc(self, timings, key, start, tensor=None):
        if start is None:
            return
        self._sync(tensor)
        timings[key] = timings.get(key, 0.0) + (time.perf_counter() - start) * 1000.0

    def _select_topk(self, logits, bbox_pred, topk_queries):
        if self.use_focal_loss:
            scores = F.sigmoid(logits)
            topk = min(int(topk_queries), scores.flatten(1).shape[1])
            scores, index = torch.topk(scores.flatten(1), topk, dim=-1)
            labels = mod(index, self.num_classes)
            query_index = index // self.num_classes
        else:
            scores = F.softmax(logits, dim=-1)[:, :, :-1]
            scores, labels = scores.max(dim=-1)
            topk = min(int(topk_queries), scores.shape[1])
            if scores.shape[1] > topk:
                scores, query_index = torch.topk(scores, topk, dim=-1)
                labels = torch.gather(labels, dim=1, index=query_index)
            else:
                query_index = torch.arange(scores.shape[1], device=scores.device).unsqueeze(0).tile(scores.shape[0], 1)

        boxes = bbox_pred.gather(dim=1, index=query_index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1]))
        return labels, boxes, scores, query_index

    def _remap_labels(self, labels, device):
        if self.remap_mscoco_category:
            from ..data.dataset import mscoco_label2category
            return torch.tensor([mscoco_label2category[int(x.item())] for x in labels.flatten()])\
                .to(device).reshape(labels.shape)
        if self.category_id_offset:
            return labels + int(self.category_id_offset)
        return labels

    @staticmethod
    def _mask_panel(mask, width, height, color):
        mask = mask.detach().float().cpu()
        if mask.ndim == 4:
            mask = mask[:, 0]
        elif mask.ndim == 2:
            mask = mask[None]
        union = (mask > 0.5).any(dim=0).numpy() if mask.numel() else np.zeros((height, width), dtype=bool)
        panel = np.zeros((height, width, 3), dtype=np.uint8)
        panel[union] = np.asarray(color, dtype=np.uint8)
        return Image.fromarray(panel)

    def _save_debug_visualization(self, result, target, width, height):
        if (not self.debug or self._debug_saved >= self.debug_max_images or
                not dist_utils.is_main_process()):
            return

        pred_masks = result.get('masks')
        if pred_masks is None:
            return

        os.makedirs(self.debug_dir, exist_ok=True)
        gt_masks = target.get('masks') if target is not None else None
        if gt_masks is not None:
            gt_masks = F.interpolate(
                gt_masks.detach().float().cpu()[:, None],
                size=(int(height), int(width)),
                mode='nearest',
            )
        else:
            gt_masks = torch.zeros((0, 1, int(height), int(width)), dtype=torch.float32)

        pred_panel = self._mask_panel(pred_masks, width, height, (220, 40, 40))
        gt_panel = self._mask_panel(gt_masks, width, height, (40, 180, 80))
        overlay = Image.blend(gt_panel, pred_panel, 0.5)
        draw = ImageDraw.Draw(overlay)

        boxes = result.get('mask_boxes', result.get('boxes'))
        scores = result.get('mask_scores', result.get('scores'))
        if boxes is not None:
            boxes = boxes.detach().cpu()
            scores = scores.detach().cpu() if scores is not None else torch.ones(len(boxes))
            for box, score in zip(boxes, scores):
                if float(score) < self.debug_score_threshold:
                    continue
                x0, y0, x1, y1 = [float(v) for v in box.tolist()]
                draw.rectangle([x0, y0, x1, y1], outline=(255, 220, 0), width=2)

        canvas = Image.new('RGB', (int(width) * 3, int(height)), color=(0, 0, 0))
        canvas.paste(pred_panel, (0, 0))
        canvas.paste(gt_panel, (int(width), 0))
        canvas.paste(overlay, (int(width) * 2, 0))
        image_id = int(target['image_id'].item()) if target is not None and 'image_id' in target else self._debug_saved
        canvas.save(os.path.join(self.debug_dir, f'sar_stage1_debug_{self._debug_saved:03d}_img_{image_id}.png'))
        self._debug_saved += 1

    def forward(self, outputs, orig_target_sizes: torch.Tensor, targets=None):
        timings = {}
        logits, boxes = outputs['pred_logits'], outputs['pred_boxes']
        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy')
        bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1)

        t0 = self._tic(logits)
        labels, boxes, scores, query_index = self._select_topk(logits, bbox_pred, self.num_top_queries_bbox)
        self._toc(timings, 'topk_selection', t0, logits)

        if self.deploy_mode:
            return labels, boxes, scores

        labels = self._remap_labels(labels, boxes.device)

        pred_masks = outputs.get('pred_masks')
        selected_mask_logits = None
        mask_labels = mask_boxes = mask_scores = mask_query_index = None
        if pred_masks is not None:
            # Select segmentation queries before any expensive full-resolution upsampling or RLE encoding.
            t0 = self._tic(logits)
            mask_labels, mask_boxes, mask_scores, mask_query_index = self._select_topk(
                logits, bbox_pred, self.num_top_queries_segm)
            mask_labels = self._remap_labels(mask_labels, boxes.device)
            self._toc(timings, 'topk_selection', t0, logits)

            mask_index = mask_query_index.unsqueeze(-1).unsqueeze(-1).repeat(
                1, 1, pred_masks.shape[-2], pred_masks.shape[-1])
            selected_mask_logits = pred_masks.gather(dim=1, index=mask_index)

        geo_outputs = outputs.get('geo_outputs', {})
        results = []
        for batch_idx, (lab, box, sco) in enumerate(zip(labels, boxes, scores)):
            result = dict(labels=lab, boxes=box, scores=sco)
            if selected_mask_logits is not None:
                width, height = orig_target_sizes[batch_idx].to(dtype=torch.int64).tolist()
                t0 = self._tic(selected_mask_logits)
                mask_logits = F.interpolate(
                    selected_mask_logits[batch_idx][:, None],
                    size=(int(height), int(width)),
                    mode='bilinear',
                    align_corners=False,
                )
                self._toc(timings, 'mask_upsampling', t0, mask_logits)
                t0 = self._tic(mask_logits)
                result['masks'] = (mask_logits.sigmoid() > self.mask_threshold).to(mask_logits.dtype)
                self._toc(timings, 'threshold_sigmoid', t0, result['masks'])
                result['mask_labels'] = mask_labels[batch_idx]
                result['mask_boxes'] = mask_boxes[batch_idx]
                result['mask_scores'] = mask_scores[batch_idx]
                if targets is not None:
                    self._save_debug_visualization(result, targets[batch_idx], int(width), int(height))

            if self.return_geometry and geo_outputs:
                result['geometry'] = {}
                for key, value in geo_outputs.items():
                    if torch.is_tensor(value) and value.ndim >= 3:
                        gather_index = query_index[batch_idx].unsqueeze(-1).repeat(1, value.shape[-1])
                        result['geometry'][key] = value[batch_idx].gather(dim=0, index=gather_index)

            results.append(result)

        self.last_timing = timings
        return results

    def deploy(self):
        self.eval()
        self.deploy_mode = True
        return self
