"""
Lightweight query mask head for SAR instance segmentation.

This module borrows the useful instance-segmentation pattern from
EdgeCrafter/RF-DETR-Seg: dense pixel embeddings, decoder query embeddings, and
dot-product mask logits. It deliberately avoids teacher-student distillation or
heavy pixel decoders so the DEIMv2 detection path stays intact.
"""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core import register


class DepthwiseConvBlock(nn.Module):
    """Depthwise channel-last refinement block for pixel embeddings."""

    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=True)
        self.norm = nn.LayerNorm(dim)
        self.pwlinear = nn.Linear(dim, dim)
        self.act = nn.GELU()

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwlinear(x)
        x = self.act(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return shortcut + x


class MLPBlock(nn.Module):
    """Small residual MLP block for decoder query embeddings."""

    def __init__(self, dim, expansion=4):
        super().__init__()
        hidden_dim = int(dim * expansion)
        self.block = nn.Sequential(OrderedDict([
            ('norm', nn.LayerNorm(dim)),
            ('fc1', nn.Linear(dim, hidden_dim)),
            ('act', nn.GELU()),
            ('fc2', nn.Linear(hidden_dim, dim)),
        ]))

    def forward(self, x):
        return x + self.block(x)


@register()
class SARSegmentationHead(nn.Module):
    """EdgeCrafter-style query x pixel embedding mask predictor."""

    def __init__(self,
                 in_dim=256,
                 num_blocks=3,
                 mask_hidden_dim=128,
                 mask_output_stride=8,
                 bottleneck_ratio=None,
                 use_sparse_train=False):
        super().__init__()
        self.in_dim = int(in_dim)
        self.num_blocks = int(num_blocks)
        self.mask_hidden_dim = int(mask_hidden_dim)
        self.mask_output_stride = int(mask_output_stride)
        self.use_sparse_train = bool(use_sparse_train)

        if bottleneck_ratio is not None:
            self.mask_hidden_dim = max(1, int(round(self.in_dim * float(bottleneck_ratio))))

        self.pixel_proj = nn.Conv2d(self.in_dim, self.mask_hidden_dim, 1)
        self.pixel_blocks = nn.Sequential(*[
            DepthwiseConvBlock(self.mask_hidden_dim) for _ in range(max(self.num_blocks, 1))
        ])
        self.query_blocks = nn.ModuleList([
            MLPBlock(self.in_dim) for _ in range(max(self.num_blocks, 1))
        ])
        self.query_proj = nn.Linear(self.in_dim, self.mask_hidden_dim)
        self.bias = nn.Parameter(torch.zeros(1))
        self.scale = self.mask_hidden_dim ** -0.5

    def _as_query_list(self, query_features):
        if isinstance(query_features, torch.Tensor):
            if query_features.ndim == 4:
                return [query_features[i] for i in range(query_features.shape[0])]
            return [query_features]
        return list(query_features)

    def _target_size(self, spatial_features, image_size):
        if image_size is None:
            return spatial_features.shape[-2:]
        height, width = image_size
        return (
            max(1, int(round(float(height) / self.mask_output_stride))),
            max(1, int(round(float(width) / self.mask_output_stride))),
        )

    def build_pixel_embeddings(self, spatial_features, image_size=None):
        target_size = self._target_size(spatial_features, image_size)
        if tuple(spatial_features.shape[-2:]) != tuple(target_size):
            spatial_features = F.interpolate(
                spatial_features,
                size=target_size,
                mode='bilinear',
                align_corners=False,
            )
        pixel_embed = self.pixel_proj(spatial_features)
        return self.pixel_blocks(pixel_embed)

    def build_query_embeddings(self, query_features):
        query_embeds = []
        for layer_idx, query in enumerate(self._as_query_list(query_features)):
            block = self.query_blocks[min(layer_idx, len(self.query_blocks) - 1)]
            query_embeds.append(self.query_proj(block(query)))
        return query_embeds

    def _dense_logits(self, pixel_embed, query_embed):
        return torch.einsum('bchw,bnc->bnhw', pixel_embed, query_embed) * self.scale + self.bias

    def make_sparse_output(self, pixel_embed, query_embed):
        return {
            'sparse': True,
            'pixel_embed': pixel_embed,
            'query_embed': query_embed,
            'bias': self.bias,
            'scale': self.scale,
            'mask_size': pixel_embed.shape[-2:],
        }

    @staticmethod
    def materialize_sparse(mask_repr):
        pixel_embed = mask_repr['pixel_embed']
        query_embed = mask_repr['query_embed']
        scale = mask_repr.get('scale', 1.0)
        bias = mask_repr.get('bias', 0.0)
        return torch.einsum('bchw,bnc->bnhw', pixel_embed, query_embed) * scale + bias

    @staticmethod
    def materialize_matched(mask_repr, src_idx):
        batch_idx, query_idx = src_idx
        if batch_idx.numel() == 0:
            h, w = mask_repr['pixel_embed'].shape[-2:]
            return mask_repr['pixel_embed'].new_zeros((0, h, w))
        pixel_embed = mask_repr['pixel_embed'][batch_idx]
        query_embed = mask_repr['query_embed'][batch_idx, query_idx]
        scale = mask_repr.get('scale', 1.0)
        bias = mask_repr.get('bias', 0.0)
        return torch.einsum('mchw,mc->mhw', pixel_embed, query_embed) * scale + bias

    def forward(self, spatial_features, query_features, image_size=None, sparse=False):
        pixel_embed = self.build_pixel_embeddings(spatial_features, image_size=image_size)
        query_embeds = self.build_query_embeddings(query_features)
        if sparse:
            sparse_masks = [self.make_sparse_output(pixel_embed, query_embed) for query_embed in query_embeds]
            return {
                'pred_masks': sparse_masks[-1],
                'aux_pred_masks': sparse_masks[:-1],
                'pixel_embed': pixel_embed,
            }

        dense_masks = [self._dense_logits(pixel_embed, query_embed) for query_embed in query_embeds]
        return {
            'pred_masks': dense_masks[-1],
            'aux_pred_masks': dense_masks[:-1],
            'pixel_embed': pixel_embed,
        }

    def forward_export(self, spatial_features, query_features, image_size=None):
        return self.forward(spatial_features, query_features, image_size=image_size, sparse=False)['pred_masks']
