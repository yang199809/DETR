"""
Minimal Stage-1 SAR instance segmentation extension for DEIMv2.
"""

import time
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from ..core import register
from .deim_decoder import DEIMTransformer
from .denoising import get_contrastive_denoising_training_group
from .sar_segmentation_head import SARSegmentationHead


@register()
class DEIMv2_SAR_INS_STAGE1(nn.Module):
    """DEIMv2 wrapper that keeps the backbone/encoder path unchanged."""
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self, backbone: nn.Module, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, targets=None):
        x = self.backbone(x)
        x = self.encoder(x)
        return self.decoder(x, targets)

    def deploy(self):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self


class ConvBlock(nn.Module):
    def __init__(self, channels, act_layer=nn.SiLU):
        super().__init__()
        self.block = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(channels, channels, 3, padding=1, bias=False)),
            ('norm', nn.BatchNorm2d(channels)),
            ('act', act_layer(inplace=True)),
        ]))

    def forward(self, x):
        return self.block(x)


@register()
class LightweightPixelDecoder(nn.Module):
    """Small FPN-like mask feature decoder."""

    def __init__(self, in_channels=256, mask_dim=128, num_levels=3):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, mask_dim, 1) for _ in range(num_levels)
        ])
        self.output_convs = nn.Sequential(
            ConvBlock(mask_dim),
            ConvBlock(mask_dim),
        )

    def forward(self, feats):
        feats = feats[:len(self.lateral_convs)]
        laterals = [proj(feat) for proj, feat in zip(self.lateral_convs, feats)]
        out = laterals[-1]
        for lateral in reversed(laterals[:-1]):
            out = F.interpolate(out, size=lateral.shape[-2:], mode='nearest') + lateral
        return self.output_convs(out)


@register()
class QueryBasedMaskHead(nn.Module):
    """Dot-product query mask head."""

    def __init__(self, hidden_dim=256, mask_dim=128):
        super().__init__()
        self.query_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, mask_dim),
        )
        self.scale = mask_dim ** -0.5

    def forward(self, queries, pixel_features):
        mask_embed = self.query_proj(queries)
        return torch.einsum('bqc,bchw->bqhw', mask_embed, pixel_features) * self.scale


@register()
class WeakGeometryQueryInit(nn.Module):
    """Inject latent mask-derived geometry predictions into initial queries."""

    def __init__(self, hidden_dim=256, act_layer=nn.SiLU):
        super().__init__()
        self.pred_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            act_layer(inplace=True),
            nn.Linear(hidden_dim, 7),
        )
        self.geo_embed = nn.Sequential(
            nn.Linear(7, hidden_dim),
            act_layer(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim + 7, hidden_dim),
            act_layer(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, queries, pos_embed=None, reference_boxes=None):
        src = queries if pos_embed is None else queries + pos_embed
        raw = self.pred_head(src)
        center = raw[..., 0:2].sigmoid()
        scale = raw[..., 2:4].sigmoid()
        direction = F.normalize(raw[..., 4:6], p=2, dim=-1, eps=1e-6)
        anisotropy = raw[..., 6:7].sigmoid()

        geo = torch.cat([center, scale, direction, anisotropy], dim=-1)
        alpha = self.gate(torch.cat([src, geo], dim=-1)).sigmoid()
        enhanced = queries + alpha * self.geo_embed(geo)

        return enhanced, {
            'pred_center': center,
            'pred_scale': scale,
            'pred_dir': direction,
            'pred_anisotropy': anisotropy,
            'geo_alpha': alpha,
        }


@register()
class SARStage1DEIMTransformer(DEIMTransformer):
    __share__ = ['num_classes', 'eval_spatial_size']

    def __init__(self,
                 num_classes=80,
                 hidden_dim=256,
                 num_queries=300,
                 feat_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 num_levels=3,
                 num_points=4,
                 nhead=8,
                 num_layers=6,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 num_denoising=100,
                 label_noise_ratio=0.5,
                 box_noise_scale=1.0,
                 learn_query_content=False,
                 eval_spatial_size=None,
                 eval_idx=-1,
                 eps=1e-2,
                 aux_loss=True,
                 cross_attn_method='default',
                 query_select_method='default',
                 reg_max=32,
                 reg_scale=4.,
                 layer_scale=1,
                 mlp_act='relu',
                 use_gateway=True,
                 share_bbox_head=False,
                 share_score_head=False,
                 mask_hidden_dim=128,
                 mask_output_stride=8,
                 mask_num_blocks=None,
                 use_mask_aux_loss=True,
                 use_sparse_mask_train=False,
                 use_weak_geometry=True,
                 return_geometry=True,
                 enable_timing=False,
                 timing_sync_cuda=False):
        super().__init__(
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            feat_channels=feat_channels,
            feat_strides=feat_strides,
            num_levels=num_levels,
            num_points=num_points,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            num_denoising=num_denoising,
            label_noise_ratio=label_noise_ratio,
            box_noise_scale=box_noise_scale,
            learn_query_content=learn_query_content,
            eval_spatial_size=eval_spatial_size,
            eval_idx=eval_idx,
            eps=eps,
            aux_loss=aux_loss,
            cross_attn_method=cross_attn_method,
            query_select_method=query_select_method,
            reg_max=reg_max,
            reg_scale=reg_scale,
            layer_scale=layer_scale,
            mlp_act=mlp_act,
            use_gateway=use_gateway,
            share_bbox_head=share_bbox_head,
            share_score_head=share_score_head,
        )
        self.mask_hidden_dim = mask_hidden_dim
        self.mask_output_stride = int(mask_output_stride)
        self.mask_base_stride = min(feat_strides[:num_levels])
        self.use_mask_aux_loss = use_mask_aux_loss
        self.use_sparse_mask_train = use_sparse_mask_train
        self.use_weak_geometry = use_weak_geometry
        self.return_geometry = return_geometry
        self.enable_timing = enable_timing
        self.timing_sync_cuda = timing_sync_cuda
        self.mask_head = SARSegmentationHead(
            in_dim=hidden_dim,
            num_blocks=num_layers if mask_num_blocks is None else mask_num_blocks,
            mask_hidden_dim=mask_hidden_dim,
            mask_output_stride=self.mask_output_stride,
            use_sparse_train=use_sparse_mask_train,
        )
        self.weak_geometry_init = WeakGeometryQueryInit(hidden_dim) if use_weak_geometry else None

    def _build_mask_refine(self, mask_hidden_dim):
        """Optionally refine the stride-8 mask feature to a higher spatial resolution."""
        if self.mask_output_stride == self.mask_base_stride:
            return nn.Identity()
        if self.mask_output_stride < self.mask_base_stride:
            if self.mask_base_stride % self.mask_output_stride != 0:
                raise ValueError(
                    f"mask_output_stride={self.mask_output_stride} must divide "
                    f"the base mask stride {self.mask_base_stride}"
                )
            scale_factor = self.mask_base_stride // self.mask_output_stride
            return nn.Sequential(
                nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False),
                ConvBlock(mask_hidden_dim),
            )
        if self.mask_output_stride % self.mask_base_stride != 0:
            raise ValueError(
                f"mask_output_stride={self.mask_output_stride} must be a multiple of "
                f"the base mask stride {self.mask_base_stride}"
            )
        scale_factor = self.mask_output_stride // self.mask_base_stride
        return nn.AvgPool2d(kernel_size=scale_factor, stride=scale_factor)

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

    def _reset_parameters(self, feat_channels):
        super()._reset_parameters(feat_channels)
        if hasattr(self, 'pixel_decoder'):
            for module in self.pixel_decoder.modules():
                if isinstance(module, nn.Conv2d):
                    init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        init.constant_(module.bias, 0)
        if hasattr(self, 'mask_refine'):
            for module in self.mask_refine.modules():
                if isinstance(module, nn.Conv2d):
                    init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        init.constant_(module.bias, 0)
        if hasattr(self, 'mask_head'):
            for module in self.mask_head.modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        init.constant_(module.bias, 0)

    def _memory_to_feature_maps(self, memory, spatial_shapes):
        batch_size, _, channels = memory.shape
        lengths = [h * w for h, w in spatial_shapes]
        feats = []
        for feat, (h, w) in zip(memory.split(lengths, dim=1), spatial_shapes):
            feats.append(feat.permute(0, 2, 1).reshape(batch_size, channels, h, w))
        return feats

    def _get_decoder_input(self,
                           memory: torch.Tensor,
                           spatial_shapes,
                           denoising_logits=None,
                           denoising_bbox_unact=None):
        if self.training or self.eval_spatial_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
        else:
            anchors = self.anchors
            valid_mask = self.valid_mask
        if memory.shape[0] > 1:
            anchors = anchors.repeat(memory.shape[0], 1, 1)

        memory = valid_mask.to(memory.dtype) * memory
        enc_outputs_logits = self.enc_score_head(memory)
        enc_topk_memory, enc_topk_logits, enc_topk_anchors = \
            self._select_topk(memory, enc_outputs_logits, anchors, self.num_queries)
        enc_topk_bbox_unact = self.enc_bbox_head(enc_topk_memory) + enc_topk_anchors

        enc_topk_bboxes_list, enc_topk_logits_list = [], []
        if self.training:
            enc_topk_bboxes = F.sigmoid(enc_topk_bbox_unact)
            enc_topk_bboxes_list.append(enc_topk_bboxes)
            enc_topk_logits_list.append(enc_topk_logits)

        if self.learn_query_content:
            content = self.tgt_embed.weight.unsqueeze(0).tile([memory.shape[0], 1, 1])
        else:
            content = enc_topk_memory.detach()

        geo_outputs = {}
        if self.weak_geometry_init is not None:
            content, geo_outputs = self.weak_geometry_init(
                content,
                reference_boxes=F.sigmoid(enc_topk_bbox_unact.detach()),
            )

        enc_topk_bbox_unact = enc_topk_bbox_unact.detach()

        if denoising_bbox_unact is not None:
            enc_topk_bbox_unact = torch.concat([denoising_bbox_unact, enc_topk_bbox_unact], dim=1)
            content = torch.concat([denoising_logits, content], dim=1)

        return content, enc_topk_bbox_unact, enc_topk_bboxes_list, enc_topk_logits_list, geo_outputs

    def forward(self, feats, targets=None):
        timings = {}
        memory, spatial_shapes = self._get_encoder_input(feats)
        feature_maps = self._memory_to_feature_maps(memory, spatial_shapes)
        # The highest-resolution encoded feature (normally stride 8) feeds the lightweight mask head.
        spatial_feature = feature_maps[0]
        image_size = (
            spatial_feature.shape[-2] * self.mask_base_stride,
            spatial_feature.shape[-1] * self.mask_base_stride,
        )

        if self.training and self.num_denoising > 0:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = \
                get_contrastive_denoising_training_group(
                    targets,
                    self.num_classes,
                    self.num_queries,
                    self.denoising_class_embed,
                    num_denoising=self.num_denoising,
                    label_noise_ratio=self.label_noise_ratio,
                    box_noise_scale=1.0,
                )
        else:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

        init_ref_contents, init_ref_points_unact, enc_topk_bboxes_list, enc_topk_logits_list, geo_outputs = \
            self._get_decoder_input(memory, spatial_shapes, denoising_logits, denoising_bbox_unact)

        t0 = self._tic(memory)
        decoder_outputs = self.decoder(
            init_ref_contents,
            init_ref_points_unact,
            memory,
            spatial_shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            self.pre_bbox_head,
            self.integral,
            self.up,
            self.reg_scale,
            attn_mask=attn_mask,
            dn_meta=dn_meta,
            return_intermediate_queries=True)
        self._toc(timings, 'decoder_forward', t0, memory)
        out_bboxes, out_logits, out_corners, out_refs, pre_bboxes, pre_logits, out_queries = decoder_outputs

        if self.training and dn_meta is not None:
            dn_pre_logits, pre_logits = torch.split(pre_logits, dn_meta['dn_num_split'], dim=1)
            dn_pre_bboxes, pre_bboxes = torch.split(pre_bboxes, dn_meta['dn_num_split'], dim=1)
            dn_out_logits, out_logits = torch.split(out_logits, dn_meta['dn_num_split'], dim=2)
            dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta['dn_num_split'], dim=2)
            dn_out_corners, out_corners = torch.split(out_corners, dn_meta['dn_num_split'], dim=2)
            dn_out_refs, out_refs = torch.split(out_refs, dn_meta['dn_num_split'], dim=2)
            dn_out_queries, out_queries = torch.split(out_queries, dn_meta['dn_num_split'], dim=2)

        final_queries = out_queries[-1]
        t0 = self._tic(final_queries)
        query_features = [layer_queries for layer_queries in out_queries]
        mask_outputs = self.mask_head(
            spatial_feature,
            query_features,
            image_size=image_size,
            sparse=self.training and self.use_sparse_mask_train,
        )
        pred_masks = mask_outputs['pred_masks']
        aux_pred_masks = []
        if self.training and self.aux_loss and self.use_mask_aux_loss:
            # Deep mask supervision keeps segmentation query learning closer to detection supervision density.
            aux_pred_masks = mask_outputs['aux_pred_masks']
        self._toc(timings, 'mask_projection', t0, pred_masks)

        if self.training:
            out = {
                'pred_logits': out_logits[-1],
                'pred_boxes': out_bboxes[-1],
                'pred_corners': out_corners[-1],
                'ref_points': out_refs[-1],
                'pred_masks': pred_masks,
                'geo_outputs': geo_outputs,
                'up': self.up,
                'reg_scale': self.reg_scale,
            }
        else:
            out = {
                'pred_logits': out_logits[-1],
                'pred_boxes': out_bboxes[-1],
                'pred_masks': pred_masks,
            }
            if self.return_geometry:
                out['geo_outputs'] = geo_outputs

        if timings:
            out['timings'] = timings

        if self.training and self.aux_loss:
            aux_outputs = self._set_aux_loss2(
                out_logits[:-1], out_bboxes[:-1], out_corners[:-1], out_refs[:-1],
                out_corners[-1], out_logits[-1])
            if aux_pred_masks is not None:
                for aux_output, aux_mask in zip(aux_outputs, aux_pred_masks):
                    aux_output['pred_masks'] = aux_mask
            out['aux_outputs'] = aux_outputs
            out['enc_aux_outputs'] = self._set_aux_loss(enc_topk_logits_list, enc_topk_bboxes_list)
            out['pre_outputs'] = {'pred_logits': pre_logits, 'pred_boxes': pre_bboxes}
            out['enc_meta'] = {'class_agnostic': self.query_select_method == 'agnostic'}

            if dn_meta is not None:
                out['dn_outputs'] = self._set_aux_loss2(
                    dn_out_logits, dn_out_bboxes, dn_out_corners, dn_out_refs,
                    dn_out_corners[-1], dn_out_logits[-1])
                out['dn_pre_outputs'] = {'pred_logits': dn_pre_logits, 'pred_boxes': dn_pre_bboxes}
                out['dn_meta'] = dn_meta

        return out
