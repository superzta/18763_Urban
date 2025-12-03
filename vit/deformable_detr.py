
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
from scipy.optimize import linear_sum_assignment
import math
import copy
import yaml
import os
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Optional, List, Dict
from torchvision.transforms.functional import to_tensor


# Import from rcnn to reuse config and dataset
from rcnn import (
    Config,
    MultiClassUrbanDataset,
    collate_fn,
    compute_iou,
    plot_training_curves,
    visualize_predictions,
    calculate_map,
    create_evaluation_visualizations as rcnn_create_eval_viz, 
)

from PIL import Image

# ============================================================================
# Deformable Attention Module (Pure PyTorch Implementation)
# ============================================================================
# Utils
# ============================================================================

def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps, max=1-eps)
    return torch.log(x1 / (1 - x1))

def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = 0.25 (for positive).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes

class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads')
        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        nn.init.constant_(self.attention_weights.weight.data, 0.)
        nn.init.constant_(self.attention_weights.bias.data, 0.)
        nn.init.xavier_uniform_(self.value_proj.weight.data)
        nn.init.constant_(self.value_proj.bias.data, 0.)
        nn.init.xavier_uniform_(self.output_proj.weight.data)
        nn.init.constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None):
        """
        :param query: (N, Length_{query}, C)
        :param reference_points: (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1,1), including padding area
                                 or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten: (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes: (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index: (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, ...]
        :param input_padding_mask: (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding
        :return: output: (N, Length_{query}, C)
        """
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
        
        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            
            # Handle both 3D (decoder) and 4D (encoder) reference points
            if reference_points.dim() == 3:
                # Decoder: (BS, Lq, 2) -> expand to (BS, Lq, 1, 2) -> (BS, Lq, n_levels, 2)
                n_levels = input_spatial_shapes.shape[0]
                reference_points_expanded = reference_points[:, :, None, :].expand(-1, -1, n_levels, -1)
                sampling_locations = reference_points_expanded[:, :, None, :, None, :] \
                                     + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            else:
                # Encoder: (BS, Lq, n_levels, 2)
                sampling_locations = reference_points[:, :, None, :, None, :] \
                                     + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            if reference_points.dim() == 3:
                n_levels = input_spatial_shapes.shape[0]
                reference_points = reference_points[:, :, None, :].expand(-1, -1, n_levels, -1)
                
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(f'Last dim of reference_points must be 2 or 4, but get {reference_points.shape[-1]}')
        
        output = self.ms_deform_attn_core_pytorch(value, input_spatial_shapes, sampling_locations, attention_weights)
        output = self.output_proj(output)
        return output

    def ms_deform_attn_core_pytorch(self, value, value_spatial_shapes, sampling_locations, attention_weights):
        # Debug sanity checks
        if torch.isnan(value).any() or torch.isinf(value).any():
            raise RuntimeError("NaN/Inf detected in 'value' before grid_sample")
        if torch.isnan(sampling_locations).any() or torch.isinf(sampling_locations).any():
            raise RuntimeError("NaN/Inf detected in 'sampling_locations'")
        if torch.isnan(attention_weights).any() or torch.isinf(attention_weights).any():
            raise RuntimeError("NaN/Inf detected in 'attention_weights'")

        N_, S_, M_, D_ = value.shape
        _, Lq_, M_, L_, P_, _ = sampling_locations.shape

        value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
        sampling_grids = 2 * sampling_locations - 1  # should be roughly in [-1,1], but can go a bit out

        sampling_value_list = []
        for lid_, (H_, W_) in enumerate(value_spatial_shapes):
            value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_*M_, D_, H_, W_)
            sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)

            # Optional: clamp grids to [-1.5, 1.5] just to avoid crazy out-of-range
            sampling_grid_l_ = sampling_grid_l_.clamp(min=-2.0, max=2.0)

            sampling_value_l_ = F.grid_sample(
                value_l_, sampling_grid_l_,
                mode='bilinear', padding_mode='zeros', align_corners=False
            )
            sampling_value_list.append(sampling_value_l_)

        attention_weights = attention_weights.transpose(1, 2).reshape(N_*M_, 1, Lq_, L_*P_)
        output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(N_, M_*D_, Lq_)
        return output.transpose(1, 2).contiguous()


# ============================================================================
# Transformer Components
# ============================================================================

class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = getattr(F, activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = getattr(F, activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask=None):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # cross attention
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos), reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # ffn
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6, num_decoder_layers=6, 
                 dim_feedforward=1024, dropout=0.1, activation="relu", num_feature_levels=4, 
                 dec_n_points=4, enc_n_points=4):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_feature_levels = num_feature_levels

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward, dropout, activation, num_feature_levels, nhead, enc_n_points)
        self.encoder = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_encoder_layers)])

        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward, dropout, activation, num_feature_levels, nhead, dec_n_points)
        self.decoder = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_decoder_layers)])

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        self.reference_points = nn.Linear(d_model, 2)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.level_embed)
        nn.init.xavier_uniform_(self.reference_points.weight.data, gain=1.0)
        nn.init.constant_(self.reference_points.bias.data, 0.)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds, query_embed, bbox_embed=None):
        # assert self.d_model == srcs[0].shape[1]
        
        # ---------------- Encoder prep (unchanged) ----------------
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            
            src = src.flatten(2).transpose(1, 2)   # (bs, h*w, c)
            mask = mask.flatten(1)                 # (bs, h*w)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)  # (bs, h*w, c)
            
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            
            src_flatten.append(src)
            mask_flatten.append(mask)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            
        src_flatten = torch.cat(src_flatten, 1)           # (bs, sum_l H_l*W_l, C)
        mask_flatten = torch.cat(mask_flatten, 1)         # (bs, sum_l H_l*W_l)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)  # (bs, n_levels, 2)

        # ---------------- Encoder (unchanged logic) ----------------
        memory = src_flatten
        for layer in self.encoder:
            ref_points = []
            for lvl, (H, W) in enumerate(spatial_shapes):
                ref_y, ref_x = torch.meshgrid(
                    torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=memory.device),
                    torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=memory.device),
                    indexing='ij'
                )
                ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H)
                ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W)
                ref = torch.stack((ref_x, ref_y), -1)  # (bs, H*W, 2)
                ref_points.append(ref)
            reference_points = torch.cat(ref_points, 1)        # (bs, sum_l H_l*W_l, 2)
            reference_points = reference_points[:, :, None] * valid_ratios[:, None]  # (bs, S, n_levels, 2)
            
            memory = layer(
                memory,
                lvl_pos_embed_flatten,
                reference_points,
                spatial_shapes,
                level_start_index,
                mask_flatten,
            )

        # ---------------- Decoder ----------------
        bs, _, c = memory.shape
        query_embed, tgt = torch.split(query_embed, c, dim=1)  # (num_queries, C) each
        query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)  # (bs, num_queries, C)
        tgt = tgt.unsqueeze(0).expand(bs, -1, -1)                 # (bs, num_queries, C)
        
        reference_points = self.reference_points(query_embed).sigmoid()  # (bs, num_queries, 2)
        init_reference_out = reference_points
        
        hs = []
        inter_references = []
        
        for i, layer in enumerate(self.decoder):
            tgt = layer(
                tgt,
                query_embed,
                reference_points,
                memory,
                spatial_shapes,
                level_start_index,
                mask_flatten,
            )
            hs.append(tgt)
            
            if bbox_embed is not None:
                # Predict offsets
                tmp = bbox_embed[i](tgt)
                
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    # First layer: reference_points is 2D
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                
                reference_points = new_reference_points.detach()
                inter_references.append(new_reference_points)
            
        # hs: (num_layers, bs, num_queries, C)
        hs = torch.stack(hs, dim=0)
        
        if bbox_embed is not None:
            inter_references = torch.stack(inter_references, dim=0)
            return hs, init_reference_out, inter_references
        else:
            return hs, init_reference_out

class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        # Use C3, C4, C5
        self.body = nn.ModuleDict({
            'layer2': backbone.layer2, # C3
            'layer3': backbone.layer3, # C4
            'layer4': backbone.layer4  # C5
        })
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1
        )
        self.num_channels = [512, 1024, 2048]

    def forward(self, x):
        xs = {}
        x = self.stem(x)
        x = self.body['layer2'](x)
        xs['0'] = x
        x = self.body['layer3'](x)
        xs['1'] = x
        x = self.body['layer4'](x)
        xs['2'] = x
        return xs

class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, mask):
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

# ============================================================================
# Deformable DETR Model
# ============================================================================

class DeformableDETR(nn.Module):
    def __init__(self, num_classes, num_queries=300, aux_loss=False):
        super().__init__()
        self.num_queries = num_queries
        self.aux_loss = aux_loss
        self.hidden_dim = 256
        
        # Backbone
        self.backbone = Backbone()
        self.pos_trans = nn.Conv2d(2048, 256, 1) # For C5 to match hidden dim? No, we use multiple levels
        
        # Input projections
        self.input_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(512, 256, 1),
                nn.GroupNorm(32, 256),
            ),
            nn.Sequential(
                nn.Conv2d(1024, 256, 1),
                nn.GroupNorm(32, 256),
            ),
            nn.Sequential(
                nn.Conv2d(2048, 256, 1),
                nn.GroupNorm(32, 256),
            ),
        ])
        # Extra level (C6)
        self.input_proj.append(
            nn.Sequential(
                nn.Conv2d(2048, 256, 3, 2, 1),
                nn.GroupNorm(32, 256),
            )
        )
        
        self.transformer = DeformableTransformer(d_model=256, num_feature_levels=4)
        
        # Class Embed: ModuleList (one per layer)
        self.class_embed = nn.ModuleList([
            nn.Linear(256, num_classes) for _ in range(len(self.transformer.decoder))
        ])
        
        # BBox Embed: ModuleList (one per layer)
        self.bbox_embed = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 4)
            ) for _ in range(len(self.transformer.decoder))
        ])
        
        self.query_embed = nn.Embedding(num_queries, 256 * 2)
        
        self.pos_embed = PositionEmbeddingSine(num_pos_feats=128, normalize=True)
        
        # Initialization
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        
        for class_embed_layer in self.class_embed:
            class_embed_layer.bias.data = torch.ones(num_classes) * bias_value
            
        for bbox_embed_layer in self.bbox_embed:
            nn.init.constant_(bbox_embed_layer[-1].weight.data, 0)
            nn.init.constant_(bbox_embed_layer[-1].bias.data, 0)
        
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # Matcher and Criterion
        self.matcher = HungarianMatcher(
            cost_class=2.0,   # Focal loss cost weight
            cost_bbox=5.0,
            cost_giou=2.0,
        )
        self.criterion = SetCriterion(num_classes, matcher=self.matcher, weight_dict={'loss_ce': 2.0, 'loss_bbox': 5.0, 'loss_giou': 2.0})
        
        if aux_loss:
            aux_weight_dict = {}
            for i in range(len(self.transformer.decoder) - 1):
                aux_weight_dict.update({k + f'_{i}': v for k, v in self.criterion.weight_dict.items()})
            self.criterion.weight_dict.update(aux_weight_dict)

    def forward(self, samples: List[torch.Tensor], targets: Optional[List[Dict]] = None):
        """
        samples: list of images
        targets: list of dicts (boxes, labels)
        """
        # Handle batching (pad images to same size)
        # samples is a list of tensors. We need to stack them with padding.
        # But we can use nested tensor approach or just pad.
        
        # Find max size
        max_h = max([img.shape[1] for img in samples])
        max_w = max([img.shape[2] for img in samples])
        
        # Pad to multiple of 32
        stride = 32
        max_h = (max_h + stride - 1) // stride * stride
        max_w = (max_w + stride - 1) // stride * stride
        
        batch_size = len(samples)
        padded_imgs = samples[0].new_zeros(batch_size, 3, max_h, max_w)
        masks = samples[0].new_ones(batch_size, max_h, max_w, dtype=torch.bool) # 1 is padding
        
        for i, img in enumerate(samples):
            padded_imgs[i, :, :img.shape[1], :img.shape[2]] = img
            masks[i, :img.shape[1], :img.shape[2]] = False
            
        # Extract features
        features = self.backbone(padded_imgs)
        
        srcs = []
        pos_embeds = []
        input_masks = []
        
        # Prepare multi-scale features
        # Level 0: C3 (layer2)
        srcs.append(self.input_proj[0](features['0']))
        m = F.interpolate(masks.unsqueeze(1).float(), size=srcs[-1].shape[-2:]).squeeze(1).to(torch.bool)
        input_masks.append(m)
        pos_embeds.append(self.pos_embed(m))
        
        # Level 1: C4 (layer3)
        srcs.append(self.input_proj[1](features['1']))
        m = F.interpolate(masks.unsqueeze(1).float(), size=srcs[-1].shape[-2:]).squeeze(1).to(torch.bool)
        input_masks.append(m)
        pos_embeds.append(self.pos_embed(m))
        
        # Level 2: C5 (layer4)
        srcs.append(self.input_proj[2](features['2']))
        m = F.interpolate(masks.unsqueeze(1).float(), size=srcs[-1].shape[-2:]).squeeze(1).to(torch.bool)
        input_masks.append(m)
        pos_embeds.append(self.pos_embed(m))
        
        # Level 3: C6 (generated from C5)
        srcs.append(self.input_proj[3](features['2']))
        m = F.interpolate(masks.unsqueeze(1).float(), size=srcs[-1].shape[-2:]).squeeze(1).to(torch.bool)
        input_masks.append(m)
        pos_embeds.append(self.pos_embed(m))
        
        # Transformer
        # We pass bbox_embed to transformer for iterative refinement
        hs, init_reference, inter_references = self.transformer(srcs, input_masks, pos_embeds, self.query_embed.weight, self.bbox_embed)
        
        # Output heads
        outputs_classes = []
        outputs_coords = []
        
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            
            outputs_class = self.class_embed[lvl](hs[lvl])
            outputs_classes.append(outputs_class)
            
            # For boxes, we already have inter_references which ARE the predicted boxes
            outputs_coords.append(inter_references[lvl])
            
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        
        if targets is not None:
            # Convert targets to DETR format (cx, cy, w, h normalized)
            detr_targets = []
            for i, t in enumerate(targets):
                # Get valid h, w for this image
                valid_h = (~masks[i, :, 0]).sum()
                valid_w = (~masks[i, 0, :]).sum()
                
                boxes = t['boxes']
                labels = t['labels']
                
                # Convert xyxy to cxcywh normalized
                if len(boxes) > 0:
                    cx = (boxes[:, 0] + boxes[:, 2]) / 2 / valid_w
                    cy = (boxes[:, 1] + boxes[:, 3]) / 2 / valid_h
                    w = (boxes[:, 2] - boxes[:, 0]) / valid_w
                    h = (boxes[:, 3] - boxes[:, 1]) / valid_h
                    cxcywh = torch.stack([cx, cy, w, h], dim=1)
                else:
                    cxcywh = torch.zeros((0, 4), device=boxes.device)
                
                detr_targets.append({'boxes': cxcywh, 'labels': labels})
            
            loss_dict = self.criterion(out, detr_targets)
            return loss_dict
            
        else:
            # Inference
            pred_logits = out['pred_logits']
            pred_boxes = out['pred_boxes']
            
            # For Sigmoid Focal Loss, scores are just sigmoids
            probas = pred_logits.sigmoid()
            
            # Top-k or thresholding
            # We'll return all and let caller filter
            
            results = []
            for i in range(batch_size):
                valid_h = (~masks[i, :, 0]).sum()
                valid_w = (~masks[i, 0, :]).sum()
                
                p_boxes = pred_boxes[i]
                p_logits = pred_logits[i]
                
                # Convert to xyxy
                cx, cy, w, h = p_boxes.unbind(-1)
                x1 = (cx - 0.5 * w) * valid_w
                y1 = (cy - 0.5 * h) * valid_h
                x2 = (cx + 0.5 * w) * valid_w
                y2 = (cy + 0.5 * h) * valid_h
                
                boxes = torch.stack([x1, y1, x2, y2], dim=-1)
                
                # Get scores and labels
                probs = p_logits.sigmoid()
                scores, labels = probs.max(-1)
                # Labels are 0-based index of object classes
                # If we want 1-based (matching RCNN dataset), we add 1
                labels = labels + 1 
                
                results.append({'boxes': boxes, 'labels': labels, 'scores': scores})
                
            return results

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict(str: list) and dict(str: float).
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

# ============================================================================
# Matcher and Criterion
# ============================================================================

class HungarianMatcher(nn.Module):
    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost.
        alpha = 0.25
        gamma = 2.0
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost between boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict, losses=['labels', 'boxes', 'cardinality']):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=0.25, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        num_boxes = torch.clamp(num_boxes / 1, min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            if loss == 'labels':
                losses.update(self.loss_labels(outputs, targets, indices, num_boxes))
            elif loss == 'boxes':
                losses.update(self.loss_boxes(outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'labels':
                        l_dict = self.loss_labels(aux_outputs, targets, indices, num_boxes)
                        losses.update({k + f'_{i}': v for k, v in l_dict.items()})
                    elif loss == 'boxes':
                        l_dict = self.loss_boxes(aux_outputs, targets, indices, num_boxes)
                        losses.update({k + f'_{i}': v for k, v in l_dict.items()})

        # Apply weights
        weighted_losses = {}
        for k, v in losses.items():
            if k in self.weight_dict:
                weighted_losses[k] = v * self.weight_dict[k]
        
        return weighted_losses

# ============================================================================
# Utils
# ============================================================================

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def box_iou(boxes1, boxes2):
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union

def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/
    The boxes should be in [x0, y0, x1, y1] format
    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area

# ============================================================================
# Main Functions
# ============================================================================

def get_model(num_classes):
    return DeformableDETR(num_classes=num_classes)

def plot_training_curves_detr(history, save_dir: str):
    """
    Plot training & validation curves for Deformable DETR.
    Expects each history element to contain:
      - 'epoch'
      - 'total_loss', 'loss_ce', 'loss_bbox', 'loss_giou'
      - 'val_total_loss', 'val_loss_ce', 'val_loss_bbox', 'val_loss_giou'
    """
    if not history:
        print("No history to plot.")
        return

    epochs = [h['epoch'] for h in history]

    train_total = [h['total_loss'] for h in history]
    val_total   = [h['val_total_loss'] for h in history]

    train_ce    = [h['loss_ce'] for h in history]
    val_ce      = [h['val_loss_ce'] for h in history]

    train_bbox  = [h['loss_bbox'] for h in history]
    val_bbox    = [h['val_loss_bbox'] for h in history]

    train_giou  = [h['loss_giou'] for h in history]
    val_giou    = [h['val_loss_giou'] for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Total loss
    ax = axes[0, 0]
    ax.plot(epochs, train_total, label="train")
    ax.plot(epochs, val_total,   label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total loss")
    ax.set_title("Total loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # CE (classification) loss
    ax = axes[0, 1]
    ax.plot(epochs, train_ce, label="train")
    ax.plot(epochs, val_ce,   label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("CE loss")
    ax.set_title("Classification loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # BBox L1 loss
    ax = axes[1, 0]
    ax.plot(epochs, train_bbox, label="train")
    ax.plot(epochs, val_bbox,   label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BBox L1 loss")
    ax.set_title("BBox regression loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # GIoU loss
    ax = axes[1, 1]
    ax.plot(epochs, train_giou, label="train")
    ax.plot(epochs, val_giou,   label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("GIoU loss")
    ax.set_title("GIoU loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "training_curves_detr.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Deformable DETR training curves saved to {save_path}")


def train(config: Config):
    print("=" * 80)
    print("Starting Training (Deformable DETR)")
    print("=" * 80)

    # Make sure dirs exist
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.results_dir, exist_ok=True)

    # Load main config for class names
    with open(config.main_config_yaml, 'r') as f:
        main_config = yaml.safe_load(f)

    # Build class names list: background + selected classes
    selected_class_names = ['background']
    for cls_id in config.urban_issue_classes:
        selected_class_names.append(main_config['names'][cls_id])

    print(f"Training on {len(config.urban_issue_classes)} urban issue class(es):")
    for cls_id in config.urban_issue_classes:
        print(f"  Class {cls_id}: {main_config['names'][cls_id]}")
    print(f"\nModel classes: {selected_class_names}")
    print(f"Number of classes (including background): {config.num_classes}\n")

    # Create datasets
    print("Loading datasets...")
    train_dataset = MultiClassUrbanDataset(config, split='train')
    valid_dataset = MultiClassUrbanDataset(config, split='valid')

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    # Create model
    print(f"\nCreating Deformable DETR model on device: {config.device}")
    # For Sigmoid Focal Loss, we use N classes (0..N-1). 
    # config.num_classes includes background (N+1), so we subtract 1.
    model = get_model(config.num_classes - 1)
    model.to(config.device)

    # Optimizer / scheduler
    param_dicts = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if "backbone" not in n and p.requires_grad
            ]
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if "backbone" in n and p.requires_grad
            ],
            "lr": config.learning_rate * 0.1,
        },
    ]
    optimizer = torch.optim.AdamW(
        param_dicts,
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.lr_scheduler_step_size,
        gamma=config.lr_scheduler_gamma
    )

    warmup_epochs = getattr(config, "warmup_epochs", 0)  # 0 = no warmup if not set
    base_lrs = [g["lr"] for g in optimizer.param_groups]

    print(f"\nStarting training for {config.num_epochs} epochs...")
    training_history = []

    best_val_loss = float('inf')
    best_ckpt_path = os.path.join(config.checkpoint_dir, 'deformable_detr_best.pth')

    for epoch in range(1, config.num_epochs + 1):
        # ----------------- Train -----------------
        epoch_stats = train_one_epoch_detr(
            model, optimizer, train_loader, config.device, epoch, config.print_freq
        )

        if warmup_epochs > 0 and epoch <= warmup_epochs:
            # Linear warmup from 0 → base_lr over warmup_epochs
            warmup_factor = float(epoch) / float(max(1, warmup_epochs))
            for g, base_lr in zip(optimizer.param_groups, base_lrs):
                g["lr"] = base_lr * warmup_factor
        else:
            lr_scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']

        print(f"\nEpoch {epoch}/{config.num_epochs}")
        print(f"  LR: {current_lr:.6f}")
        print(f"  Train Total Loss: {epoch_stats['total_loss']:.4f}")
        print(f"  Train CE Loss: {epoch_stats['loss_ce']:.4f}")
        print(f"  Train BBox Loss: {epoch_stats['loss_bbox']:.4f}")
        print(f"  Train GIoU Loss: {epoch_stats['loss_giou']:.4f}")

        # ----------------- Validate -----------------
        val_stats = eval_one_epoch_detr(model, valid_loader, config.device)
        print(f"  Val   Total Loss: {val_stats['total_loss']:.4f}")
        print(f"  Val   CE Loss: {val_stats['loss_ce']:.4f}")
        print(f"  Val   BBox Loss: {val_stats['loss_bbox']:.4f}")
        print(f"  Val   GIoU Loss: {val_stats['loss_giou']:.4f}")

        epoch_stats['epoch'] = epoch
        epoch_stats['lr'] = current_lr
        epoch_stats['val_total_loss'] = val_stats['total_loss']
        epoch_stats['val_loss_ce'] = val_stats['loss_ce']
        epoch_stats['val_loss_bbox'] = val_stats['loss_bbox']
        epoch_stats['val_loss_giou'] = val_stats['loss_giou']
        training_history.append(epoch_stats)

        # ----------------- Save "regular" checkpoint -----------------
        if epoch % config.save_freq == 0 or epoch == config.num_epochs:
            ckpt_path = os.path.join(
                config.checkpoint_dir,
                f'deformable_detr_epoch_{epoch}.pth'
            )
            tmp_path = ckpt_path + '.tmp'
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': config.__dict__,
                    'val_loss': val_stats['total_loss'],
                },
                tmp_path
            )
            # Atomic replace -> avoids half-written/corrupted files
            os.replace(tmp_path, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

        # ----------------- Save BEST checkpoint (atomic, non-corrupted) -----------------
        val_loss = val_stats['total_loss']
        if not math.isnan(val_loss) and val_loss < best_val_loss:
            best_val_loss = val_loss
            tmp_best = best_ckpt_path + '.tmp'

            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': config.__dict__,
                    'val_loss': val_loss,
                },
                tmp_best
            )
            # Atomic replace: if the save crashes mid-way, old best file stays intact
            os.replace(tmp_best, best_ckpt_path)
            print(f"New best model (val_loss={val_loss:.4f}). Saved to: {best_ckpt_path}")

    # Save history
    history_path = os.path.join(config.results_dir, 'training_history_detr.json')
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)

    # Plot training curves
    plot_training_curves_detr(training_history, config.results_dir)

    return model


def train_one_epoch_detr(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    running_loss = 0.0
    running_loss_ce = 0.0
    running_loss_bbox = 0.0
    running_loss_giou = 0.0
    successful_batches = 0
    
    
    pbar = tqdm(data_loader, desc=f"Epoch {epoch}")
    
    for i, (images, targets) in enumerate(pbar):
        try:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Shift labels for Focal Loss (1-based -> 0-based)
            for t in targets:
                t['labels'] = t['labels'] - 1
            
            # Filter empty targets? DETR can handle empty targets (no objects)
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            
            running_loss += losses.item()
            running_loss_ce += loss_dict.get('loss_ce', torch.tensor(0.)).item()
            running_loss_bbox += loss_dict.get('loss_bbox', torch.tensor(0.)).item()
            running_loss_giou += loss_dict.get('loss_giou', torch.tensor(0.)).item()
            
            successful_batches += 1
            
            if (successful_batches + 1) % print_freq == 0:
                pbar.set_postfix({
                    'loss': f'{running_loss/successful_batches:.4f}',
                    'ce': f'{running_loss_ce/successful_batches:.4f}',
                    'bbox': f'{running_loss_bbox/successful_batches:.4f}'
                })
                
        except Exception as e:
            import traceback
            print(f"Error: {e}", flush=True)
            print(f"Traceback: {traceback.format_exc()}", flush=True)
            # continue
            
    return {
        'total_loss': running_loss / successful_batches if successful_batches > 0 else 0,
        'loss_ce': running_loss_ce / successful_batches if successful_batches > 0 else 0,
        'loss_bbox': running_loss_bbox / successful_batches if successful_batches > 0 else 0,
        'loss_giou': running_loss_giou / successful_batches if successful_batches > 0 else 0,
        'successful_batches': successful_batches
    }

@torch.no_grad()
def eval_one_epoch_detr(model, data_loader, device):
    model.eval()
    running_loss = 0.0
    running_loss_ce = 0.0
    running_loss_bbox = 0.0
    running_loss_giou = 0.0
    successful_batches = 0

    pbar = tqdm(data_loader, desc="Valid")

    for images, targets in pbar:
        try:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Shift labels for Focal Loss (1-based -> 0-based)
            for t in targets:
                t['labels'] = t['labels'] - 1

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            running_loss += losses.item()
            running_loss_ce += loss_dict.get('loss_ce', torch.tensor(0.0)).item()
            running_loss_bbox += loss_dict.get('loss_bbox', torch.tensor(0.0)).item()
            running_loss_giou += loss_dict.get('loss_giou', torch.tensor(0.0)).item()

            successful_batches += 1

            if successful_batches > 0:
                pbar.set_postfix({
                    'val_loss': f'{running_loss/successful_batches:.4f}',
                    'val_ce': f'{running_loss_ce/successful_batches:.4f}',
                    'val_bbox': f'{running_loss_bbox/successful_batches:.4f}',
                })
        except Exception as e:
            import traceback
            print(f"[VAL] Error: {e}", flush=True)
            print(traceback.format_exc(), flush=True)

    if successful_batches == 0:
        return {
            'total_loss': math.inf,
            'loss_ce': math.inf,
            'loss_bbox': math.inf,
            'loss_giou': math.inf,
            'successful_batches': 0,
        }

    return {
        'total_loss': running_loss / successful_batches,
        'loss_ce': running_loss_ce / successful_batches,
        'loss_bbox': running_loss_bbox / successful_batches,
        'loss_giou': running_loss_giou / successful_batches,
        'successful_batches': successful_batches,
    }

def evaluate_from_dicts(
    pred_dict,
    gt_dict,
    iou_threshold: float = 0.5,
    is_proxy: bool = False,
):
    """
    Convert dict-of-dicts with image_id keys into the list format expected by
    `calculate_map` in rcnn.py and compute evaluation metrics.

    pred_dict: {img_id: {"boxes": Tensor[N,4], "labels": Tensor[N], "scores": Tensor[N]}}
    gt_dict:   {img_id: {"boxes": Tensor[M,4], "labels": Tensor[M]}}
    """
    # Make sure we iterate in a consistent order
    img_ids = sorted(gt_dict.keys())

    predictions = []
    ground_truths = []

    for img_id in img_ids:
        # Ground truth is guaranteed to exist for this id
        gt = gt_dict[img_id]

        # If there are no predictions for this id, create an empty prediction
        pred = pred_dict.get(
            img_id,
            {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "scores": torch.zeros((0,), dtype=torch.float32),
            },
        )

        ground_truths.append(gt)
        predictions.append(pred)

    # Use your existing mAP implementation from rcnn.py
    results = calculate_map(
        predictions,
        ground_truths,
        conf_threshold=config.conf_threshold,       # you can also parameterize this if needed
        iou_threshold=iou_threshold,
        use_proxy=is_proxy,
    )
    return results



@torch.no_grad()
def evaluate_detr(model, data_loader, device, config):
    """
    Evaluate Deformable DETR in EXACTLY the same way as RCNN:
      - collect lists: all_images, all_predictions, all_ground_truths
      - call the same calculate_map()
      - later reuse rcnn_create_eval_viz() for GT vs Pred visualizations
    """
    model.eval()

    all_images = []
    all_predictions = []
    all_ground_truths = []

    print("=" * 80)
    print("Evaluating Deformable DETR")
    print("=" * 80)

    for images, targets in tqdm(data_loader, desc="Evaluating"):
        # store raw images for visualization (on CPU)
        all_images.extend([img.cpu() for img in images])

        # move to device
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # forward – inference branch of DeformableDETR returns List[dict]
        outputs = model(images)  # list[dict]

        filtered_outputs = []
        for out in outputs:
            scores = out["scores"]
            keep = scores > config.conf_threshold   # e.g. 0.5

            filtered_outputs.append({
                "boxes":  out["boxes"][keep].detach().cpu(),
                "labels": out["labels"][keep].detach().cpu(),
                "scores": scores[keep].detach().cpu(),
            })

        targets = [{k: v.detach().cpu() for k, v in t.items()} for t in targets]

        all_predictions.extend(filtered_outputs)
        all_ground_truths.extend(targets)


    # Use the SAME mAP implementation as RCNN.py
    results = calculate_map(
        all_predictions,
        all_ground_truths,
        conf_threshold=config.conf_threshold,
        iou_threshold=0.5,
        use_proxy=config.use_proxy_map,
    )

    # Pretty-print like RCNN
    print("\nEvaluation Results:")
    print(f"  mAP@0.5 ({'Proxy' if results['is_proxy'] else 'Exact'}): {results['mAP@0.5']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall: {results['recall']:.4f}")
    print(f"  Total Predictions: {results['total_predictions']}")
    print(f"  Total Ground Truths: {results['total_ground_truths']}")

    # Save JSON to same place / format as RCNN
    results_path = os.path.join(config.results_dir, 'evaluation_results_detr.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return results, all_images, all_predictions, all_ground_truths



def test(config: Config, checkpoint_path: str):
    print("=" * 80)
    print("Testing Deformable DETR Model")
    print("=" * 80)

    # Load main config for class names
    with open(config.main_config_yaml, 'r') as f:
        main_config = yaml.safe_load(f)

    class_names = ['background']
    for cls_id in config.urban_issue_classes:
        class_names.append(main_config['names'][cls_id])

    print(f"Testing on {len(config.urban_issue_classes)} urban issue class(es):")
    for cls_id in config.urban_issue_classes:
        print(f"  Class {cls_id}: {main_config['names'][cls_id]}")
    print()

    # Dataset / loader
    print("Loading test dataset...")
    test_dataset = MultiClassUrbanDataset(config, split='test')
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Load model
    print(f"Loading model from {checkpoint_path}...")
    model = get_model(config.num_classes - 1)
    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(config.device)

    # Evaluate
    results, all_images, all_predictions, all_ground_truths = evaluate_detr(
        model, test_loader, config.device, config
    )

    # Create visualizations EXACTLY as in rcnn.py
    print("\nCreating evaluation visualizations...")
    rcnn_create_eval_viz(
        all_images,
        all_predictions,
        all_ground_truths,
        config,
        num_samples=10,   # same default as RCNN
    )

    return results



def create_evaluation_visualizations(dataset, predictions_dict, class_names, output_dir, max_images=20):
    os.makedirs(output_dir, exist_ok=True)

    from torchvision.transforms.functional import to_pil_image
    import torch

    for idx in range(min(len(dataset), max_images)):
        img, target = dataset[idx]
        img_id = int(target["image_id"].item())

        preds = predictions_dict.get(img_id, None)
        if preds is None:
            continue

        boxes = preds["boxes"].numpy()
        labels = preds["labels"].numpy()
        scores = preds["scores"].numpy()

        if isinstance(img, torch.Tensor):
            img_vis = to_pil_image(img)   # (C,H,W) in [0,1] → PIL
        else:
            img_vis = img                 # already PIL

        save_path = os.path.join(output_dir, f"eval_comparison_{img_id}.png")
        visualize_predictions(img_vis, boxes, labels, scores, class_names, save_path=save_path)




@torch.no_grad()
def inference(config: Config, checkpoint_path: str, image_path: str, save_path: Optional[str] = None):
    """Run inference on a single image."""
    print("=" * 80)
    print("Running Inference (Deformable DETR)")
    print("=" * 80)
    
    # Load main config for class names
    with open(config.main_config_yaml, 'r') as f:
        main_config = yaml.safe_load(f)
    
    # Build class names list
    class_names = ['background']
    for cls_id in config.urban_issue_classes:
        class_names.append(main_config['names'][cls_id])
    
    # Load model
    print(f"Loading model from {checkpoint_path}...")
    model = get_model(config.num_classes - 1)
    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(config.device)
    model.eval()
    
    # Load image
    image = Image.open(image_path).convert("RGB")
    image_tensor = to_tensor(image).to(config.device)
    
    # Run inference
    print("Running inference...")
    # DeformableDETR returns list of dicts
    predictions = model([image_tensor])[0]
    
    # Filter by confidence
    mask = predictions['scores'] > config.conf_threshold
    boxes = predictions['boxes'][mask].cpu().numpy()
    labels = predictions['labels'][mask].cpu().numpy()
    scores = predictions['scores'][mask].cpu().numpy()
    
    print(f"\nDetected {len(boxes)} objects:")
    for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        class_name = class_names[label]
        print(f"  {i+1}. {class_name}: {score:.3f} - Box: [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")
    
    # Visualize
    visualize_predictions(image, boxes, labels, scores, class_names, save_path)
    
    return boxes, labels, scores

