# --------------------------------------------------------
# SEEM -- Segment Everything Everywhere All At Once
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Xueyan Zou (xueyan@cs.wisc.edu), Jianwei Yang (jianwyan@microsoft.com)
# --------------------------------------------------------


import logging
from typing import Optional

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from timm.models.layers import trunc_normal_
from detectron2.layers import Conv2d
import fvcore.nn.weight_init as weight_init

from .utils.utils import rand_sample, prepare_features
from .utils.attn import MultiheadAttention
from .utils.attention_data_struct import AttentionDataStruct
from .registry import register_decoder
from ...utils import configurable
from ...modules import PositionEmbeddingSine
from ...modules.point_features import point_sample


class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        
        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2, avg_attn = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt, avg_attn

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2, avg_attn = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout(tgt2)

        return tgt, avg_attn

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MultiScaleMaskedTransformerDecoder(nn.Module):

    _version = 2

    @configurable
    def __init__(
        self,
        lang_encoder: nn.Module,
        in_channels,
        mask_classification=True,
        *,
        hidden_dim: int,
        dim_proj: int,
        num_queries: int,
        contxt_len: int,
        nheads: int,
        dim_feedforward: int,
        dec_layers: int,
        pre_norm: bool,
        mask_dim: int,
        task_switch: dict,
        enforce_input_project: bool,
        max_spatial_len: int,
        attn_arch: dict,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
        super().__init__()
        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        
        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.contxt_len = contxt_len
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        # learnable positive negative indicator
        self.pn_indicator = nn.Embedding(2, hidden_dim)
        
        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        self.task_switch = task_switch
        self.query_index = {}

        # output FFNs
        self.lang_encoder = lang_encoder
        if self.task_switch['mask']:
            self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

        self.class_embed = nn.Parameter(torch.empty(hidden_dim, dim_proj))
        trunc_normal_(self.class_embed, std=.02)

        if task_switch['bbox']:
            self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        if task_switch['spatial']:
            # spatial query
            self.mask_sptial_embed = nn.ParameterList([nn.Parameter(torch.empty(hidden_dim, hidden_dim)) for x in range(3)])
            trunc_normal_(self.mask_sptial_embed[0], std=.02)
            trunc_normal_(self.mask_sptial_embed[1], std=.02)
            trunc_normal_(self.mask_sptial_embed[2], std=.02)

            self.max_spatial_len = max_spatial_len
            # spatial memory
            num_spatial_memories = attn_arch['SPATIAL_MEMORIES']
            self.spatial_embed = nn.Embedding(num_spatial_memories, hidden_dim)
            self.spatial_featured = nn.Embedding(num_spatial_memories, hidden_dim)

        # build AttentionDataStruct
        attn_arch['NUM_LAYERS'] = self.num_layers
        self.attention_data = AttentionDataStruct(attn_arch, task_switch)

    @classmethod
    def from_config(cls, cfg, in_channels, lang_encoder, mask_classification, extra):
        ret = {}

        ret["lang_encoder"] = lang_encoder
        ret["in_channels"] = in_channels
        ret["mask_classification"] = mask_classification

        enc_cfg = cfg['MODEL']['ENCODER']
        dec_cfg = cfg['MODEL']['DECODER']

        ret["hidden_dim"] = dec_cfg['HIDDEN_DIM']
        ret["dim_proj"] = cfg['MODEL']['DIM_PROJ']
        ret["num_queries"] = dec_cfg['NUM_OBJECT_QUERIES']
        ret["contxt_len"] = cfg['MODEL']['TEXT']['CONTEXT_LENGTH']

        # Transformer parameters:
        ret["nheads"] = dec_cfg['NHEADS']
        ret["dim_feedforward"] = dec_cfg['DIM_FEEDFORWARD']

        # NOTE: because we add learnable query features which requires supervision,
        # we add minus 1 to decoder layers to be consistent with our loss
        # implementation: that is, number of auxiliary losses is always
        # equal to number of decoder layers. With learnable query features, the number of
        # auxiliary losses equals number of decoders plus 1.
        assert dec_cfg['DEC_LAYERS'] >= 1
        ret["dec_layers"] = dec_cfg['DEC_LAYERS'] - 1
        ret["pre_norm"] = dec_cfg['PRE_NORM']
        ret["enforce_input_project"] = dec_cfg['ENFORCE_INPUT_PROJ']
        ret["mask_dim"] = enc_cfg['MASK_DIM']
        ret["task_switch"] = extra['task_switch']
        ret["max_spatial_len"] = dec_cfg['MAX_SPATIAL_LEN']

        # attn data struct
        ret["attn_arch"] = cfg['ATTENTION_ARCH']

        return ret

    def forward(self, x, mask_features, mask=None, target_queries=None, target_vlp=None, task='seg', extra={}):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels; del mask
        spatial_extra_flag = 'spatial_query_pos_mask' in extra.keys() or task == 'refimg'
        grounding_extra_flag = 'grounding_tokens' in extra.keys()
        visual_extra_flag = 'visual_query_pos' in extra.keys()
        audio_extra_flag = 'audio_tokens' in extra.keys()
        spatial_memory_flag = 'prev_mask' in extra.keys()
        flags = {"spatial": spatial_extra_flag, "grounding": grounding_extra_flag, "memories_spatial": spatial_memory_flag, "visual": visual_extra_flag, "audio": audio_extra_flag}
        self.attention_data.reset(flags, task, extra)

        src, pos, size_list = prepare_features(x, self.num_feature_levels, self.pe_layer, self.input_proj, self.level_embed)
        _, bs, _ = src[0].shape

        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
        self.attention_data.set('queries_object', 'queries', output, query_embed)

        if self.task_switch['spatial'] and spatial_extra_flag:
            # get divisor
            _,h,w = extra['spatial_query_pos_mask'][0].shape
            divisor = torch.tensor([h,w], device=output.device)[None,]

            # Get mean pos spatial query
            non_zero_pos_point = [rand_sample((m.nonzero()[:,1:]/divisor).t(), self.max_spatial_len[-1]).t() for m in extra['spatial_query_pos_mask']]
            non_zero_pos_point = nn.utils.rnn.pad_sequence(non_zero_pos_point, padding_value=-1).permute(1,0,2)
            non_zero_pos_mask = (non_zero_pos_point.sum(dim=-1) < 0)
            spatial_query_pos = point_sample(mask_features, non_zero_pos_point.flip(dims=(2,)).type(mask_features.dtype), align_corners=True)
            spatial_query_pos = torch.stack([x[m].mean(dim=0, keepdim=True) for x, m in zip(spatial_query_pos.transpose(1,2), ~non_zero_pos_mask)]).transpose(0,1).nan_to_num()

            # Get mean neg spatial query
            non_zero_neg_point = [rand_sample((m.nonzero()[:,1:]/divisor).t(), self.max_spatial_len[-1]).t() for m in extra['spatial_query_neg_mask']]
            non_zero_neg_point = nn.utils.rnn.pad_sequence(non_zero_neg_point, padding_value=-1).permute(1,0,2)
            non_zero_neg_mask = (non_zero_neg_point.sum(dim=-1) < 0)
            spatial_query_neg = point_sample(mask_features, non_zero_neg_point.flip(dims=(2,)).type(mask_features.dtype), align_corners=True)
            spatial_query_neg = torch.stack([x[m].mean(dim=0, keepdim=True) for x, m in zip(spatial_query_neg.transpose(1,2), ~non_zero_neg_mask)]).transpose(0,1).nan_to_num()

            # merge positive and negative sample points for self attention

            # Get layerwise spatial query
            src_spatial_queries = []
            src_spatial_maskings = []
            for i in range(len(src)):
                hw,_,dc = src[i].shape
                src_mask_features = src[i].view(size_list[i][0],size_list[i][1],bs,dc)
                src_mask_features = src_mask_features @ self.mask_sptial_embed[i]

                non_zero_query_point_pos = [rand_sample((m.nonzero()[:,1:]/divisor).t(), self.max_spatial_len[i]).t() for m in extra['spatial_query_pos_mask']]
                non_zero_query_point_neg = [rand_sample((m.nonzero()[:,1:]/divisor).t(), self.max_spatial_len[i]).t() for m in extra['spatial_query_neg_mask']]
                non_zero_query_point = [torch.cat([x,y], dim=0) for x,y in zip(non_zero_query_point_pos, non_zero_query_point_neg)]

                pos_neg_indicator = [torch.cat([torch.ones(x.shape[0], device=x.device), -torch.ones(y.shape[0], device=y.device)]) for x,y in zip(non_zero_query_point_pos, non_zero_query_point_neg)]
                pos_neg_indicator = nn.utils.rnn.pad_sequence(pos_neg_indicator, padding_value=0)

                non_zero_query_point = nn.utils.rnn.pad_sequence(non_zero_query_point, padding_value=-1).permute(1,0,2)
                non_zero_query_mask = (non_zero_query_point.sum(dim=-1) < 0)
                non_zero_query_point[non_zero_query_mask] = 0

                spatial_tokens = point_sample(src_mask_features.permute(2,3,0,1), non_zero_query_point.flip(dims=(2,)).type(src_mask_features.dtype), align_corners=True).permute(2,0,1)
                spatial_tokens[pos_neg_indicator==1] += self.pn_indicator.weight[0:1]
                spatial_tokens[pos_neg_indicator==-1] += self.pn_indicator.weight[1:2]

                src_spatial_queries += [spatial_tokens]
                src_spatial_maskings += [non_zero_query_mask]

            if 'refimg' in task:
                output_refimg = {}
                output_refimg['visual_query_pos'] = spatial_query_pos
                output_refimg['visual_query_neg'] = spatial_query_neg
                output_refimg['src_visual_queries'] = src_spatial_queries
                output_refimg['src_visual_maskings'] = src_spatial_maskings
                return output_refimg

            if task != 'demo':
                # Get object query for spatial index
                self.attention_data.set('queries_spatial', 'queries')

        if self.task_switch['visual'] and visual_extra_flag:
            visual_query_pos = extra['visual_query_pos']
            visual_query_neg = extra['visual_query_neg']
            src_visual_queries = extra['src_visual_queries']
            src_visual_maskings = extra['src_visual_maskings']

        if self.task_switch['grounding'] and grounding_extra_flag:
            # Get grounding tokens
            grounding_tokens = extra['grounding_tokens']
            _grounding_tokens = grounding_tokens.detach().clone()

            self.attention_data.set('tokens_grounding', 'tokens', grounding_tokens, _grounding_tokens)
            self.attention_data.set_maskings('tokens_grounding', extra['grounding_nonzero_mask'])

        if self.task_switch['audio'] and audio_extra_flag:
            # Get grounding tokens
            grounding_tokens = extra['audio_tokens']
            _grounding_tokens = grounding_tokens.detach().clone()

            self.attention_data.set('tokens_audio', 'tokens', grounding_tokens, _grounding_tokens)
            self.attention_data.set_maskings('tokens_audio', extra['audio_nonzero_mask'])

        output, query_embed = self.attention_data.cross_attn_variables()
        # prediction heads on learnable query features
        results = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[0])
        results["predictions_pos_spatial"] = spatial_query_pos.transpose(0,1) if spatial_extra_flag else None
        results["predictions_neg_spatial"] = spatial_query_neg.transpose(0,1) if spatial_extra_flag else None
        results["predictions_pos_visual"] = visual_query_pos.transpose(0,1) if visual_extra_flag else None
        results["predictions_neg_visual"] = visual_query_neg.transpose(0,1) if visual_extra_flag else None
        self.attention_data.set_results(results)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            # CROSS ATTENTION
            output, avg_attn = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=self.attention_data.cross_attn_mask(size_list[level_index], self.num_heads),
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], query_pos=query_embed
            )
            self.attention_data.update_variables(output, 'cross_attn')

            # SELF ATTENTION
            self_attn_mask = torch.zeros((bs, self.num_queries, self.num_queries), device=query_embed.device).bool() # Default False (attend oq)
            if self.task_switch['spatial'] and spatial_extra_flag:
                # get spatial tokens
                spatial_tokens = src_spatial_queries[level_index]
                _spatial_tokens = spatial_tokens.detach().clone()

                self.attention_data.set('tokens_spatial', 'tokens', spatial_tokens, _spatial_tokens)
                self.attention_data.set_maskings('tokens_spatial', src_spatial_maskings[level_index])

            if self.task_switch['visual'] and visual_extra_flag:
                # get spatial tokens
                visual_tokens = src_visual_queries[level_index]
                _visual_tokens = visual_tokens.detach().clone()

                self.attention_data.set('tokens_visual', 'tokens', visual_tokens, _visual_tokens)
                self.attention_data.set_maskings('tokens_visual', src_visual_maskings[level_index])

            output, query_embed, self_attn_mask = self.attention_data.self_attn(bs, self.num_heads)
            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=self_attn_mask,
                tgt_key_padding_mask=None,
                query_pos=query_embed)

            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )

            self.attention_data.update_variables(output, 'self_attn')
            output, query_embed = self.attention_data.cross_attn_variables()
            results = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels], layer_id=i)
            results["predictions_pos_spatial"] = spatial_query_pos.transpose(0,1) if spatial_extra_flag else None
            results["predictions_neg_spatial"] = spatial_query_neg.transpose(0,1) if spatial_extra_flag else None
            results["predictions_pos_visual"] = visual_query_pos.transpose(0,1) if visual_extra_flag else None
            results["predictions_neg_visual"] = visual_query_neg.transpose(0,1) if visual_extra_flag else None
            self.attention_data.set_results(results)

        return self.attention_data.organize_output()

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size, layer_id=-1):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        class_embed = decoder_output @ self.class_embed
        outputs_class = self.lang_encoder.compute_similarity(class_embed)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
        
        outputs_bbox = [None for i in range(len(outputs_mask))]
        if self.task_switch['bbox']:
            outputs_bbox = self.bbox_embed(decoder_output)

        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)

        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()

        outputs_caption = class_embed

        results = {
            "attn_mask": attn_mask,
            "predictions_class": outputs_class,
            "predictions_mask": outputs_mask,
            "predictions_bbox": outputs_bbox,
            "predictions_caption": outputs_caption,
            "predictions_maskemb": mask_embed,
        }
        return results


@register_decoder
def get_masked_transformer_decoder(cfg, in_channels, lang_encoder, mask_classification, extra):
    return MultiScaleMaskedTransformerDecoder(cfg, in_channels, lang_encoder, mask_classification, extra)