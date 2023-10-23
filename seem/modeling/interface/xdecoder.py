# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------

import logging
from typing import Optional

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from timm.models.layers import trunc_normal_
from detectron2.layers import Conv2d
import fvcore.nn.weight_init as weight_init

from .build import register_decoder
from .modules import SelfAttentionLayer, CrossAttentionLayer, FFNLayer, MLP
from ..utils import configurable
from ..modules import PositionEmbeddingSine


class XDecoder(nn.Module):

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
        captioning_step: int,
        enforce_input_project: bool,
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

        # output FFNs
        self.lang_encoder = lang_encoder
        if self.task_switch['mask']:
            self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

        self.class_embed = nn.Parameter(torch.empty(hidden_dim, dim_proj))
        trunc_normal_(self.class_embed, std=.02)

        if task_switch['bbox']:
            self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        # Caption Project and query
        if task_switch['captioning']:
            self.caping_embed = nn.Parameter(torch.empty(hidden_dim, dim_proj))
            trunc_normal_(self.caping_embed, std=.02)
            self.pos_embed_caping = nn.Embedding(contxt_len, hidden_dim)
            self.captioning_step = captioning_step

        # register self_attn_mask to avoid information leakage, it includes interaction between object query, class query and caping query
        self_attn_mask = torch.zeros((1, num_queries + contxt_len, num_queries + contxt_len)).bool()
        self_attn_mask[:, :num_queries, num_queries:] = True # object+class query does not attend with caption query.
        self_attn_mask[:, num_queries:, num_queries:] = torch.triu(torch.ones((1, contxt_len, contxt_len)), diagonal=1).bool() # caption query only attend with previous token.
        self_attn_mask[:, :num_queries-1, num_queries-1:num_queries] = True # object query does not attend with class query.
        self_attn_mask[:, num_queries-1:num_queries, :num_queries-1] = True # class query does not attend with object query.
        self.register_buffer("self_attn_mask", self_attn_mask)


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
        ret["captioning_step"] = dec_cfg['CAPTIONING'].get('STEP', 50)

        return ret

    def forward(self, x, mask_features, mask=None, target_queries=None, target_vlp=None, task='seg', extra={}):
        if task == 'captioning_infer':
            return self.forward_captioning(x, mask_features, mask=mask, target_queries=target_queries, target_vlp=target_vlp, task=task, extra=extra)
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []
        
        # disable mask, it does not affect performance
        del mask
        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        predictions_class = []
        predictions_mask = []
        predictions_bbox = []
        predictions_caption = []
        predictions_captioning = []
        
        self_tgt_mask = None
        if self.training and task == 'vlp' and self.task_switch['captioning']:
            # output = torch.cat((output, self.query_feat_caping.weight.unsqueeze(1).repeat(1, bs, 1)), dim=0) # concat object query, class token and caption token.
            caping_lang_embed = torch.cat([caption['caption_tokens'] for caption in target_vlp], dim=0).transpose(0, 1) # language output
            _caping_lang_embed = caping_lang_embed.detach().clone()
            output = torch.cat((output, _caping_lang_embed), dim=0) # concat object query, class token and caption token.
            caping_lang_embed += self.pos_embed_caping.weight.unsqueeze(1).repeat(1, bs, 1)
            query_embed = torch.cat((query_embed, caping_lang_embed), dim=0) # may not add at the beginning.
            self_tgt_mask = self.self_attn_mask.repeat(output.shape[1]*self.num_heads, 1, 1)
        elif (((self.training and task == 'seg') or (task == 'grounding_eval')) and self.task_switch['grounding']):
            self_tgt_mask = self.self_attn_mask[:,:self.num_queries,:self.num_queries].repeat(output.shape[1]*self.num_heads, 1, 1)
            grounding_tokens = extra['grounding_tokens']
            _grounding_tokens = grounding_tokens.detach().clone()
            # initialize with negative attention at the beginning.
            pad_tgt_mask = torch.ones((1, self.num_queries + (self.num_queries-1) + len(grounding_tokens), self.num_queries + (self.num_queries-1) + len(grounding_tokens)), device=self_tgt_mask.device).bool().repeat(output.shape[1]*self.num_heads, 1, 1)
            pad_tgt_mask[:,:self.num_queries,:self.num_queries] = self_tgt_mask
            pad_tgt_mask[:,self.num_queries:,self.num_queries:] = False # grounding tokens could attend with eatch other
            self_tgt_mask = pad_tgt_mask
            output = torch.cat((output, output[:-1]), dim=0)
            query_embed = torch.cat((query_embed, query_embed[:-1]), dim=0) # also pad language embdding to fix embedding
        else:
            self_tgt_mask = self.self_attn_mask[:,:self.num_queries,:self.num_queries].repeat(output.shape[1]*self.num_heads, 1, 1)

        # prediction heads on learnable query features
        results = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[0], task=task)
        attn_mask = results["attn_mask"]
        predictions_class.append(results["outputs_class"])
        predictions_mask.append(results["outputs_mask"])
        predictions_bbox.append(results["outputs_bbox"])
        predictions_caption.append(results["outputs_caption"])
        predictions_captioning.append(results["outputs_captionting"])
        
        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            if self.training and task == 'vlp' and self.task_switch['captioning']:
                attn_mask = torch.cat((attn_mask, torch.zeros_like(attn_mask[:, :self.contxt_len, :])), dim=1)
            # attention: cross-attention first
            output, avg_attn = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], query_pos=query_embed
            )

            if (((self.training and task == 'seg') or (task == 'grounding_eval')) and self.task_switch['grounding']):
                output = torch.cat((output, _grounding_tokens), dim=0)
                query_embed = torch.cat((query_embed, grounding_tokens), dim=0)

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=self_tgt_mask,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )
            
            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )

            if ((self.training and task == 'seg') or (task == 'grounding_eval')) and self.task_switch['grounding']:
                _grounding_tokens = output[-len(_grounding_tokens):]
                output = output[:-len(_grounding_tokens)]
                query_embed = query_embed[:-len(_grounding_tokens)]

            results = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels], layer_id=i, task=task)
            attn_mask = results["attn_mask"]
            predictions_class.append(results["outputs_class"])
            predictions_mask.append(results["outputs_mask"])
            predictions_bbox.append(results["outputs_bbox"])
            predictions_caption.append(results["outputs_caption"])
            predictions_captioning.append(results["outputs_captionting"])

        assert len(predictions_class) == self.num_layers + 1
        if task == 'vlp':
            out = {'pred_captionings': predictions_captioning[-1], 
                   'pred_captions': predictions_caption[-1], 
                   'aux_outputs': [{'pred_captionings': x, 'pred_captions': y } for x, y in zip(predictions_captioning[:-1], predictions_caption[:-1])]}
            return out
        else:
            out = {
                'pred_logits': predictions_class[-1],
                'pred_masks': predictions_mask[-1],
                'pred_boxes': predictions_bbox[-1],
                'pred_captions': predictions_caption[-1],
                'aux_outputs': self._set_aux_loss(
                    predictions_class if self.mask_classification else None, predictions_mask, predictions_bbox, predictions_caption
                )
            }
            return out

    def forward_captioning(self, x, mask_features, mask = None, target_queries = None, target_vlp = None, task='seg', extra={}):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []
        
        # disable mask, it does not affect performance
        del mask
        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        # QxNxC
        query_embed_ = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        query_feat = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)        
        caping_lang_token = extra['start_token'].repeat(bs, 1)
        pos_embed_caping = self.pos_embed_caping.weight.unsqueeze(1).repeat(1, bs, 1)

        # prepare token embedding for evaluation
        token_embs = self.lang_encoder.lang_encoder.token_embedding.weight
        # token_embs = (token_embs / token_embs.norm(dim=-1, keepdim=True) + 1e-7)
        
        for cap_idx in range(0, self.captioning_step):
            caping_lang_embed = self.lang_encoder.forward_language_token((caping_lang_token,))[0].transpose(0, 1)
            output = torch.cat((query_feat, caping_lang_embed), dim=0) # concat object query, class token and caption token.
            caping_lang_embed += pos_embed_caping
            query_embed = torch.cat((query_embed_, caping_lang_embed), dim=0) # may not add at the beginning.
            # output = torch.cat((query_feat, query_feat_caping), dim=0) # concat object query, class token and caption token.

            # prediction heads on learnable query features
            results = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[0], task=task)
            attn_mask = results["attn_mask"]
        
            for i in range(self.num_layers):
                level_index = i % self.num_feature_levels
                attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
                attn_mask = torch.cat((attn_mask, torch.zeros_like(attn_mask[:, :self.contxt_len, :])), dim=1)
                self_tgt_mask = self.self_attn_mask.repeat(output.shape[1]*self.num_heads, 1, 1)

                if extra['captioning_mask'] is not None:
                    bs,nq,wh = attn_mask.shape
                    assert bs==self.num_heads, "Only support single image referring captioning."
                    cap_mask = extra['captioning_mask']
                    attn_mask = attn_mask.reshape(bs,nq,size_list[i%3][0],size_list[i%3][1])
                    cap_mask = F.interpolate(cap_mask[None,].float(), size_list[i%3], mode='nearest').bool()[0,0]
                    attn_mask[:,self.num_queries:, cap_mask] = True
                    attn_mask = attn_mask.reshape(bs,nq,wh)
                
                # attention: cross-attention first
                output, avg_attn = self.transformer_cross_attention_layers[i](
                    output, src[level_index],
                    memory_mask=attn_mask,
                    memory_key_padding_mask=None,  # here we do not apply masking on padded region
                    pos=pos[level_index], query_pos=query_embed
                )

                output = self.transformer_self_attention_layers[i](
                    output, tgt_mask=self_tgt_mask,
                    tgt_key_padding_mask=None,
                    query_pos=query_embed
                )
                
                # FFN
                output = self.transformer_ffn_layers[i](
                    output
                )

                results = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels], layer_id=i, task=task)
                attn_mask = results["attn_mask"]
            
            pred_captions_gen = results['outputs_captionting']
            # pred_captions_gen = (pred_captions_gen / pred_captions_gen.norm(dim=-1, keepdim=True) + 1e-7)
            pred_captions_gen = pred_captions_gen @ token_embs.t()
            caping_lang_token[:,cap_idx+1] = pred_captions_gen[:,cap_idx].max(-1)[1]

        texts = self.lang_encoder.tokenizer.batch_decode(caping_lang_token, skip_special_tokens=False)
        texts_new = []
        
        for x in texts:
            x = x.split('<|endoftext|>')[0]
            x = x.replace('<|endoftext|>','')
            x = x.replace('<|startoftext|>','')
            x = x.strip()
            texts_new.append(x)

        out = {'pred_captionings': caping_lang_token,
               'pred_texts': texts_new}
        return out


    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size, layer_id=-1, task='seg'):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)

        # extract image captioning token from decoder output.
        if self.task_switch['captioning'] and (task == 'vlp' or task == 'captioning_infer'):
            outputs_captionting = decoder_output[:,self.num_queries:] @ self.caping_embed
        else:
            outputs_captionting = None

        # recompute class token output.
        norm_decoder_output = decoder_output / (decoder_output.norm(dim=-1, keepdim=True) + 1e-7)
        obj_token = norm_decoder_output[:,:self.num_queries-1]
        cls_token = norm_decoder_output[:,self.num_queries-1:self.num_queries]

        sim = (cls_token @ obj_token.transpose(1,2)).softmax(-1)[:,0,:,None] # TODO include class token.
        cls_token = (sim * decoder_output[:,:self.num_queries-1]).sum(dim=1, keepdim=True)

        if (((self.training and task == 'seg') or (task == 'grounding_eval')) and self.task_switch['grounding']):
            decoder_output = torch.cat((decoder_output[:,:self.num_queries-1], cls_token, decoder_output[:,self.num_queries:2*self.num_queries-1]), dim=1)
        else:
            decoder_output = torch.cat((decoder_output[:,:self.num_queries-1], cls_token), dim=1)

        # compute class, mask and bbox.
        class_embed = decoder_output @ self.class_embed
        # HACK do not compute similarity if mask is not on
        outputs_class = self.lang_encoder.compute_similarity(class_embed, fake=(((not self.task_switch['mask']) and self.training)))

        if self.task_switch['mask']:
            mask_embed = self.mask_embed(decoder_output)
            outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

            # NOTE: prediction is of higher-resolution
            # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
            attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bicubic", align_corners=False, antialias=True)

            # must use bool type
            # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
            attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
            attn_mask = attn_mask.detach()

            # NOTE: fill False for cls token (JY)
            attn_mask[:, self.num_queries:self.num_queries+1].fill_(False)
        else:
            outputs_mask = None
            attn_mask = torch.zeros((list(decoder_output.shape[:2]) + [attn_mask_target_size[0]*attn_mask_target_size[1]]), device=decoder_output.device).repeat(self.num_heads, 1, 1).bool()

        outputs_bbox = [None for i in range(len(decoder_output))]
        if self.task_switch['bbox']:
            outputs_bbox = self.bbox_embed(decoder_output)

        outputs_caption = None
        if self.task_switch['caption']:
            outputs_caption = class_embed
            

        results = {
            "outputs_class": outputs_class,
            "outputs_mask": outputs_mask,
            "outputs_bbox": outputs_bbox,
            "attn_mask": attn_mask,
            "outputs_caption": outputs_caption,
            "outputs_captionting": outputs_captionting,
        }
        return results

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks, outputs_boxes, outputs_captions):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b, "pred_boxes": c, "pred_captions": d}
                for a, b, c, d in zip(outputs_class[:-1], outputs_seg_masks[:-1], outputs_boxes[:-1], outputs_captions[:-1])
            ]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]


@register_decoder
def get_xdecoder_interface(cfg, in_channels, lang_encoder, mask_classification, extra):
    return XDecoder(cfg, in_channels, lang_encoder, mask_classification, extra)