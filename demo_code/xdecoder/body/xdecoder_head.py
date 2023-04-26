# Copyright (c) Facebook, Inc. and its affiliates.

# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Xueyan Zou (xueyan@cs.wisc.edu), Jianwei Yang (jianwyan@microsoft.com)
# --------------------------------------------------------

from typing import Dict

from torch import nn

from detectron2.layers import ShapeSpec

from .registry import register_body
from .encoder import build_encoder
from .decoder import build_decoder
from ..utils import configurable


class XDecoderHead(nn.Module):

    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        num_classes: int,
        pixel_decoder: nn.Module,
        loss_weight: float = 1.0,
        ignore_value: int = -1,
        # extra parameters
        transformer_predictor: nn.Module,
        transformer_in_feature: str,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            num_classes: number of classes to predict
            pixel_decoder: the pixel decoder module
            loss_weight: loss weight
            ignore_value: category id to be ignored during training.
            transformer_predictor: the transformer decoder that makes prediction
            transformer_in_feature: input feature name to the transformer_predictor
        """
        super().__init__()

        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]
        feature_strides = [v.stride for k, v in input_shape]
        feature_channels = [v.channels for k, v in input_shape]

        self.ignore_value = ignore_value
        self.common_stride = 4
        self.loss_weight = loss_weight

        self.pixel_decoder = pixel_decoder
        self.predictor = transformer_predictor
        self.transformer_in_feature = transformer_in_feature

        self.num_classes = num_classes

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec], lang_encoder: nn.Module, extra: dict):

        in_features_type = cfg['MODEL']['DECODER']['TRANSFORMER_IN_FEATURE']
        enc_cfg = cfg['MODEL']['ENCODER']
        dec_cfg = cfg['MODEL']['DECODER']

        # figure out in_channels to transformer predictor
        if in_features_type == "transformer_encoder":
            transformer_predictor_in_channels = enc_cfg['CONVS_DIM']
        elif in_features_type == "pixel_embedding":
            transformer_predictor_in_channels = enc_cfg['MASK_DIM']
        elif in_features_type == "multi_scale_pixel_decoder":  # for maskformer2
            transformer_predictor_in_channels = enc_cfg['CONVS_DIM']
        else:
            transformer_predictor_in_channels = input_shape[dec_cfg['TRANSFORMER_IN_FEATURE']].channels

        return {
            "input_shape": {
                k: v for k, v in input_shape.items() if k in enc_cfg['IN_FEATURES']
            },
            "ignore_value": enc_cfg['IGNORE_VALUE'],
            "num_classes": enc_cfg.get('NUM_CLASSES', None),
            "pixel_decoder": build_encoder(cfg, input_shape),
            "loss_weight": enc_cfg['LOSS_WEIGHT'],
            "transformer_in_feature": dec_cfg['TRANSFORMER_IN_FEATURE'],
            "transformer_predictor": build_decoder(
                cfg,
                transformer_predictor_in_channels,
                lang_encoder,
                mask_classification=True,
                extra=extra,
            ),
        }

    def forward(self, features, mask=None, target_queries=None, target_vlp=None, task='seg', extra={}):
        return self.layers(features, mask, target_queries, target_vlp, task, extra)

    def layers(self, features, mask=None, target_queries=None, target_vlp=None, task='seg', extra={}):
        mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(features)
        
        if self.transformer_in_feature == "multi_scale_pixel_decoder":
            predictions = self.predictor(multi_scale_features, mask_features, mask, target_queries, target_vlp, task, extra)
        else:
            if self.transformer_in_feature == "transformer_encoder":
                assert (
                    transformer_encoder_features is not None
                ), "Please use the TransformerEncoderPixelDecoder."
                predictions = self.predictor(transformer_encoder_features, mask_features, mask)
            elif self.transformer_in_feature == "pixel_embedding":
                predictions = self.predictor(mask_features, mask_features, mask)
            else:
                predictions = self.predictor(features[self.transformer_in_feature], mask_features, mask)
        return predictions


@register_body
def get_xdecoder_head(cfg, input_shape, lang_encoder, extra):
    return XDecoderHead(cfg, input_shape, lang_encoder, extra)