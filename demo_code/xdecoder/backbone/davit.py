import os
import itertools
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from collections import OrderedDict

from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_

from detectron2.utils.file_io import PathManager
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec

from .registry import register_backbone

logger = logging.getLogger(__name__)


class MySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class PreNorm(nn.Module):
    def __init__(self, norm, fn, drop_path=None):
        super().__init__()
        self.norm = norm
        self.fn = fn
        self.drop_path = drop_path

    def forward(self, x, *args, **kwargs):
        shortcut = x
        if self.norm != None:
            x, size = self.fn(self.norm(x), *args, **kwargs)
        else:
            x, size = self.fn(x, *args, **kwargs)

        if self.drop_path:
            x = self.drop_path(x)

        x = shortcut + x

        return x, size


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.net = nn.Sequential(OrderedDict([
            ("fc1", nn.Linear(in_features, hidden_features)),
            ("act", act_layer()),
            ("fc2", nn.Linear(hidden_features, out_features))
        ]))

    def forward(self, x, size):
        return self.net(x), size


class DepthWiseConv2d(nn.Module):
    def __init__(
        self,
        dim_in,
        kernel_size,
        padding,
        stride,
        bias=True,
    ):
        super().__init__()
        self.dw = nn.Conv2d(
            dim_in, dim_in,
            kernel_size=kernel_size,
            padding=padding,
            groups=dim_in,
            stride=stride,
            bias=bias
        )

    def forward(self, x, size):
        B, N, C = x.shape
        H, W = size
        assert N == H * W

        x = self.dw(x.transpose(1, 2).view(B, C, H, W))
        size = (x.size(-2), x.size(-1))
        x = x.flatten(2).transpose(1, 2)
        return x, size


class ConvEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(
        self,
        patch_size=7,
        in_chans=3,
        embed_dim=64,
        stride=4,
        padding=2,
        norm_layer=None,
        pre_norm=True
    ):
        super().__init__()
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=padding
        )

        dim_norm = in_chans if pre_norm else embed_dim
        self.norm = norm_layer(dim_norm) if norm_layer else None

        self.pre_norm = pre_norm

    def forward(self, x, size):
        H, W = size
        if len(x.size()) == 3:
            if self.norm and self.pre_norm:
                x = self.norm(x)
            x = rearrange(
                x, 'b (h w) c -> b c h w',
                h=H, w=W
            )

        x = self.proj(x)

        _, _, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        if self.norm and not self.pre_norm:
            x = self.norm(x)

        return x, (H, W)


class ChannelAttention(nn.Module):

    def __init__(self, dim, groups=8, qkv_bias=True):
        super().__init__()

        self.groups = groups
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, size):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.groups, C // self.groups).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * (N ** -0.5)
        attention = q.transpose(-1, -2) @ k
        attention = attention.softmax(dim=-1)
        x = (attention @ v.transpose(-1, -2)).transpose(-1, -2)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x, size


class ChannelBlock(nn.Module):

    def __init__(self, dim, groups, mlp_ratio=4., qkv_bias=True,
                 drop_path_rate=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 conv_at_attn=True, conv_at_ffn=True):
        super().__init__()

        drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        self.conv1 = PreNorm(None, DepthWiseConv2d(dim, 3, 1, 1)) if conv_at_attn else None
        self.channel_attn = PreNorm(
            norm_layer(dim),
            ChannelAttention(dim, groups=groups, qkv_bias=qkv_bias),
            drop_path
        )
        self.conv2 = PreNorm(None, DepthWiseConv2d(dim, 3, 1, 1)) if conv_at_ffn else None
        self.ffn = PreNorm(
            norm_layer(dim),
            Mlp(in_features=dim, hidden_features=int(dim*mlp_ratio), act_layer=act_layer),
            drop_path
        )

    def forward(self, x, size):
        if self.conv1:
            x, size = self.conv1(x, size)
        x, size = self.channel_attn(x, size)

        if self.conv2:
            x, size = self.conv2(x, size)
        x, size = self.ffn(x, size)

        return x, size


def window_partition(x, window_size: int):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size, qkv_bias=True):

        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, size):

        H, W = size
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        x = window_partition(x, self.window_size)
        x = x.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        # attn_windows = self.attn(x_windows)

        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)

        # merge windows
        x = x.view(
            -1, self.window_size, self.window_size, C
        )
        x = window_reverse(x, self.window_size, Hp, Wp)

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        return x, size


class SpatialBlock(nn.Module):

    def __init__(self, dim, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop_path_rate=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, conv_at_attn=True, conv_at_ffn=True):
        super().__init__()

        drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        self.conv1 = PreNorm(None, DepthWiseConv2d(dim, 3, 1, 1)) if conv_at_attn else None
        self.window_attn = PreNorm(
            norm_layer(dim),
            WindowAttention(dim, num_heads, window_size, qkv_bias=qkv_bias),
            drop_path
        )
        self.conv2 = PreNorm(None, DepthWiseConv2d(dim, 3, 1, 1)) if conv_at_ffn else None
        self.ffn = PreNorm(
            norm_layer(dim),
            Mlp(in_features=dim, hidden_features=int(dim*mlp_ratio), act_layer=act_layer),
            drop_path
        )

    def forward(self, x, size):
        if self.conv1:
            x, size = self.conv1(x, size)
        x, size = self.window_attn(x, size)

        if self.conv2:
            x, size = self.conv2(x, size)
        x, size = self.ffn(x, size)
        return x, size


class DaViT(nn.Module):
    """ DaViT: Dual-Attention Transformer

    Args:
        img_size (int): Image size, Default: 224.
        in_chans (int): Number of input image channels. Default: 3.
        num_classes (int): Number of classes for classification head. Default: 1000.
        patch_size (tuple(int)): Patch size of convolution in different stages. Default: (7, 2, 2, 2).
        patch_stride (tuple(int)): Patch stride of convolution in different stages. Default: (4, 2, 2, 2).
        patch_padding (tuple(int)): Patch padding of convolution in different stages. Default: (3, 0, 0, 0).
        patch_prenorm (tuple(bool)): If True, perform norm before convlution layer. Default: (True, False, False, False).
        embed_dims (tuple(int)): Patch embedding dimension in different stages. Default: (64, 128, 192, 256).
        num_heads (tuple(int)): Number of spatial attention heads in different stages. Default: (4, 8, 12, 16).
        num_groups (tuple(int)): Number of channel groups in different stages. Default: (4, 8, 12, 16).
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True.
        drop_path_rate (float): Stochastic depth rate. Default: 0.1.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        enable_checkpoint (bool): If True, enable checkpointing. Default: False.
        conv_at_attn (bool): If True, performe depthwise convolution before attention layer. Default: True.
        conv_at_ffn (bool): If True, performe depthwise convolution before ffn layer. Default: True.
    """

    def __init__(
        self,
        img_size=224,
        in_chans=3,
        num_classes=1000,
        depths=(1, 1, 3, 1),
        patch_size=(7, 2, 2, 2),
        patch_stride=(4, 2, 2, 2),
        patch_padding=(3, 0, 0, 0),
        patch_prenorm=(False, False, False, False),
        embed_dims=(64, 128, 192, 256),
        num_heads=(3, 6, 12, 24),
        num_groups=(3, 6, 12, 24),
        window_size=7,
        mlp_ratio=4.,
        qkv_bias=True,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        enable_checkpoint=False,
        conv_at_attn=True,
        conv_at_ffn=True,
        out_indices=[],
     ):
        super().__init__()

        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.num_stages = len(self.embed_dims)
        self.enable_checkpoint = enable_checkpoint
        assert self.num_stages == len(self.num_heads) == len(self.num_groups)

        num_stages = len(embed_dims)
        self.img_size = img_size
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths)*2)]


        depth_offset = 0
        convs = []
        blocks = []
        for i in range(num_stages):
            conv_embed = ConvEmbed(
                patch_size=patch_size[i],
                stride=patch_stride[i],
                padding=patch_padding[i],
                in_chans=in_chans if i == 0 else self.embed_dims[i - 1],
                embed_dim=self.embed_dims[i],
                norm_layer=norm_layer,
                pre_norm=patch_prenorm[i]
            )
            convs.append(conv_embed)

            print(f'=> Depth offset in stage {i}: {depth_offset}')
            block = MySequential(
                *[
                    MySequential(OrderedDict([
                        (
                            'spatial_block', SpatialBlock(
                                embed_dims[i],
                                num_heads[i],
                                window_size,
                                drop_path_rate=dpr[depth_offset+j*2],
                                qkv_bias=qkv_bias,
                                mlp_ratio=mlp_ratio,
                                conv_at_attn=conv_at_attn,
                                conv_at_ffn=conv_at_ffn,
                            )
                        ),
                        (
                            'channel_block', ChannelBlock(
                                embed_dims[i],
                                num_groups[i],
                                drop_path_rate=dpr[depth_offset+j*2+1],
                                qkv_bias=qkv_bias,
                                mlp_ratio=mlp_ratio,
                                conv_at_attn=conv_at_attn,
                                conv_at_ffn=conv_at_ffn,
                            )
                        )
                    ])) for j in range(depths[i])
                ]
            )
            blocks.append(block)
            depth_offset += depths[i]*2

        self.convs = nn.ModuleList(convs)
        self.blocks = nn.ModuleList(blocks)

        self.out_indices = out_indices
        # self.norms = norm_layer(self.embed_dims[-1])
        # self.avgpool = nn.AdaptiveAvgPool1d(1)
        # self.head = nn.Linear(self.embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    @property
    def dim_out(self):
        return self.embed_dims[-1]

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.02)
            for name, _ in m.named_parameters():
                if name in ['bias']:
                    nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def _try_remap_keys(self, pretrained_dict):
        remap_keys = {
            "conv_embeds": "convs",
            "main_blocks": "blocks",
            "0.cpe.0.proj": "spatial_block.conv1.fn.dw",
            "0.attn": "spatial_block.window_attn.fn",
            "0.cpe.1.proj": "spatial_block.conv2.fn.dw",
            "0.mlp": "spatial_block.ffn.fn.net",
            "1.cpe.0.proj": "channel_block.conv1.fn.dw",
            "1.attn": "channel_block.channel_attn.fn",
            "1.cpe.1.proj": "channel_block.conv2.fn.dw",
            "1.mlp": "channel_block.ffn.fn.net",
            "0.norm1": "spatial_block.window_attn.norm",
            "0.norm2": "spatial_block.ffn.norm",
            "1.norm1": "channel_block.channel_attn.norm",
            "1.norm2": "channel_block.ffn.norm"
        }

        full_key_mappings = {}
        for k in pretrained_dict.keys():
            old_k = k
            for remap_key in remap_keys.keys():
                if remap_key in k:
                    print(f'=> Repace {remap_key} with {remap_keys[remap_key]}')
                    k = k.replace(remap_key, remap_keys[remap_key])

            full_key_mappings[old_k] = k

        return full_key_mappings

    def from_state_dict(self, pretrained_dict, pretrained_layers=[], verbose=True):
        model_dict = self.state_dict()
        stripped_key = lambda x: x[14:] if x.startswith('image_encoder.') else x
        full_key_mappings = self._try_remap_keys(pretrained_dict)

        pretrained_dict = {
            stripped_key(full_key_mappings[k]): v for k, v in pretrained_dict.items()
            if stripped_key(full_key_mappings[k]) in model_dict.keys()
        }
        need_init_state_dict = {}
        for k, v in pretrained_dict.items():
            need_init = (
                k.split('.')[0] in pretrained_layers
                or pretrained_layers[0] == '*'
            )
            if need_init:
                if verbose:
                    print(f'=> init {k} from pretrained state dict')

                need_init_state_dict[k] = v
        self.load_state_dict(need_init_state_dict, strict=False)

    def from_pretrained(self, pretrained='', pretrained_layers=[], verbose=True):
        if os.path.isfile(pretrained):
            print(f'=> loading pretrained model {pretrained}')
            pretrained_dict = torch.load(pretrained, map_location='cpu')

            self.from_state_dict(pretrained_dict, pretrained_layers, verbose)

    def forward_features(self, x):
        input_size = (x.size(2), x.size(3))

        outs = {}
        for i, (conv, block) in enumerate(zip(self.convs, self.blocks)):
            x, input_size = conv(x, input_size)
            if self.enable_checkpoint:
                x, input_size = checkpoint.checkpoint(block, x, input_size)
            else:
                x, input_size = block(x, input_size)
            if i in self.out_indices:
                out = x.view(-1, *input_size, self.embed_dims[i]).permute(0, 3, 1, 2).contiguous()
                outs["res{}".format(i + 2)] = out       

        if len(self.out_indices) == 0:
            outs["res5"] = x.view(-1, *input_size, self.embed_dims[-1]).permute(0, 3, 1, 2).contiguous()
        
        return outs

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)
        return x

class D2DaViT(DaViT, Backbone):
    def __init__(self, cfg, input_shape):

        spec = cfg['BACKBONE']['DAVIT']

        super().__init__(
            num_classes=0,
            depths=spec['DEPTHS'],
            embed_dims=spec['DIM_EMBED'],
            num_heads=spec['NUM_HEADS'],
            num_groups=spec['NUM_GROUPS'],
            patch_size=spec['PATCH_SIZE'],
            patch_stride=spec['PATCH_STRIDE'],
            patch_padding=spec['PATCH_PADDING'],
            patch_prenorm=spec['PATCH_PRENORM'],
            drop_path_rate=spec['DROP_PATH_RATE'],
            img_size=input_shape,
            window_size=spec.get('WINDOW_SIZE', 7),
            enable_checkpoint=spec.get('ENABLE_CHECKPOINT', False),
            conv_at_attn=spec.get('CONV_AT_ATTN', True),
            conv_at_ffn=spec.get('CONV_AT_FFN', True),
            out_indices=spec.get('OUT_INDICES', []),
        )

        self._out_features = cfg['BACKBONE']['DAVIT']['OUT_FEATURES']

        self._out_feature_strides = {
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
        }
        self._out_feature_channels = {
            "res2": self.embed_dims[0],
            "res3": self.embed_dims[1],
            "res4": self.embed_dims[2],
            "res5": self.embed_dims[3],
        }

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.
        Returns:
            dict[str->Tensor]: names and the corresponding features
        """
        assert (
            x.dim() == 4
        ), f"SwinTransformer takes an input of shape (N, C, H, W). Got {x.shape} instead!"
        outputs = {}
        y = super().forward(x)

        for k in y.keys():
            if k in self._out_features:
                outputs[k] = y[k]
        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    @property
    def size_divisibility(self):
        return 32

@register_backbone
def get_davit_backbone(cfg):
    davit = D2DaViT(cfg['MODEL'], 224)    

    if cfg['MODEL']['BACKBONE']['LOAD_PRETRAINED'] is True:
        filename = cfg['MODEL']['BACKBONE']['PRETRAINED']
        logger.info(f'=> init from {filename}')
        davit.from_pretrained(
            filename, 
            cfg['MODEL']['BACKBONE']['DAVIT'].get('PRETRAINED_LAYERS', ['*']), 
            cfg['VERBOSE'])

    return davit