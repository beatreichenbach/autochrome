# Code from MST++: Multi-stage Spectral-wise Transformer for
# Efficient Spectral Reconstruction (CVPRW 2022)
# https://github.com/caiyuanhao1998/MST-plus-plus/blob/master/predict_code/architecture/MST_Plus_Plus.py

import logging
import math
from typing import Callable

import torch
from einops import rearrange
from torch import Tensor, nn

logger = logging.getLogger(__name__)


def _no_grad_trunc_normal_(
    tensor: Tensor, mean: float, std: float, a: float, b: float
) -> Tensor:
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        logger.warning(
            'Mean is more than 2 std from [a, b]. '
            'The distribution of values may be incorrect.'
        )
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(
    tensor: Tensor, mean: float = 0.0, std: float = 1.0, a: float = -2.0, b: float = 2.0
) -> Tensor:
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2
    else:
        raise ValueError(f'Unsupported mode: {mode}')
    variance = scale / denom
    if distribution == 'truncated_normal':
        trunc_normal_(tensor, std=math.sqrt(variance) / 0.87962566103423978)
    elif distribution == 'normal':
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == 'uniform':
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f'invalid distribution {distribution}')


def lecun_normal_(tensor: Tensor) -> None:
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')


class PreNorm(nn.Module):
    def __init__(self, dim, fn: Callable) -> None:
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return nn.functional.gelu(x)


def conv(
    in_channels: int,
    out_channels: int,
    kernel_size: int | tuple,
    bias: bool = False,
    padding: int | tuple | str | None = 1,
    stride: int | tuple | None = 1,
):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        padding=(kernel_size // 2),
        bias=bias,
        stride=stride,
    )


def shift_back(inputs, step=2):
    # input [bs,28,256,310]
    # output [bs, 28, 256, 256]

    [bs, nc, row, col] = inputs.shape
    down_sample = 256 // row
    step = float(step) / float(down_sample * down_sample)
    out_col = row
    for i in range(nc):
        inputs[:, i, :, :out_col] = inputs[
            :, i, :, int(step * i) : int(step * i) + out_col
        ]
    return inputs[:, :, :, :out_col]


class MS_MSA(nn.Module):
    def __init__(
        self,
        dim,
        dim_head,
        heads,
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, x_in):
        # x_in: [b,h,w,c]
        # out: [b,h,w,c]
        b, h, w, c = x_in.shape
        x = x_in.reshape(b, h * w, c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
            (q_inp, k_inp, v_inp),
        )
        v = v
        # q: b,heads,hw,c
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = nn.functional.normalize(q, dim=-1, p=2)
        k = nn.functional.normalize(k, dim=-1, p=2)
        attn = k @ q.transpose(-2, -1)  # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v  # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)  # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(v_inp.reshape(b, h, w, c).permute(0, 3, 1, 2)).permute(
            0, 2, 3, 1
        )
        out = out_c + out_p

        return out


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        # x: [b,h,w,c]
        # out: [b,h,w,c]
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)


class MSAB(nn.Module):
    def __init__(
        self,
        dim,
        dim_head,
        heads,
        num_blocks,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(
                nn.ModuleList(
                    [
                        MS_MSA(dim=dim, dim_head=dim_head, heads=heads),
                        PreNorm(dim, FeedForward(dim=dim)),
                    ]
                )
            )

    def forward(self, x):
        # x: [b,c,h,w]
        # out: [b,c,h,w]
        x = x.permute(0, 2, 3, 1)
        for attn, ff in self.blocks:
            x = attn(x) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)
        return out


class MST(nn.Module):
    def __init__(
        self,
        in_dim: int = 31,
        out_dim: int = 31,
        dim: int = 31,
        stage: int = 2,
        num_blocks: list | None = None,
    ) -> None:
        super(MST, self).__init__()
        self.dim = dim
        self.stage = stage

        # Input projection
        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        if num_blocks is None:
            num_blocks = [2, 4, 4]
        dim_stage = dim
        for i in range(stage):
            self.encoder_layers.append(
                nn.ModuleList(
                    [
                        MSAB(
                            dim=dim_stage,
                            num_blocks=num_blocks[i],
                            dim_head=dim,
                            heads=dim_stage // dim,
                        ),
                        nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False),
                    ]
                )
            )
            dim_stage *= 2

        # Bottleneck
        self.bottleneck = MSAB(
            dim=dim_stage,
            dim_head=dim,
            heads=dim_stage // dim,
            num_blocks=num_blocks[-1],
        )

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(stage):
            self.decoder_layers.append(
                nn.ModuleList(
                    [
                        nn.ConvTranspose2d(
                            dim_stage,
                            dim_stage // 2,
                            stride=2,
                            kernel_size=2,
                            padding=0,
                            output_padding=0,
                        ),
                        nn.Conv2d(dim_stage, dim_stage // 2, 1, 1, bias=False),
                        MSAB(
                            dim=dim_stage // 2,
                            num_blocks=num_blocks[stage - 1 - i],
                            dim_head=dim,
                            heads=(dim_stage // 2) // dim,
                        ),
                    ]
                )
            )
            dim_stage //= 2

        # Output projection
        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # x: [b,c,h,w]
        # out:[b,c,h,w]

        # Embedding
        fea = self.embedding(x)

        # Encoder
        fea_encoder = []
        for MSAB, FeaDownSample in self.encoder_layers:
            fea = MSAB(fea)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)

        # Bottleneck
        fea = self.bottleneck(fea)

        # Decoder
        for i, (FeaUpSample, Fution, LeWinBlock) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fution(torch.cat([fea, fea_encoder[self.stage - 1 - i]], dim=1))
            fea = LeWinBlock(fea)

        # Mapping
        out = self.mapping(fea) + x

        return out


class MST_Plus_Plus(nn.Module):
    def __init__(self, in_channels=3, out_channels=31, n_feat=31, stage=3):
        super(MST_Plus_Plus, self).__init__()
        self.stage = stage
        self.conv_in = nn.Conv2d(
            in_channels, n_feat, kernel_size=3, padding=(3 - 1) // 2, bias=False
        )
        modules_body = [
            MST(dim=31, stage=2, num_blocks=[1, 1, 1]) for _ in range(stage)
        ]
        self.body = nn.Sequential(*modules_body)
        self.conv_out = nn.Conv2d(
            n_feat, out_channels, kernel_size=3, padding=(3 - 1) // 2, bias=False
        )

    def forward(self, x):
        # x: [b,c,h,w]
        # out:[b,c,h,w]
        b, c, h_inp, w_inp = x.shape
        hb, wb = 8, 8
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        x = nn.functional.pad(x, [0, pad_w, 0, pad_h], mode='reflect')
        x = self.conv_in(x)
        h = self.body(x)
        h = self.conv_out(h)
        h += x
        return h[:, :, :h_inp, :w_inp]
