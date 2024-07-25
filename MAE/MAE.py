from math import log2
from typing import Callable, Tuple

import torch as th
import torch.nn.functional as F
import torch.optim as optim
from einops import repeat
from timm.layers import PatchEmbed
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block
from torch import nn

from .utils import get_2d_sincos_pos_embed

# 以下は https://github.com/younggyoseo/MWM の Masked Auto Encoder部分をPyTorchに変換したもの


class Base(nn.Module):
    patch_size: int

    def __init__(self):
        super().__init__()

    def patchify(self, imgs: th.Tensor) -> th.Tensor:
        """
        imgs: [N, C, H, W]
        x: [N, L, patch_size**2 * C]
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        unfolder = nn.Unfold(kernel_size=p, stride=p, padding=0)
        x = unfolder.forward(imgs).permute([0, 2, 1])

        return x

    def unpatchify(self, x: th.Tensor) -> th.Tensor:
        """
        x: [N, L, patch_size**2 * C]
        imgs: [N, C, H, W]
        """
        x = x.permute([0, 2, 1])
        p = self.patch_size
        c = x.shape[1] // (p**2)
        h = w = int(x.shape[2] ** 0.5)
        assert h * w == x.shape[2]

        x = x.reshape([-1, c, p, p, h, w])
        x = th.einsum("ncpqhw->nchpwq", x)
        imgs = x.reshape([-1, c, h * p, w * p])
        return imgs


class Encoder(Base):
    def __init__(
        self,
        img_size: int,
        img_channel: int,
        patch_size: int,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        early_conv: bool = False,
    ):
        super().__init__()

        assert int(log2(patch_size)) == log2((patch_size))

        self.patch_size = patch_size

        self.pos_embed = nn.Parameter(
            th.tensor(
                get_2d_sincos_pos_embed(embed_dim, int(img_size // patch_size), cls_token=True, add_token=False)[None],
                dtype=th.float,
            ),
            requires_grad=False,
        )

        modules = []
        if early_conv:
            in_channel = img_channel
            n = int(log2(patch_size))
            for i in range(n):
                d = embed_dim // (2 ** (n - i))
                modules.append(nn.Conv2d(in_channel, d, kernel_size=4, stride=2, padding=1))
                modules.append(nn.ReLU())
                in_channel = d
            modules.append(nn.Conv2d(in_channel, embed_dim, kernel_size=1, stride=1))
            modules.append(nn.Flatten(start_dim=2))
        else:
            modules.append(
                PatchEmbed(
                    img_size=img_size,
                    patch_size=patch_size,
                    in_chans=img_channel,
                    embed_dim=embed_dim,
                    output_fmt="NCL",
                )
            )
        self.conv_embed = nn.Sequential(*modules)

        self.cls_token = nn.Parameter(th.zeros([1, 1, embed_dim]), requires_grad=True)
        trunc_normal_(self.cls_token, std=0.02)

        self.vit = nn.Sequential(
            *[
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    proj_drop=0.0,
                    attn_drop=0.0,
                    drop_path=0.0,
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)

    def random_tube_masking(self, x: th.Tensor, mask_ratio: float, t: int):
        device = x.device

        N, L, _ = x.shape
        len_keep = int(L * (1 - mask_ratio))
        B = N // t
        assert B * t == N

        noise = th.rand([B, L], device=device)
        if mask_ratio == 0.0:
            noise, _ = th.sort(noise)
        noise = th.repeat_interleave(noise, t, dim=0)
        _, ids_shuffle = th.sort(noise, dim=1)
        _, ids_restore = th.sort(ids_shuffle, dim=1)

        ids_keep = repeat(ids_shuffle, "n l -> n l c", c=x.shape[-1])[:, :len_keep, :]
        x_masked = th.gather(x, 1, ids_keep)

        mask = th.concat([th.zeros([N, len_keep], device=device), th.ones([N, L - len_keep], device=device)], dim=1)
        mask = th.gather(mask, 1, ids_restore)

        return x_masked, mask, ids_restore

    def forward(self, x: th.Tensor, mask_ratio: float, t: int):
        batch_size = x.shape[0]

        x = self.conv_embed(x)
        x = x.permute([0, 2, 1]) + self.pos_embed[:, 1:, :]
        x, mask, ids_restore = self.random_tube_masking(x, mask_ratio, t)

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_token = th.repeat_interleave(cls_token, batch_size, dim=0)
        x = th.concat([cls_token, x], dim=1)

        x = self.norm(self.vit(x))

        return x, mask, ids_restore


class Decoder(Base):
    def __init__(
        self,
        img_size: int,
        img_channel: int,
        patch_size: int,
        embed_dim: int = 512,
        depth: int = 8,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()

        assert int(log2(patch_size)) == log2((patch_size))

        self.patch_size = patch_size

        self.pos_embed = nn.Parameter(
            th.tensor(
                get_2d_sincos_pos_embed(embed_dim, int(img_size // patch_size), cls_token=True, add_token=False)[None],
                dtype=th.float,
            ),
            requires_grad=False,
        )

        self.embed = nn.LazyLinear(embed_dim)

        self.mask_token = nn.Parameter(th.zeros([1, 1, embed_dim]), requires_grad=True)
        trunc_normal_(self.mask_token, std=0.02)

        self.vit = nn.Sequential(
            *[
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    proj_drop=0.0,
                    attn_drop=0.0,
                    drop_path=0.0,
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)

        self.vit_pred = nn.LazyLinear(patch_size**2 * img_channel)

    def forward(self, x: th.Tensor, ids_restore: th.Tensor):
        x = self.embed(x)

        ids_restore = repeat(ids_restore, "n l -> n l c", c=x.shape[-1])
        mask_tokens = th.tile(self.mask_token, dims=[x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1])
        x_ = th.concat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = th.gather(x_, 1, ids_restore)
        x = th.concat([x[:, :1, :], x_], dim=1)

        x += self.pos_embed

        x = self.norm(self.vit(x))

        x = self.vit_pred(x[:, 1:, :])

        return x


class MAE(nn.Module):
    def __init__(
        self,
        img_size: int,
        img_channel: int,
        patch_size: int,
        mask_ratio: float,
        lr: float = 1e-4,
        norm_pix_loss: bool = False,
        masked_loss: bool = False,
        enc_kwargs: dict = {},
        dec_kwargs: dict = {},
        hidden_activation: Callable[[th.Tensor], th.Tensor] = F.tanh,
    ):
        super().__init__()

        self.encoder = Encoder(img_size=img_size, img_channel=img_channel, patch_size=patch_size, **enc_kwargs)
        self.decoder = Decoder(img_size=img_size, img_channel=img_channel, patch_size=patch_size, **dec_kwargs)

        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss
        self.masked_loss = masked_loss

        self.hidden_activation = hidden_activation
        self.optim = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x: th.Tensor, return_pred: bool = False) -> th.Tensor | Tuple[th.Tensor, th.Tensor]:
        z, _, ids_restore = self.encoder.forward(x, self.mask_ratio, 1)
        if return_pred:
            y = self.decoder.forward(z, ids_restore)
            return z, y
        else:
            return self.hidden_activation(z)

    def loss(self, x: th.Tensor) -> th.Tensor:
        z, mask, ids_restore = self.encoder.forward(x, self.mask_ratio, 1)
        y = self.decoder.forward(z, ids_restore)  # [N, L, p^2*C]

        target = self.encoder.patchify(x)

        if self.norm_pix_loss:
            var, mean = th.var_mean(target, dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6) ** 0.5

        loss = (y - target) ** 2
        loss = th.mean(loss, dim=-1)  # [N, L], mean loss per patch

        if self.masked_loss:
            loss = (loss * mask).sum() / mask.sum()
        else:
            loss = loss.mean()

        return loss
