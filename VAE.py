from typing import Callable, Tuple, Union

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .utils import FE, SSIMLoss

# 以下はhttps://github.com/sksq96/pytorch-vaeを利用


class Flatten(nn.Module):
    def forward(self, input: th.Tensor):
        if input.ndim < 4:
            batch_size = 1
        else:
            batch_size = input.size(0)
        return input.view(batch_size, -1)


class UnFlatten(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.size = size

    def forward(self, input: th.Tensor):
        return input.view(-1, self.size, 1, 1)


class ConvVAE(FE):
    def __init__(
        self,
        img_height,
        img_width,
        img_channel,
        hidden_dim,
        lr=1e-3,
        hidden_activation: Callable[[th.Tensor], th.Tensor] = F.tanh,
    ) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(img_channel, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(),
            Flatten(),
        )

        with th.no_grad():
            self.eval()
            n_flatten = self.encoder(th.zeros((1, img_channel, img_height, img_width), dtype=th.float)).numel()
            self.train()

        self.enc_mean = nn.Linear(n_flatten, hidden_dim)
        self.enc_var = nn.Linear(n_flatten, hidden_dim)

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, n_flatten),
            UnFlatten(size=n_flatten),
            nn.ConvTranspose2d(n_flatten, 128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, img_channel, kernel_size=6, stride=2),
            nn.Sigmoid(),
        )

        self.hidden_activation = hidden_activation
        self.optim = optim.Adam(self.parameters(), lr=lr)

        self.re_loss = SSIMLoss(channel=img_channel)

    def _encode(self, input: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        tmp = self.encoder(input)
        mu = self.enc_mean(tmp)
        log_var = self.enc_var(tmp)
        return mu, log_var

    def _decode(self, z: th.Tensor) -> th.Tensor:
        return self.decoder(z)

    def _reparameterize(self, mu: th.Tensor, log_var: th.Tensor) -> th.Tensor:
        if self.training:
            std = log_var.mul(0.5).exp_()
            eps = th.randn_like(mu)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def _bottleneck(self, x: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        mu, log_var = self._encode(x)
        z = self._reparameterize(mu, log_var)
        return mu, log_var, z

    def forward(self, x: th.Tensor, return_pred: bool = False) -> Union[th.Tensor, Tuple[th.Tensor, th.Tensor]]:
        _, _, z = self._bottleneck(x)
        if return_pred:
            y = self._decode(z)
            return z, y
        else:
            return self.hidden_activation(z)

    def loss(self, x: th.Tensor) -> th.Tensor:
        mu, log_var, z = self._bottleneck(x)
        kl = -0.5 * th.sum(1 + log_var - mu.pow(2) - log_var.exp_())
        y = self._decode(z)
        re = self.re_loss(y, x) * x.numel()
        return kl + re


# 以下はhttps://github.com/LukeDitria/CNN-VAEを利用


def get_norm_layer(channels, norm_type="bn"):
    if norm_type == "bn":
        return nn.BatchNorm2d(channels, eps=1e-4)
    elif norm_type == "gn":
        return nn.GroupNorm(8, channels, eps=1e-4)
    else:
        ValueError("norm_type must be bn or gn")


class ResDown(nn.Module):
    """
    Residual down sampling block for the encoder
    """

    def __init__(self, channel_in, channel_out, kernel_size=3, norm_type="bn"):
        super(ResDown, self).__init__()
        self.norm1 = get_norm_layer(channel_in, norm_type=norm_type)

        self.conv1 = nn.Conv2d(channel_in, (channel_out // 2) + channel_out, kernel_size, 2, kernel_size // 2)
        self.norm2 = get_norm_layer(channel_out // 2, norm_type=norm_type)

        self.conv2 = nn.Conv2d(channel_out // 2, channel_out, kernel_size, 1, kernel_size // 2)

        self.act_fnc = nn.ELU()
        self.channel_out = channel_out

    def forward(self, x):
        x = self.act_fnc(self.norm1(x))

        # Combine skip and first conv into one layer for speed
        x_cat = self.conv1(x)
        skip = x_cat[:, : self.channel_out]
        x = x_cat[:, self.channel_out :]

        x = self.act_fnc(self.norm2(x))
        x = self.conv2(x)

        return x + skip


class ResUp(nn.Module):
    """
    Residual up sampling block for the decoder
    """

    def __init__(self, channel_in, channel_out, kernel_size=3, scale_factor=2, norm_type="bn"):
        super(ResUp, self).__init__()
        self.norm1 = get_norm_layer(channel_in, norm_type=norm_type)

        self.conv1 = nn.Conv2d(channel_in, (channel_in // 2) + channel_out, kernel_size, 1, kernel_size // 2)
        self.norm2 = get_norm_layer(channel_in // 2, norm_type=norm_type)

        self.conv2 = nn.Conv2d(channel_in // 2, channel_out, kernel_size, 1, kernel_size // 2)

        self.up_nn = nn.Upsample(scale_factor=scale_factor, mode="nearest")
        self.act_fnc = nn.ELU()
        self.channel_out = channel_out

    def forward(self, x_in):
        x = self.up_nn(self.act_fnc(self.norm1(x_in)))

        # Combine skip and first conv into one layer for speed
        x_cat = self.conv1(x)
        skip = x_cat[:, : self.channel_out]
        x = x_cat[:, self.channel_out :]

        x = self.act_fnc(self.norm2(x))
        x = self.conv2(x)

        return x + skip


class ResBlock(nn.Module):
    """
    Residual block
    """

    def __init__(self, channel_in, channel_out, kernel_size=3, norm_type="bn"):
        super(ResBlock, self).__init__()
        self.norm1 = get_norm_layer(channel_in, norm_type=norm_type)

        first_out = channel_in // 2 if channel_in == channel_out else (channel_in // 2) + channel_out
        self.conv1 = nn.Conv2d(channel_in, first_out, kernel_size, 1, kernel_size // 2)

        self.norm2 = get_norm_layer(channel_in // 2, norm_type=norm_type)

        self.conv2 = nn.Conv2d(channel_in // 2, channel_out, kernel_size, 1, kernel_size // 2)
        self.act_fnc = nn.ELU()
        self.skip = channel_in == channel_out
        self.bttl_nk = channel_in // 2

    def forward(self, x_in):
        x = self.act_fnc(self.norm1(x_in))

        x_cat = self.conv1(x)
        x = x_cat[:, : self.bttl_nk]

        # If channel_in == channel_out we do a simple identity skip
        if self.skip:
            skip = x_in
        else:
            skip = x_cat[:, self.bttl_nk :]

        x = self.act_fnc(self.norm2(x))
        x = self.conv2(x)

        return x + skip


class Encoder(nn.Module):
    """
    Encoder block
    """

    def __init__(
        self,
        channels,
        ch=64,
        blocks=(1, 2, 4, 8),
        latent_channels=256,
        num_res_blocks=1,
        norm_type="bn",
        deep_model=False,
    ):
        super(Encoder, self).__init__()
        self.conv_in = nn.Conv2d(channels, blocks[0] * ch, 3, 1, 1)

        widths_in = list(blocks)
        widths_out = list(blocks[1:]) + [2 * blocks[-1]]

        self.layer_blocks = nn.ModuleList([])
        for w_in, w_out in zip(widths_in, widths_out):

            if deep_model:
                # Add an additional non down-sampling block before down-sampling
                self.layer_blocks.append(ResBlock(w_in * ch, w_in * ch, norm_type=norm_type))

            self.layer_blocks.append(ResDown(w_in * ch, w_out * ch, norm_type=norm_type))

        for _ in range(num_res_blocks):
            self.layer_blocks.append(ResBlock(widths_out[-1] * ch, widths_out[-1] * ch, norm_type=norm_type))

        self.conv_mu = nn.Conv2d(widths_out[-1] * ch, latent_channels, 1, 1)
        self.conv_log_var = nn.Conv2d(widths_out[-1] * ch, latent_channels, 1, 1)
        self.act_fnc = nn.ELU()

    def sample(self, mu, log_var):
        std = th.exp(0.5 * log_var)
        eps = th.randn_like(std)
        return mu + eps * std

    def forward(self, x, sample=False):
        x = self.conv_in(x)

        for block in self.layer_blocks:
            x = block(x)
        x = self.act_fnc(x)

        mu = self.conv_mu(x)
        log_var = self.conv_log_var(x)

        if self.training or sample:
            x = self.sample(mu, log_var)
        else:
            x = mu

        return x, mu, log_var


class Decoder(nn.Module):
    """
    Decoder block
    Built to be a mirror of the encoder block
    """

    def __init__(
        self,
        channels,
        ch=64,
        blocks=(1, 2, 4, 8),
        latent_channels=256,
        num_res_blocks=1,
        norm_type="bn",
        deep_model=False,
    ):
        super(Decoder, self).__init__()
        widths_out = list(blocks)[::-1]
        widths_in = (list(blocks[1:]) + [2 * blocks[-1]])[::-1]

        self.conv_in = nn.Conv2d(latent_channels, widths_in[0] * ch, 1, 1)

        self.layer_blocks = nn.ModuleList([])
        for _ in range(num_res_blocks):
            self.layer_blocks.append(ResBlock(widths_in[0] * ch, widths_in[0] * ch, norm_type=norm_type))

        for w_in, w_out in zip(widths_in, widths_out):
            self.layer_blocks.append(ResUp(w_in * ch, w_out * ch, norm_type=norm_type))
            if deep_model:
                # Add an additional non up-sampling block after up-sampling
                self.layer_blocks.append(ResBlock(w_out * ch, w_out * ch, norm_type=norm_type))

        self.conv_out = nn.Conv2d(blocks[0] * ch, channels, 5, 1, 2)
        self.act_fnc = nn.ELU()

    def forward(self, x):
        x = self.conv_in(x)

        for block in self.layer_blocks:
            x = block(x)
        x = self.act_fnc(x)

        return th.tanh(self.conv_out(x))


class ResVAE(FE):
    """
    VAE network, uses the above encoder and decoder blocks
    """

    def __init__(
        self,
        img_channel=3,
        ch=64,
        blocks=(1, 2, 4, 8),
        latent_channels=256,
        num_res_blocks=1,
        norm_type="bn",
        deep_model=False,
        lr=1e-4,
        hidden_activation: Callable[[th.Tensor], th.Tensor] = F.tanh,
    ):
        super(ResVAE, self).__init__()
        """Res VAE Network
        channel_in  = number of channels of the image
        z = the number of channels of the latent representation
        (for a 64x64 image this is the size of the latent vector)
        """

        self.encoder = Encoder(
            img_channel,
            ch=ch,
            blocks=blocks,
            latent_channels=latent_channels,
            num_res_blocks=num_res_blocks,
            norm_type=norm_type,
            deep_model=deep_model,
        )
        self.decoder = Decoder(
            img_channel,
            ch=ch,
            blocks=blocks,
            latent_channels=latent_channels,
            num_res_blocks=num_res_blocks,
            norm_type=norm_type,
            deep_model=deep_model,
        )

        self.hidden_activation = hidden_activation
        self.optim = optim.Adam(self.parameters(), lr=lr)

        self.re_loss = SSIMLoss(channel=img_channel)

    def forward(self, x: th.Tensor, return_pred: bool = False) -> Union[th.Tensor, Tuple[th.Tensor, th.Tensor]]:
        z, _, _ = self.encoder(x)
        if return_pred:
            y = self.decoder(z)
            return z, y
        else:
            return z

    def loss(self, x: th.Tensor) -> th.Tensor:
        z, mu, log_var = self.encoder(x)
        kl = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp_()).mean()
        y = self.decoder(z)
        re = self.re_loss(y, x)
        return kl + re
