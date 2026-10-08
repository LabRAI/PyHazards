"""TropiCycloneNet (TCN_M): multimodal GAN for global tropical cyclone track and intensity forecasts.

Paper: C. Huang, P. Mu, J. Zhang, S. Chan, S. Zhang, H. Yan, S. Chen and C. Bai, "Benchmark dataset
and deep learning method for global tropical cyclone forecasting", Nature Communications 16, 5923
(2025), doi:10.1038/s41467-025-61087-4.

Port of the generator and discriminator of the official code release "TropiCycloneNet Model: A
Multimodal Tropical Cyclone Prediction Model Guided by Environmental Information" (Zenodo
doi:10.5281/zenodo.15024028, ``TropiCycloneNet.zip``, git commit b4e7101 of 2025-03-14), licensed
CC BY 4.0 by Cheng Huang et al.: ``TCNM/models_prior_unet.py`` (``make_mlp``, ``Encoder``,
``Decoder``, ``TrajectoryGenerator``, ``TrajectoryDiscriminator``), ``TCNM/Unet3D_merge_tiny.py``
(``Conv3d``, ``Down``, ``Up``, ``OutConv``, ``Unet3D``) and ``TCNM/env_net_transformer_gphsplit.py``
(``Env_net``). These files are identical (up to line endings) to github.com/xiaochengfuhuo/TropiCycloneNet
at de9cf0e9. Changes made here: device-agnostic tensors instead of ``.cuda()``, input validation,
the generator-index loop fix described below, and the :meth:`TropiCycloneNet.forecast` adapter.
Module and parameter names are unchanged, so the released checkpoint
(``checkpoint_with_model_16000.pt``, CC BY 4.0) loads with ``strict=True``, and modules are created
in the official order, so the same seed gives the same initial weights.

The generator (4,767,195 parameters in the released configuration) observes 8 six-hourly steps of
the best track (``obs_traj``: longitude, latitude, central pressure, maximum wind, normalised as in
:data:`TCND_NORMALIZATION`; ``obs_traj_rel``: their 6-hourly differences, 0 at the first step),
500 hPa geopotential height crops resized to 64 x 64 (``image_obs``, ``(batch, 1, 8, 64, 64)``) and
the TCND environment features (``env_data``, nine one-hot or scalar features per step). A 3D U-Net
extrapolates the GPH sequence, LSTM encoders summarise the track with the GPH embeddings, and the
Env-T-Net transformer (``env_net_chooser``) summarises the environment. Six LSTM decoders
(generators) each roll out 4 steps (6, 12, 18, 24 h) of relative displacements; a generator-chooser
MLP gives categorical logits over the six decoders, and every sample draws a decoder index from
them plus a 16-d Gaussian noise vector. Outputs are relative (6-hourly) normalised displacements;
:meth:`TropiCycloneNet.forecast` turns them into absolute positions (degrees), central pressure
(hPa) and maximum sustained wind (m/s).

Official behaviour that is kept: unused modules (``env_net``, ``time_embedding`` layers,
``traj_score``/``inte_score`` placeholders) exist and are counted; the decoders ignore the absolute
position they track; the mixing MLP ends with a ReLU; ``Up`` pads the decoder features with the
time difference on the width axis and the height difference on the time axis (all differences
are 0 for the released 8-step, 64 x 64 configuration).
"""

from __future__ import annotations

import collections
import hashlib
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

# Data1d normalisation of the TropiCycloneNet Dataset (TCND) and of the official evaluation
# (TCNM/losses.py ``toNE``): physical = normalised * scale + offset.
TCND_NORMALIZATION: Dict[str, Tuple[float, float]] = {
    "lon": (5.0, 180.0),  # degrees east
    "lat": (5.0, 0.0),  # degrees north
    "pres": (50.0, 960.0),  # hPa
    "wind": (25.0, 40.0),  # m/s, 2-minute mean maximum sustained wind
}
TRACK_VARIABLES = ("lon", "lat", "pres", "wind")

# TCND Env-Data features (key, width) in the order Env_net concatenates them.
ENV_FEATURES: Tuple[Tuple[str, int], ...] = (
    ("wind", 1),
    ("intensity_class", 6),
    ("move_velocity", 1),
    ("month", 12),
    ("location_long", 36),
    ("location_lat", 12),
    ("history_direction12", 8),
    ("history_direction24", 8),
    ("history_inte_change24", 4),
)

# Released checkpoint (Zenodo record 15024028, CC BY 4.0).
CHECKPOINT_URL = "https://zenodo.org/records/15024028/files/checkpoint_with_model_16000.pt"
CHECKPOINT_SHA256 = "68575d1d10fcd2d44eba2d54d0625410f5108cd3bf5d7b3e15df7491ba9f6215"
# Generator / discriminator arguments of that checkpoint (its ``args``).
RELEASED_CONFIG: Dict[str, Any] = {
    "obs_len": 8,
    "pred_len": 4,
    "embedding_dim": 32,
    "encoder_h_dim": 64,
    "decoder_h_dim": 64,
    "mlp_dim": 128,
    "num_layers": 1,
    "noise_dim": (16,),
    "noise_type": "gaussian",
    "noise_mix_type": "ped",
    "pooling_type": None,
    "pool_every_timestep": False,
    "dropout": 0.0,
    "bottleneck_dim": 16,
    "batch_norm": False,
}


def make_mlp(dim_list: Sequence[int], activation: str = "relu", batch_norm: bool = True, dropout: float = 0) -> nn.Sequential:
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))
        if activation == "relu":
            layers.append(nn.ReLU())
        elif activation == "leakyrelu":
            layers.append(nn.LeakyReLU())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)


def get_noise(shape: Sequence[int], noise_type: str, device=None, dtype=None) -> torch.Tensor:
    # Drawn on the CPU generator and then moved, as the official ``torch.randn(...).cuda()`` does,
    # so a seed gives the official noise on any device.
    if noise_type == "gaussian":
        noise = torch.randn(*shape)
    elif noise_type == "uniform":
        noise = torch.rand(*shape).sub_(0.5).mul_(2.0)
    else:
        raise ValueError('Unrecognized noise type "%s"' % noise_type)
    return noise.to(device=device, dtype=dtype)


# ---------------------------------------------------------------------------------------------
# 3D U-Net over the GPH sequence (TCNM/Unet3D_merge_tiny.py)
# ---------------------------------------------------------------------------------------------


class Conv3d(nn.Module):
    """Two 3D conv - BN - ReLU layers plus a bias-free 1x1x1 residual convolution."""

    def __init__(self, in_channel: int, out_channel: int, kernel_size, stride, padding):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(True),
        )
        self.residual = nn.Conv3d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(x)) + self.residual(x)


class Down(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, kernel_size, stride):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(kernel_size=kernel_size, stride=stride),
            Conv3d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, x1_in: int, x2_in: int, out_channel: int, kernel_size, stride, padding):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose3d(x1_in, x1_in, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            nn.ReLU(),
        )
        self.conv = Conv3d(x1_in + x2_in, out_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diff_t = x2.size(2) - x1.size(2)
        diff_y = x2.size(3) - x1.size(3)
        diff_x = x2.size(4) - x1.size(4)
        # Official padding order: the time difference pads the last (width) axis and the height
        # difference pads the time axis. All differences are 0 in the released configuration.
        x1 = F.pad(
            x1,
            [diff_t // 2, diff_t - diff_t // 2, diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
        )
        return self.conv(torch.cat([x2, x1], dim=1))


class OutConv(nn.Module):
    """Upsamples the four decoder levels to full resolution, stacks them in time and convolves."""

    def __init__(self, in_channel_list: Sequence[int], out_channel: int, kernel_size, stride, padding):
        super().__init__()
        levels = len(in_channel_list) - 1
        up_list = []
        for index, channel in enumerate(in_channel_list[:-1]):
            factor = 2 ** (levels - index)
            up_list.append(
                nn.Sequential(
                    nn.ConvTranspose3d(
                        channel, channel, kernel_size=[1, factor, factor], stride=[1, factor, factor], padding=padding, bias=True
                    ),
                    nn.ReLU(),
                    nn.Conv3d(channel, in_channel_list[-1], kernel_size=3, stride=1, padding=1, bias=True),
                    nn.BatchNorm3d(in_channel_list[-1]),
                    nn.ReLU(inplace=True),
                )
            )
        self.up_list = nn.ModuleList(up_list)
        self.conv = nn.Sequential(
            nn.Conv3d(in_channel_list[-1], out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(),
            nn.Conv3d(out_channel, out_channel, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x: Sequence[torch.Tensor]) -> torch.Tensor:
        x6, x7, x8, x9 = tuple(x)
        x6 = self.up_list[0](x6)
        x7 = self.up_list[1](x7)
        x8 = self.up_list[2](x8)
        return self.conv(torch.cat([x6, x7, x8, x9], dim=2))


class Unet3D(nn.Module):
    """3D U-Net mapping 8 observed GPH frames ``(batch, 1, 8, 64, 64)`` to 11 frames (steps 2-12)."""

    def __init__(self, in_channel: int, out_channel: int):
        super().__init__()
        self.inc = Conv3d(in_channel, 16, kernel_size=3, stride=1, padding=1)
        self.down1 = Down(16, 32, kernel_size=[1, 2, 2], stride=[1, 2, 2])
        self.down2 = Down(32, 64, kernel_size=[1, 2, 2], stride=[1, 2, 2])
        self.down3 = Down(64, 128, kernel_size=[2, 2, 2], stride=[2, 2, 2])
        self.down4 = Down(128, 128, kernel_size=[2, 2, 2], stride=[2, 2, 2])
        self.up1 = Up(128, 128, 64, kernel_size=[2, 2, 2], stride=[2, 2, 2], padding=0)
        self.up2 = Up(64, 64, 32, kernel_size=[2, 2, 2], stride=[2, 2, 2], padding=0)
        self.up3 = Up(32, 32, 16, kernel_size=[1, 2, 2], stride=[1, 2, 2], padding=0)
        self.up4 = Up(16, 16, 16, kernel_size=[1, 2, 2], stride=[1, 2, 2], padding=0)
        self.outc = OutConv([64, 32, 16, 16], out_channel, kernel_size=[18, 1, 1], stride=[1, 1, 1], padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.up1(x5, x4)
        x7 = self.up2(x6, x3)
        x8 = self.up3(x7, x2)
        x9 = self.up4(x8, x1)
        return self.outc([x6, x7, x8, x9])


# ---------------------------------------------------------------------------------------------
# Env-T-Net (TCNM/env_net_transformer_gphsplit.py)
# ---------------------------------------------------------------------------------------------


class Env_net(nn.Module):
    """Embeds the TCND environment features and GPH maps per step and encodes them with a transformer.

    ``forward(env_data, gph)`` with ``env_data[key]`` ``(batch, obs_len, width)`` for the keys of
    :data:`ENV_FEATURES` and ``gph`` ``(batch, 1, obs_len, 64, 64)``; returns the 64-d feature of
    the last step and two placeholder zeros, as the official module does.
    """

    def __init__(self, obs_len: int = 8):
        super().__init__()
        embed_dim = 16
        self.data_embed = nn.ModuleDict()
        for key, width in ENV_FEATURES:
            self.data_embed[key] = nn.Linear(width, embed_dim)
        self.GPH_embed = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(inplace=True),
            nn.AvgPool2d(8, 8),
        )
        env_f_in = len(self.data_embed) * 16 + 8 * 8
        self.evn_extract = nn.Sequential(
            nn.Linear(env_f_in, env_f_in // 2),
            nn.ReLU(),
            nn.Linear(env_f_in // 2, env_f_in // 2),
            nn.ReLU(),
            nn.Linear(env_f_in // 2, 64),
        )
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4)
        # enable_nested_tensor only matters with padding masks (never used); False avoids a warning.
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2, enable_nested_tensor=False)

    def forward(self, env_data: Mapping[str, torch.Tensor], gph: torch.Tensor):
        gph = gph.permute(0, 2, 1, 3, 4)
        batch, steps = gph.shape[:2]
        embeds = [self.data_embed[key](env_data[key]) for key in self.data_embed]
        gph_feature = torch.stack([self.GPH_embed(gph[:, step]).reshape(batch, -1) for step in range(steps)], dim=1)
        embeds.append(gph_feature)
        feature_in = self.evn_extract(torch.cat(embeds, dim=2)).permute(1, 0, 2)
        output = self.encoder(feature_in)
        return output[-1], 0, 0


# ---------------------------------------------------------------------------------------------
# Trajectory encoder / decoders / generator / discriminator (TCNM/models_prior_unet.py)
# ---------------------------------------------------------------------------------------------


class Encoder(nn.Module):
    """LSTM over embedded relative track steps plus image embeddings."""

    def __init__(self, embedding_dim: int = 64, h_dim: int = 64, mlp_dim: int = 1024, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.mlp_dim = 1024
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.encoder = nn.LSTM(embedding_dim, h_dim, num_layers, dropout=dropout)
        self.spatial_embedding = nn.Linear(4, embedding_dim)
        self.time_embedding = nn.Linear(4, embedding_dim)  # created but unused, as in the official code

    def init_hidden(self, batch: int, reference: torch.Tensor):
        zeros = reference.new_zeros(self.num_layers, batch, self.h_dim)
        return zeros, zeros.clone()

    def forward(self, obs_traj: torch.Tensor, img_embed_input: torch.Tensor) -> Dict[str, Any]:
        batch = obs_traj.size(1)
        embedding = self.spatial_embedding(obs_traj.reshape(-1, obs_traj.size(2))).view(-1, batch, self.embedding_dim)
        embedding = embedding + img_embed_input
        output, state = self.encoder(embedding, self.init_hidden(batch, embedding))
        return {"final_h": state, "output": output}


class Decoder(nn.Module):
    """One generator: an LSTM rolling out ``seq_len`` relative steps from the mixed hidden state."""

    def __init__(
        self,
        seq_len: int,
        embedding_dim: int = 64,
        h_dim: int = 128,
        mlp_dim: int = 1024,
        num_layers: int = 1,
        pool_every_timestep: bool = True,
        dropout: float = 0.0,
        bottleneck_dim: int = 1024,
        activation: str = "relu",
        batch_norm: bool = True,
        pooling_type: Optional[str] = "pool_net",
        neighborhood_size: float = 2.0,
        grid_size: int = 8,
        embeddings_dim: int = 128,
        h_dims: int = 128,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.pool_every_timestep = pool_every_timestep
        self.decoder = nn.LSTM(embedding_dim, h_dim, num_layers, dropout=dropout)
        if pool_every_timestep:
            # Created (and counted) but never applied, as in the official code.
            self.mlp = make_mlp([h_dim + bottleneck_dim, mlp_dim, h_dim], activation=activation, batch_norm=batch_norm, dropout=dropout)
        self.spatial_embedding = nn.Linear(4, embedding_dim)
        self.time_embedding = nn.Linear(4, embedding_dim)  # unused
        self.hidden2pos = nn.Linear(h_dim, 4)

    def forward(self, obs_traj, obs_traj_rel, last_pos, last_pos_rel, state_tuple, seq_start_end, decoder_img, last_img):
        batch = last_pos.size(0)
        steps = []
        decoder_input = self.spatial_embedding(last_pos_rel).view(-1, batch, self.embedding_dim)
        decoder_input = decoder_input + last_img.unsqueeze(0)
        for step in range(self.seq_len):
            output, state_tuple = self.decoder(decoder_input, state_tuple)
            rel_pos = self.hidden2pos(output.view(-1, self.h_dim))
            # The official decoder also computes rel_pos + last_pos, which it never uses.
            rel_pos = rel_pos.unsqueeze(0)
            decoder_input = self.spatial_embedding(rel_pos).view(-1, batch, self.embedding_dim)
            decoder_input = decoder_input + decoder_img[step].unsqueeze(0)
            steps.append(rel_pos.view(batch, -1))
        return torch.stack(steps, dim=0), state_tuple[0]


def _env_from_mapping(batch: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
    env = batch.get("env_data")
    if env is None:
        env = {key: batch[key] for key, _ in ENV_FEATURES if key in batch}
    return dict(env)


class TrajectoryGenerator(nn.Module):
    """TCN_M generator (``TrajectoryGenerator`` of the official code)."""

    def __init__(
        self,
        obs_len: int = 8,
        pred_len: int = 4,
        embedding_dim: int = 32,
        encoder_h_dim: int = 64,
        decoder_h_dim: int = 64,
        mlp_dim: int = 128,
        num_layers: int = 1,
        noise_dim: Sequence[int] = (16,),
        noise_type: str = "gaussian",
        noise_mix_type: str = "ped",
        pooling_type: Optional[str] = None,
        pool_every_timestep: bool = False,
        dropout: float = 0.0,
        bottleneck_dim: int = 16,
        activation: str = "relu",
        batch_norm: bool = False,
        neighborhood_size: float = 2.0,
        grid_size: int = 8,
        num_gs: int = 6,
        num_sample: int = 6,
        official_sample_loop: bool = False,
    ):
        super().__init__()
        if pooling_type and pooling_type.lower() == "none":
            pooling_type = None
        if pooling_type is not None:
            raise ValueError(
                "TropiCycloneNet has no social-pooling module (the official 'pool_net'/'spool' branches are empty); "
                "use pooling_type=None"
            )
        if int(obs_len) != 8 or int(pred_len) != 4:
            raise ValueError("TropiCycloneNet's GPH U-Net is built for obs_len=8 and pred_len=4 (11 output frames)")
        self.obs_len = int(obs_len)
        self.pred_len = int(pred_len)
        self.mlp_dim = mlp_dim
        self.encoder_h_dim = encoder_h_dim
        self.decoder_h_dim = decoder_h_dim
        self.embedding_dim = embedding_dim
        self.noise_dim = tuple(noise_dim) if noise_dim is not None else (0,)
        self.num_layers = num_layers
        self.noise_type = noise_type
        self.noise_mix_type = noise_mix_type
        self.pooling_type = pooling_type
        self.noise_first_dim = 0
        self.pool_every_timestep = pool_every_timestep
        self.bottleneck_dim = 1024
        self.num_gs = int(num_gs)
        self.num_sample = int(num_sample)
        # The official sampling loop visits generator indices 0..(number of distinct sampled
        # indices - 1) instead of the distinct indices themselves; see ``forward``.
        self.official_sample_loop = bool(official_sample_loop)

        self.Unet = Unet3D(1, 1)
        self.img_embedding = nn.Linear(64 * 64, 32)
        self.img_embedding_real = nn.Linear(64 * 64, 32)
        self.env_net = Env_net()  # created but unused in the official forward pass
        self.env_net_chooser = Env_net()
        self.feature2dech_env = nn.Linear(128, 64)
        self.feature2dech = nn.Linear(128, 64)
        self.encoder = Encoder(embedding_dim=embedding_dim, h_dim=encoder_h_dim, mlp_dim=mlp_dim, num_layers=num_layers, dropout=dropout)
        self.encoder_env = Encoder(
            embedding_dim=embedding_dim, h_dim=encoder_h_dim, mlp_dim=mlp_dim, num_layers=num_layers, dropout=dropout
        )
        self.gs = nn.ModuleList(
            [
                Decoder(
                    pred_len,
                    embedding_dim=embedding_dim,
                    h_dim=decoder_h_dim,
                    mlp_dim=mlp_dim,
                    num_layers=num_layers,
                    pool_every_timestep=pool_every_timestep,
                    dropout=dropout,
                    bottleneck_dim=bottleneck_dim,
                    activation=activation,
                    batch_norm=batch_norm,
                    pooling_type=pooling_type,
                    grid_size=grid_size,
                    neighborhood_size=neighborhood_size,
                    embeddings_dim=embedding_dim,
                    h_dims=encoder_h_dim,
                )
                for _ in range(self.num_gs)
            ]
        )
        self.net_chooser = nn.Sequential(
            nn.Linear(encoder_h_dim, encoder_h_dim // 2),
            nn.ReLU(),
            nn.Linear(encoder_h_dim // 2, encoder_h_dim // 2),
            nn.ReLU(),
            nn.Linear(encoder_h_dim // 2, self.num_gs),
        )
        if self.noise_dim[0] == 0:
            self.noise_dim = None
        else:
            self.noise_first_dim = self.noise_dim[0]
        if self.mlp_decoder_needed():
            self.mlp_decoder_context = make_mlp(
                [encoder_h_dim, mlp_dim, decoder_h_dim - self.noise_first_dim],
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout,
            )

    # -- official helpers -------------------------------------------------------------------

    def mlp_decoder_needed(self) -> bool:
        return bool(self.noise_dim or self.pooling_type or self.encoder_h_dim != self.decoder_h_dim)

    def add_noise(self, _input: torch.Tensor, seq_start_end: torch.Tensor, user_noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not self.noise_dim:
            return _input
        if self.noise_mix_type == "global":
            noise_shape = (seq_start_end.size(0),) + self.noise_dim
        else:
            noise_shape = (_input.size(0),) + self.noise_dim
        if user_noise is not None:
            z_decoder = user_noise
        else:
            z_decoder = get_noise(noise_shape, self.noise_type, device=_input.device, dtype=_input.dtype)
        if self.noise_mix_type == "global":
            pieces = []
            for idx, (start, end) in enumerate(seq_start_end):
                start, end = int(start), int(end)
                pieces.append(torch.cat([_input[start:end], z_decoder[idx].view(1, -1).repeat(end - start, 1)], dim=1))
            return torch.cat(pieces, dim=0)
        return torch.cat([_input, z_decoder], dim=1)

    def get_samples(self, enc_h: torch.Tensor, num_samples: int = 6):
        """Chooser logits ``(batch, num_gs)`` and sampled generator indices ``(batch, num_samples)``."""
        net_chooser_out = self.net_chooser(enc_h).reshape(-1, self.num_gs)
        sampled_gen_idxs = Categorical(logits=net_chooser_out).sample((num_samples,)).transpose(0, 1)
        return net_chooser_out, sampled_gen_idxs.detach()

    def mix_noise(self, final_encoder_h, seq_start_end, batch: int, user_noise=None):
        context = final_encoder_h.view(-1, self.encoder_h_dim)
        noise_input = self.mlp_decoder_context(context) if self.mlp_decoder_needed() else context
        decoder_h = self.add_noise(noise_input, seq_start_end, user_noise=user_noise)
        decoder_h = decoder_h.view(-1, batch, self.encoder_h_dim)
        decoder_c = decoder_h.new_zeros(self.num_layers, batch, self.decoder_h_dim)
        return decoder_h, decoder_c

    # -- forward ------------------------------------------------------------------------------

    def _check_inputs(self, obs_traj, obs_traj_rel, image_obs, env_data) -> None:
        if obs_traj.ndim != 3 or obs_traj.size(0) != self.obs_len or obs_traj.size(2) != 4:
            raise ValueError(
                f"obs_traj must be shaped (obs_len={self.obs_len}, batch, 4 [lon, lat, pres, wind]), got {tuple(obs_traj.shape)}"
            )
        if obs_traj_rel.shape != obs_traj.shape:
            raise ValueError(f"obs_traj_rel must have the shape of obs_traj {tuple(obs_traj.shape)}, got {tuple(obs_traj_rel.shape)}")
        batch = obs_traj.size(1)
        if image_obs.ndim != 5 or tuple(image_obs.shape) != (batch, 1, self.obs_len, 64, 64):
            raise ValueError(f"image_obs must be shaped (batch, 1, {self.obs_len}, 64, 64), got {tuple(image_obs.shape)}")
        for key, width in ENV_FEATURES:
            if key not in env_data:
                raise ValueError(f"env_data is missing {key!r}; expected keys {[name for name, _ in ENV_FEATURES]}")
            if tuple(env_data[key].shape) != (batch, self.obs_len, width):
                raise ValueError(f"env_data[{key!r}] must be shaped (batch, {self.obs_len}, {width}), got {tuple(env_data[key].shape)}")

    def forward(
        self,
        obs_traj,
        obs_traj_rel: Optional[torch.Tensor] = None,
        seq_start_end: Optional[torch.Tensor] = None,
        image_obs: Optional[torch.Tensor] = None,
        env_data: Optional[Mapping[str, torch.Tensor]] = None,
        num_samples: int = 1,
        all_g_out: bool = False,
        predrnn_img=None,
        user_noise=None,
    ):
        """Official generator forward pass.

        Returns ``(pred_traj_fake_rel_nums, image_out, net_chooser_out, sampled_gen_idxs)``:
        relative normalised steps ``(pred_len, K, batch, 4)`` with ``K = num_samples`` (or
        ``num_gs`` with ``all_g_out=True``: every decoder once, sharing one noise draw), the
        observed-plus-extrapolated GPH frames ``(batch, 1, obs_len + pred_len, 64, 64)``, the
        chooser logits ``(batch, num_gs)`` and the sampled decoder indices ``(batch, num_samples)``.
        ``obs_traj`` may also be a mapping with keys ``obs_traj``, ``obs_traj_rel``, ``image_obs``,
        optional ``seq_start_end``, and ``env_data`` (or the nine env keys at top level).
        """
        if isinstance(obs_traj, Mapping):
            batch_dict = obs_traj
            obs_traj = batch_dict["obs_traj"]
            obs_traj_rel = batch_dict["obs_traj_rel"]
            image_obs = batch_dict["image_obs"]
            seq_start_end = batch_dict.get("seq_start_end", seq_start_end)
            env_data = _env_from_mapping(batch_dict)
        if obs_traj_rel is None or image_obs is None or env_data is None:
            raise ValueError("TropiCycloneNet needs obs_traj, obs_traj_rel, image_obs and env_data")
        self._check_inputs(obs_traj, obs_traj_rel, image_obs, env_data)
        batch = obs_traj_rel.size(1)
        obs_len = obs_traj_rel.size(0)
        if seq_start_end is None:
            starts = torch.arange(batch, device=obs_traj.device)
            seq_start_end = torch.stack([starts, starts + 1], dim=1)

        # Chooser branch: track encoder on the observed GPH embeddings plus Env-T-Net.
        encoder_img_real = self.img_embedding_real(image_obs.reshape(batch, self.obs_len, -1)).permute(1, 0, 2)
        final_encoder_env_h = self.encoder_env(obs_traj_rel, encoder_img_real)["final_h"][0]
        evn_feature_chooser, _, _ = self.env_net_chooser(env_data, image_obs)
        dec_h_evn = self.feature2dech_env(torch.cat([final_encoder_env_h.reshape(batch, -1), evn_feature_chooser], dim=1))

        # Generator branch: GPH U-Net extrapolation, track encoder, mixed decoder state.
        unet_out = self.Unet(image_obs)
        all_img = torch.cat([image_obs[:, :, 0].unsqueeze(2), unet_out], dim=2)
        img_embed_input = self.img_embedding(all_img.reshape(batch, self.obs_len + self.pred_len, -1)).permute(1, 0, 2)
        final_encoder_h = self.encoder(obs_traj_rel, img_embed_input[:obs_len])["final_h"][0]
        dec_h = self.feature2dech(torch.cat([final_encoder_h.reshape(batch, -1), evn_feature_chooser], dim=1)).unsqueeze(0)

        last_pos = obs_traj[-1]
        last_pos_rel = obs_traj_rel[-1]
        decoder_img = img_embed_input[obs_len:]
        last_img = img_embed_input[obs_len - 1]
        if all_g_out:
            preds_rel = []
            with torch.no_grad():
                state_tuple = self.mix_noise(dec_h, seq_start_end, batch, user_noise=user_noise)
                for decoder in self.gs:
                    pred_rel, _ = decoder(obs_traj, obs_traj_rel, last_pos, last_pos_rel, state_tuple, seq_start_end, decoder_img, last_img)
                    preds_rel.append(pred_rel.reshape(self.pred_len, 1, batch, 4))
            pred_traj_fake_rel_nums = torch.cat(preds_rel, dim=1)
            net_chooser_out, sampled_gen_idxs = self.get_samples(dec_h_evn, num_samples)
        else:
            with torch.no_grad():
                net_chooser_out, sampled_gen_idxs = self.get_samples(dec_h_evn, num_samples)
            preds_rel = []
            for sample in range(num_samples):
                prediction = obs_traj.new_ones(self.pred_len, batch, 4)
                gen_index = sampled_gen_idxs[:, sample]
                distinct = torch.unique(gen_index).tolist()
                # Official loop: ``for g_i in range(len(distinct))`` matches indices 0..len-1, so
                # storms whose sampled decoder index is >= the number of distinct indices in the
                # batch keep the placeholder value 1 for every step. PyHazards visits the distinct
                # indices themselves unless ``official_sample_loop`` is set.
                visit = range(len(distinct)) if self.official_sample_loop else distinct
                for g_i in visit:
                    mask = gen_index == g_i
                    count = int(mask.sum())
                    if count < 1:
                        continue
                    state_tuple = self.mix_noise(dec_h[:, mask], seq_start_end[mask], count)
                    pred_rel, _ = self.gs[g_i](
                        obs_traj[:, mask],
                        obs_traj_rel[:, mask],
                        last_pos[mask],
                        last_pos_rel[mask],
                        state_tuple,
                        seq_start_end[mask],
                        decoder_img[:, mask],
                        last_img[mask],
                    )
                    prediction[:, mask] = pred_rel
                preds_rel.append(prediction.reshape(self.pred_len, 1, batch, 4))
            pred_traj_fake_rel_nums = torch.cat(preds_rel, dim=1)
        return pred_traj_fake_rel_nums, all_img, net_chooser_out, sampled_gen_idxs


class TrajectoryDiscriminator(nn.Module):
    """TCN_M discriminator: track encoder over GPH embeddings plus a real/fake MLP (231,009 parameters)."""

    def __init__(
        self,
        obs_len: int = 8,
        pred_len: int = 4,
        embedding_dim: int = 32,
        h_dim: int = 128,
        mlp_dim: int = 128,
        num_layers: int = 1,
        activation: str = "relu",
        batch_norm: bool = False,
        dropout: float = 0.0,
        d_type: str = "local",
    ):
        super().__init__()
        if d_type != "local":
            raise ValueError("only the official d_type='local' discriminator is implemented")
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len
        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.d_type = d_type
        self.img_embedding = nn.Linear(64 * 64, 32)
        self.encoder = Encoder(embedding_dim=embedding_dim, h_dim=h_dim, mlp_dim=mlp_dim, num_layers=num_layers, dropout=dropout)
        self.real_classifier = make_mlp([h_dim, mlp_dim, 1], activation=activation, batch_norm=batch_norm, dropout=dropout)

    def forward(self, traj: torch.Tensor, traj_rel: torch.Tensor, seq_start_end: torch.Tensor, img: torch.Tensor):
        """Scores ``(batch, 1)`` (after the official final ReLU) and the encoder feature ``(batch, h_dim)``."""
        if img.ndim != 5 or traj_rel.ndim != 3 or img.size(2) != traj_rel.size(0) or img.size(0) != traj_rel.size(1):
            raise ValueError(
                f"expected traj_rel (seq_len, batch, 4) and img (batch, 1, seq_len, 64, 64), got {tuple(traj_rel.shape)} and {tuple(img.shape)}"
            )
        batch, _, length = img.shape[:3]
        img_embed = self.img_embedding(img.reshape(batch, length, -1)).permute(1, 0, 2)
        final_h = self.encoder(traj_rel, img_embed)["final_h"][0]
        classifier_input = final_h.squeeze()
        return self.real_classifier(classifier_input), classifier_input


# ---------------------------------------------------------------------------------------------
# PyHazards entry point
# ---------------------------------------------------------------------------------------------


def denormalize_track(values: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Map TCND-normalised ``[..., 4]`` values (lon, lat, pres, wind) to physical units."""
    return {name: values[..., i] * TCND_NORMALIZATION[name][0] + TCND_NORMALIZATION[name][1] for i, name in enumerate(TRACK_VARIABLES)}


def normalize_track(lon, lat, pres, wind) -> torch.Tensor:
    """Inverse of :func:`denormalize_track`; returns ``[..., 4]`` normalised values."""
    parts = [torch.as_tensor(v, dtype=torch.float32) for v in (lon, lat, pres, wind)]
    return torch.stack(
        [(part - TCND_NORMALIZATION[name][1]) / TCND_NORMALIZATION[name][0] for part, name in zip(parts, TRACK_VARIABLES)], dim=-1
    )


class TropiCycloneNet(TrajectoryGenerator):
    """The TCN_M generator with a :meth:`forecast` adapter for the PyHazards cyclone benchmark.

    The module is the official generator itself (no key prefix), so official generator state
    dicts load with ``strict=True``; :func:`load_tropicyclonenet_checkpoint` loads the released
    one. ``forward`` keeps the official signature and outputs.
    """

    def forecast(self, batch: Mapping[str, Any], num_samples: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Sampled forecasts in physical units.

        Returns ``lat``, ``lon`` (degrees), ``pres`` (hPa) and ``wind`` (m/s), each
        ``(batch, num_samples, pred_len)`` for lead times 6, 12, 18 and 24 h, plus the chooser
        ``logits`` and the sampled ``generator_index``. Positions are the last observed position
        plus the cumulative sum of the predicted steps, as in the official evaluation.
        """
        num_samples = int(num_samples or self.num_sample)
        rel, _, logits, indices = self.forward(batch, num_samples=num_samples)
        obs_traj = batch["obs_traj"]
        absolute = torch.cumsum(rel, dim=0) + obs_traj[-1].unsqueeze(0).unsqueeze(0)
        physical = denormalize_track(absolute.permute(2, 1, 0, 3))
        return {**physical, "logits": logits, "generator_index": indices}


def load_tropicyclonenet_checkpoint(
    path: Union[str, Path], model: Optional[nn.Module] = None, state: str = "g_state", verify_sha256: bool = True
) -> nn.Module:
    """Load the released ``checkpoint_with_model_16000.pt`` into a generator (or ``model``).

    The checkpoint stores training history in ``collections.defaultdict`` objects, which
    ``torch.load(weights_only=True)`` rejects, so it is unpickled with ``weights_only=False``;
    by default this only happens after the file's sha256 matched the released file
    (:data:`CHECKPOINT_SHA256`). ``state`` picks ``g_state`` (used by the official evaluation;
    identical to ``g_best_state`` in the release) or ``d_state`` for a discriminator.
    """
    path = Path(path)
    if verify_sha256:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != CHECKPOINT_SHA256:
            raise ValueError(
                f"{path} has sha256 {digest}, not the released checkpoint {CHECKPOINT_SHA256}; "
                "pass verify_sha256=False only for a file you trust (it is unpickled)"
            )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if model is None:
        model = TrajectoryDiscriminator() if state.startswith("d_") else TropiCycloneNet()
    model.load_state_dict(checkpoint[state], strict=True)
    return model


def tropicyclonenet_builder(
    task: str,
    official_sample_loop: bool = False,
    num_sample: int = 6,
    **kwargs,
) -> nn.Module:
    if task.lower() not in {"regression", "forecasting"}:
        raise ValueError("tropicyclonenet forecasts tracks and intensities; use task='regression' or 'forecasting'.")
    config = {key: kwargs[key] for key in RELEASED_CONFIG if key in kwargs}
    return TropiCycloneNet(**{**RELEASED_CONFIG, **config}, num_sample=num_sample, official_sample_loop=official_sample_loop)


__all__ = [
    "CHECKPOINT_SHA256",
    "CHECKPOINT_URL",
    "ENV_FEATURES",
    "RELEASED_CONFIG",
    "TCND_NORMALIZATION",
    "TRACK_VARIABLES",
    "TrajectoryDiscriminator",
    "TrajectoryGenerator",
    "TropiCycloneNet",
    "denormalize_track",
    "load_tropicyclonenet_checkpoint",
    "normalize_track",
    "tropicyclonenet_builder",
]
