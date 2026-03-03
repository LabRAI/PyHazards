from .backbones import CNNPatchEncoder, MLPBackbone, TemporalEncoder
from .builder import build_model, default_builder
from .heads import ClassificationHead, RegressionHead, SegmentationHead
from .registry import available_models, register_model

# Wildfire models
from .wildfire_fpa_autoencoder import (
    WildfireFPAAutoencoder,
    wildfire_fpa_autoencoder_builder,
)
from .wildfire_fpa_dnn import WildfireFPADNN, wildfire_fpa_dnn_builder
from .wildfire_fpa_forecast import WildfireFPAForecast, wildfire_fpa_forecast_builder
from .wildfire_fpa_lstm import WildfireFPALSTM, wildfire_fpa_lstm_builder
from .wildfire_mamba import WildfireMamba, wildfire_mamba_builder
from .wildfire_aspp import WildfireASPP, TverskyLoss, wildfire_aspp_builder
from .cnn_aspp import WildfireCNNASPP, cnn_aspp_builder
from .hydrographnet import HydroGraphNet, HydroGraphNetLoss, hydrographnet_builder


__all__ = [
    # Core API
    "build_model",
    "available_models",
    "register_model",

    # Backbones
    "MLPBackbone",
    "CNNPatchEncoder",
    "TemporalEncoder",

    # Heads
    "ClassificationHead",
    "RegressionHead",
    "SegmentationHead",

    # Wildfire models
    "WildfireFPADNN",
    "wildfire_fpa_dnn_builder",
    "WildfireFPAAutoencoder",
    "wildfire_fpa_autoencoder_builder",
    "WildfireFPAForecast",
    "wildfire_fpa_forecast_builder",
    "WildfireMamba",
    "wildfire_mamba_builder",
    "WildfireASPP",
    "TverskyLoss",
    "wildfire_aspp_builder",
    "WildfireCNNASPP",
    "cnn_aspp_builder",
    "HydroGraphNet",
    "HydroGraphNetLoss",
    "hydrographnet_builder",
]

# -------------------------------------------------
# Register default backbones
# -------------------------------------------------
register_model(
    "mlp",
    default_builder,
    defaults={"hidden_dim": 256, "depth": 2},
)

register_model(
    "cnn",
    default_builder,
    defaults={"hidden_dim": 64, "in_channels": 3},
)

register_model(
    "temporal",
    default_builder,
    defaults={"hidden_dim": 128, "num_layers": 1},
)

# -------------------------------------------------
# Register wildfire models
# -------------------------------------------------
register_model(
    "wildfire_fpa_dnn",
    wildfire_fpa_dnn_builder,
    defaults={
        "out_dim": 5,
        "depth": 2,
        "hidden_dim": 64,
        "activation": "relu",
        "dropout": 0.0,
    },
)

register_model(
    "wildfire_fpa_forecast",
    wildfire_fpa_forecast_builder,
    defaults={
        "hidden_dim": 64,
        "output_dim": 5,
        "latent_dim": 32,
        "num_layers": 1,
        "dropout": 0.2,
        "lookback": 50,
    },
)

register_model(
    "wildfire_fpa_autoencoder",
    wildfire_fpa_autoencoder_builder,
    defaults={
        "hidden_dim": 64,
        "latent_dim": 32,
        "num_layers": 1,
        "dropout": 0.2,
        "lookback": 50,
    },
)

register_model(
    "wildfire_fpa_lstm",
    wildfire_fpa_lstm_builder,
    defaults={
        "hidden_dim": 64,
        "output_dim": 5,
        "num_layers": 1,
        "dropout": 0.2,
        "lookback": 50,
    },
)

register_model(
    "wildfire_mamba",
    wildfire_mamba_builder,
    defaults={
        "hidden_dim": 128,
        "gcn_hidden": 64,
        "mamba_layers": 2,
        "state_dim": 64,
        "conv_kernel": 5,
        "dropout": 0.1,
        "with_count_head": False,
    },
)

register_model(
    "wildfire_aspp",
    wildfire_aspp_builder,
    defaults={
        "in_channels": 12,
    },
)

register_model(
    "wildfire_cnn_aspp",
    cnn_aspp_builder,
    defaults={
        "in_channels": 12,
        "base_channels": 32,
        "aspp_channels": 32,
        "dilations": (1, 3, 6, 12),
        "dropout": 0.0,
    },
)


register_model(
    name="hydrographnet",
    builder=hydrographnet_builder,
    defaults={
        "hidden_dim": 64,
        "harmonics": 5,
        "num_gn_blocks": 5,
    },
)
