from .backbones import CNNPatchEncoder, MLPBackbone, TemporalEncoder
from .builder import build_model, default_builder
from .heads import ClassificationHead, RegressionHead, SegmentationHead
from .registry import available_models, register_model

# Wildfire models
from .wildfire_mamba import WildfireMamba, wildfire_mamba_builder
from .wildfire_aspp import WildfireASPP, TverskyLoss, wildfire_aspp_builder
from .cnn_aspp import WildfireCNNASPP, cnn_aspp_builder

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
    "WildfireMamba",
    "wildfire_mamba_builder",
    "WildfireASPP",
    "TverskyLoss",
    "wildfire_aspp_builder",
    "WildfireCNNASPP",
    "cnn_aspp_builder",
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
