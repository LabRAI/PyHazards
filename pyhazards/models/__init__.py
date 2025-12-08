from .backbones import CNNPatchEncoder, MLPBackbone, TemporalEncoder
from .builder import build_model, default_builder
from .heads import ClassificationHead, RegressionHead, SegmentationHead
from .registry import available_models, register_model
from .wildfire_mamba import WildfireMamba, wildfire_mamba_builder

__all__ = [
    "build_model",
    "available_models",
    "register_model",
    "MLPBackbone",
    "CNNPatchEncoder",
    "TemporalEncoder",
    "ClassificationHead",
    "RegressionHead",
    "SegmentationHead",
    "WildfireMamba",
    "wildfire_mamba_builder",
]

# Register default backbones
register_model("mlp", default_builder, defaults={"hidden_dim": 256, "depth": 2})
register_model("cnn", default_builder, defaults={"hidden_dim": 64, "in_channels": 3})
register_model("temporal", default_builder, defaults={"hidden_dim": 128, "num_layers": 1})
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
