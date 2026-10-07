from .backbones import CNNPatchEncoder, MLPBackbone, TemporalEncoder
from .asufm import ASUFM, asufm_builder
from .attention_unet import AttentionUnet, TemporalAttentionUnet, attention_unet_builder
from .builder import build_model, default_builder
from .cnn_aspp import WildfireCNNASPP, cnn_aspp_builder
from .convlstm import ConvLSTM, ConvLSTMCell, ConvLSTMSegmenter, convlstm_builder
from .eqnet import EQNet, eqnet_builder
from .eqtransformer import EQTransformer, eqtransformer_builder
from .firecastnet import FireCastNet, firecastnet_builder
from .floodcast import FloodCast, floodcast_builder
from .forefire import ForeFireAdapter, forefire_builder
from .fourcastnet_tc import FourCastNetTC, fourcastnet_tc_builder
from .gpd import GPD, gpd_builder
from .google_flood_forecasting import GoogleFloodForecasting, google_flood_forecasting_builder
from .graphcast_tc import GraphCastTC, graphcast_tc_builder
from .heads import ClassificationHead, RegressionHead, SegmentationHead
from .hurricast import Hurricast, hurricast_builder
from .hydrographnet import HydroGraphNet, HydroGraphNetLoss, hydrographnet_builder
from .logistic_regression import PixelLogisticRegression, logistic_regression_builder
from .neuralhydrology_ealstm import NeuralHydrologyEALSTM, neuralhydrology_ealstm_builder
from .neuralhydrology_lstm import NeuralHydrologyLSTM, neuralhydrology_lstm_builder
from .pangu_tc import PanguTC, pangu_tc_builder
from .phasenet import PhaseNet, phasenet_builder
from .registry import available_models, register_model
from .resnet_unet import ResNetUNet, resnet18_unet_builder
from .saf_net import SAFNet, saf_net_builder
from .segformer import SegFormer, segformer_builder
from .swin_unet import SwinUnet, swin_unet_builder
from .tcif_fusion import TCIFFusion, tcif_fusion_builder
from .tcn import TCN, TemporalConvNet, tcn_builder
from .tropicalcyclone_mlp import TropicalCycloneMLP, tropicalcyclone_mlp_builder
from .tropicyclonenet import TropiCycloneNet, tropicyclonenet_builder
from .urbanfloodcast import UrbanFloodCast, urbanfloodcast_builder
from .utae import UTAE, utae_builder
from .wavecastnet import (
    ConvLEMCell,
    WaveCastNet,
    WaveCastNetLoss,
    WavefieldMetrics,
    wavecastnet_builder,
)
from .wildfire_forecasting import (
    WILDFIRE_FORECASTING_VARIANTS,
    SimpleConvLSTM,
    SimpleLSTM,
    wildfire_forecasting_builder,
)
from .wildfire_aspp import TverskyLoss, WildfireASPP, wildfire_aspp_builder
from .wildfire_fpa import WildfireFPA, wildfire_fpa_builder
from .wildfire_mamba import WildfireMamba, wildfire_mamba_builder
from .wildfirespreadts import WILDFIRESPREADTS_BASELINES, wildfirespreadts_builder
from .wrf_sfire import WRFSFireAdapter, wrf_sfire_builder


__all__ = [
    "build_model",
    "available_models",
    "register_model",
    "MLPBackbone",
    "CNNPatchEncoder",
    "TemporalEncoder",
    "ASUFM",
    "asufm_builder",
    "AttentionUnet",
    "TemporalAttentionUnet",
    "attention_unet_builder",
    "ClassificationHead",
    "RegressionHead",
    "SegmentationHead",
    "EQNet",
    "eqnet_builder",
    "EQTransformer",
    "eqtransformer_builder",
    "FireCastNet",
    "firecastnet_builder",
    "FloodCast",
    "floodcast_builder",
    "ForeFireAdapter",
    "forefire_builder",
    "FourCastNetTC",
    "fourcastnet_tc_builder",
    "GPD",
    "gpd_builder",
    "GoogleFloodForecasting",
    "google_flood_forecasting_builder",
    "GraphCastTC",
    "graphcast_tc_builder",
    "Hurricast",
    "hurricast_builder",
    "HydroGraphNet",
    "HydroGraphNetLoss",
    "hydrographnet_builder",
    "NeuralHydrologyEALSTM",
    "neuralhydrology_ealstm_builder",
    "NeuralHydrologyLSTM",
    "neuralhydrology_lstm_builder",
    "PanguTC",
    "pangu_tc_builder",
    "PhaseNet",
    "phasenet_builder",
    "SAFNet",
    "saf_net_builder",
    "TCIFFusion",
    "tcif_fusion_builder",
    "TCN",
    "TemporalConvNet",
    "tcn_builder",
    "TropicalCycloneMLP",
    "tropicalcyclone_mlp_builder",
    "TropiCycloneNet",
    "tropicyclonenet_builder",
    "UrbanFloodCast",
    "urbanfloodcast_builder",
    "WildfireASPP",
    "TverskyLoss",
    "wildfire_aspp_builder",
    "WildfireCNNASPP",
    "cnn_aspp_builder",
    "SimpleLSTM",
    "SimpleConvLSTM",
    "WILDFIRE_FORECASTING_VARIANTS",
    "wildfire_forecasting_builder",
    "WildfireFPA",
    "wildfire_fpa_builder",
    "WildfireMamba",
    "wildfire_mamba_builder",
    "WILDFIRESPREADTS_BASELINES",
    "wildfirespreadts_builder",
    "ConvLSTM",
    "ConvLSTMCell",
    "ConvLSTMSegmenter",
    "convlstm_builder",
    "PixelLogisticRegression",
    "logistic_regression_builder",
    "ResNetUNet",
    "resnet18_unet_builder",
    "SegFormer",
    "segformer_builder",
    "UTAE",
    "utae_builder",
    "SwinUnet",
    "swin_unet_builder",
    "WRFSFireAdapter",
    "wrf_sfire_builder",
    "ConvLEMCell",
    "WaveCastNet",
    "WaveCastNetLoss",
    "WavefieldMetrics",
    "wavecastnet_builder",
]


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

register_model(
    "wildfire_fpa",
    wildfire_fpa_builder,
    defaults={
        "out_dim": 5,
        "output_dim": 5,
        "depth": 2,
        "hidden_dim": 64,
        "activation": "relu",
        "dropout": None,
        "latent_dim": 32,
        "num_layers": 1,
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
    defaults={"in_channels": 12},
)

register_model(
    "wildfire_forecasting",
    wildfire_forecasting_builder,
    # hidden_size is left to the builder: its paper value depends on the variant (64 / 32).
    defaults={
        "variant": "lstm",
        "input_dim": 25,
        "lstm_layers": 1,
        "dropout": 0.5,
        "patch_size": 25,
    },
)

register_model(
    "asufm",
    asufm_builder,
    defaults={"in_channels": 6, "out_channels": 1, "img_size": 64, "window_size": 8, "focal": True},
)

register_model(
    "swin_unet",
    swin_unet_builder,
    defaults={"out_channels": 1, "history": 1, "img_size": 224, "window_size": 7, "drop_path_rate": 0.2, "pretrained": None},
)

register_model(
    "wildfirespreadts",
    wildfirespreadts_builder,
    defaults={
        "baseline": "utae",
        "in_channels": 40,
        "history": 5,
    },
)

register_model(
    "logistic_regression",
    logistic_regression_builder,
    defaults={"out_channels": 1, "history": 1, "kernel_size": 3},
)

register_model(
    "resnet18_unet",
    resnet18_unet_builder,
    defaults={"out_channels": 1, "history": 1, "encoder_name": "resnet18", "encoder_weights": None},
)

register_model(
    "segformer",
    segformer_builder,
    defaults={"out_channels": 1, "history": 1, "variant": "b2", "encoder_weights": None, "upsample": True},
)

register_model(
    "convlstm",
    convlstm_builder,
    defaults={"out_channels": 1, "hidden_dim": 64, "kernel_size": 3, "num_layers": 1, "readout": "cell"},
)

register_model(
    "utae",
    utae_builder,
    defaults={"out_channels": 1},
)

register_model(
    "attention_unet",
    attention_unet_builder,
    defaults={
        "out_channels": 2,
        "spatial_dims": 3,
        "channels": (64, 128, 256, 512, 1024),
        "strides": None,
        "dropout": 0.0,
        "time_reduction": "mean",
    },
)

register_model(
    "tcn",
    tcn_builder,
    defaults={
        "input_dim": 2,
        "out_dim": 1,
        "hidden_dim": 30,
        "num_levels": 8,
        "kernel_size": 7,
        "dropout": 0.0,
        "readout": "last",
        "head_init": "normal",
    },
)

register_model(
    "forefire",
    forefire_builder,
    defaults={
        "in_channels": 12,
        "out_channels": 1,
        "diffusion_steps": 2,
    },
)

register_model(
    "wrf_sfire",
    wrf_sfire_builder,
    defaults={
        "in_channels": 12,
        "out_channels": 1,
        "diffusion_steps": 3,
    },
)

register_model(
    "firecastnet",
    firecastnet_builder,
    defaults={
        "in_channels": 12,
        "hidden_dim": 32,
        "out_channels": 1,
        "dropout": 0.1,
    },
)

register_model(
    "wildfire_cnn_aspp",
    cnn_aspp_builder,
    defaults={"in_channels": 12, "dilations": (1, 3, 6, 12)},
)

register_model(
    "hydrographnet",
    hydrographnet_builder,
    defaults={
        "hidden_dim": 64,
        "harmonics": 5,
        "num_gn_blocks": 5,
    },
)

register_model(
    "neuralhydrology_lstm",
    neuralhydrology_lstm_builder,
    defaults={
        "input_dim": 2,
        "hidden_dim": 64,
        "num_layers": 2,
        "out_dim": 1,
        "dropout": 0.1,
    },
)

register_model(
    "neuralhydrology_ealstm",
    neuralhydrology_ealstm_builder,
    defaults={
        "input_dim": 2,
        "hidden_dim": 64,
        "num_layers": 1,
        "out_dim": 1,
        "dropout": 0.1,
    },
)

register_model(
    "floodcast",
    floodcast_builder,
    defaults={
        "in_channels": 3,
        "history": 4,
        "hidden_dim": 32,
        "out_channels": 1,
        "dropout": 0.1,
    },
)

register_model(
    "urbanfloodcast",
    urbanfloodcast_builder,
    defaults={
        "in_channels": 3,
        "history": 4,
        "base_channels": 32,
        "out_channels": 1,
    },
)

register_model(
    "google_flood_forecasting",
    google_flood_forecasting_builder,
    defaults={
        "input_dim": 2,
        "hidden_dim": 64,
        "out_dim": 1,
        "history": 4,
        "dropout": 0.1,
    },
)

register_model(
    "phasenet",
    phasenet_builder,
    defaults={
        "in_channels": 3,
        "hidden_dim": 32,
    },
)

register_model(
    "eqtransformer",
    eqtransformer_builder,
    defaults={
        "in_channels": 3,
        "hidden_dim": 48,
        "num_layers": 2,
        "dropout": 0.1,
    },
)

register_model(
    "gpd",
    gpd_builder,
    defaults={
        "in_channels": 3,
        "hidden_dim": 32,
        "dropout": 0.1,
    },
)

register_model(
    "eqnet",
    eqnet_builder,
    defaults={
        "in_channels": 3,
        "hidden_dim": 48,
        "num_heads": 4,
        "num_layers": 2,
        "dropout": 0.1,
    },
)

register_model(
    "wavecastnet",
    wavecastnet_builder,
    defaults={
        "hidden_dim": 144,
        "num_layers": 2,
        "kernel_size": 3,
        "dt": 1.0,
        "activation": "tanh",
        "dropout": 0.1,
    },
)

register_model(
    "tropicalcyclone_mlp",
    tropicalcyclone_mlp_builder,
    defaults={
        "input_dim": 8,
        "history": 6,
        "hidden_dim": 64,
        "horizon": 5,
        "output_dim": 3,
        "dropout": 0.1,
    },
)

register_model(
    "hurricast",
    hurricast_builder,
    defaults={
        "input_dim": 8,
        "hidden_dim": 64,
        "num_layers": 2,
        "horizon": 5,
        "output_dim": 3,
        "dropout": 0.1,
    },
)

register_model(
    "tropicyclonenet",
    tropicyclonenet_builder,
    defaults={
        "input_dim": 8,
        "hidden_dim": 64,
        "horizon": 5,
        "output_dim": 3,
        "num_layers": 2,
        "dropout": 0.1,
    },
)

register_model(
    "saf_net",
    saf_net_builder,
    defaults={
        "input_dim": 8,
        "hidden_dim": 64,
        "horizon": 5,
        "dropout": 0.1,
    },
)

register_model(
    "tcif_fusion",
    tcif_fusion_builder,
    defaults={
        "input_dim": 8,
        "hidden_dim": 64,
        "horizon": 5,
        "output_dim": 3,
        "dropout": 0.1,
    },
)

register_model(
    "graphcast_tc",
    graphcast_tc_builder,
    defaults={
        "input_dim": 8,
        "hidden_dim": 96,
        "horizon": 5,
        "output_dim": 3,
        "num_layers": 2,
        "num_heads": 4,
        "dropout": 0.1,
    },
)

register_model(
    "pangu_tc",
    pangu_tc_builder,
    defaults={
        "input_dim": 8,
        "hidden_dim": 96,
        "horizon": 5,
        "output_dim": 3,
        "dropout": 0.1,
    },
)

register_model(
    "fourcastnet_tc",
    fourcastnet_tc_builder,
    defaults={
        "input_dim": 8,
        "history": 6,
        "hidden_dim": 96,
        "horizon": 5,
        "output_dim": 3,
        "dropout": 0.1,
    },
)
