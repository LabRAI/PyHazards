from .backbones import CNNPatchEncoder, MLPBackbone, TemporalEncoder
from .asufm import ASUFM, asufm_builder
from .builder import build_model, default_builder
from .cnn_aspp import WildfireCNNASPP, cnn_aspp_builder
from .eqnet import EQNet, eqnet_builder
from .eqtransformer import EQTransformer, eqtransformer_builder
from .firecastnet import FireCastNet, firecastnet_builder
from .firepred import FirePred, firepred_builder
from .floodcast import FloodCast, floodcast_builder
from .forefire import ForeFireAdapter, forefire_builder
from .fourcastnet_tc import FourCastNetTC, fourcastnet_tc_builder
from .gpd import GPD, gpd_builder
from .google_flood_forecasting import GoogleFloodForecasting, google_flood_forecasting_builder
from .graphcast_tc import GraphCastTC, graphcast_tc_builder
from .heads import ClassificationHead, RegressionHead, SegmentationHead
from .hurricast import Hurricast, hurricast_builder
from .hydrographnet import HydroGraphNet, HydroGraphNetLoss, hydrographnet_builder
from .neuralhydrology_ealstm import NeuralHydrologyEALSTM, neuralhydrology_ealstm_builder
from .neuralhydrology_lstm import NeuralHydrologyLSTM, neuralhydrology_lstm_builder
from .pangu_tc import PanguTC, pangu_tc_builder
from .phasenet import PhaseNet, phasenet_builder
from .registry import available_models, register_model
from .saf_net import SAFNet, saf_net_builder
from .tcif_fusion import TCIFFusion, tcif_fusion_builder
from .tropicalcyclone_mlp import TropicalCycloneMLP, tropicalcyclone_mlp_builder
from .tropicyclonenet import TropiCycloneNet, tropicyclonenet_builder
from .urbanfloodcast import UrbanFloodCast, urbanfloodcast_builder
from .wavecastnet import (
    ConvLEMCell,
    WaveCastNet,
    WaveCastNetLoss,
    WavefieldMetrics,
    wavecastnet_builder,
)
from .wildfire_forecasting import WildfireForecasting, wildfire_forecasting_builder
from .wildfire_aspp import TverskyLoss, WildfireASPP, wildfire_aspp_builder
from .wildfire_fpa import WildfireFPA, wildfire_fpa_builder
from .wildfire_mamba import WildfireMamba, wildfire_mamba_builder
from .lightgbm import LightGBMModel, lightgbm_builder
from .logistic_regression import LogisticRegressionModel, logistic_regression_builder
from .random_forest import RandomForestModel, random_forest_builder
from .xgboost import XGBoostModel, xgboost_builder
from .unet import TinyUNet, unet_builder
from .resnet18_unet import TinyResNet18UNet, resnet18_unet_builder
from .attention_unet import TinyAttentionUNet, attention_unet_builder
from .deeplabv3p import TinyDeepLabV3P, deeplabv3p_builder
from .convlstm import TinyConvLSTM, convlstm_builder
from .mau import TinyMAU, mau_builder
from .predrnn_v2 import TinyPredRNNv2, predrnn_v2_builder
from .rainformer import TinyRainformer, rainformer_builder
from .earthformer import TinyEarthFormer, earthformer_builder
from .swinlstm import TinySwinLSTM, swinlstm_builder
from .earthfarseer import TinyEarthFarseer, earthfarseer_builder
from .convgru_trajgru import TinyConvGRTrajGRU, convgru_trajgru_builder
from .tcn import TinyTCN, tcn_builder
from .utae import TinyUTAE, utae_builder
from .segformer import TinySegFormer, segformer_builder
from .swin_unet import TinySwinUNet, swin_unet_builder
from .vit_segmenter import TinyViTSegmenter, vit_segmenter_builder
from .deep_ensemble import DeepEnsemble, deep_ensemble_builder
from .ts_satfire import TSSatFire, ts_satfire_builder
from .wildfirespreadts import WildfireSpreadTS, wildfirespreadts_builder
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
    "ClassificationHead",
    "RegressionHead",
    "SegmentationHead",
    "EQNet",
    "eqnet_builder",
    "EQTransformer",
    "eqtransformer_builder",
    "FireCastNet",
    "firecastnet_builder",
    "FirePred",
    "firepred_builder",
    "LightGBMModel",
    "lightgbm_builder",
    "LogisticRegressionModel",
    "logistic_regression_builder",
    "RandomForestModel",
    "random_forest_builder",
    "XGBoostModel",
    "xgboost_builder",
    "TinyUNet",
    "unet_builder",
    "TinyResNet18UNet",
    "resnet18_unet_builder",
    "TinyAttentionUNet",
    "attention_unet_builder",
    "TinyDeepLabV3P",
    "deeplabv3p_builder",
    "TinyConvLSTM",
    "convlstm_builder",
    "TinyMAU",
    "mau_builder",
    "TinyPredRNNv2",
    "predrnn_v2_builder",
    "TinyRainformer",
    "rainformer_builder",
    "TinyEarthFormer",
    "earthformer_builder",
    "TinySwinLSTM",
    "swinlstm_builder",
    "TinyEarthFarseer",
    "earthfarseer_builder",
    "TinyConvGRTrajGRU",
    "convgru_trajgru_builder",
    "TinyTCN",
    "tcn_builder",
    "TinyUTAE",
    "utae_builder",
    "TinySegFormer",
    "segformer_builder",
    "TinySwinUNet",
    "swin_unet_builder",
    "TinyViTSegmenter",
    "vit_segmenter_builder",
    "DeepEnsemble",
    "deep_ensemble_builder",
    "TSSatFire",
    "ts_satfire_builder",
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
    "WildfireForecasting",
    "wildfire_forecasting_builder",
    "WildfireFPA",
    "wildfire_fpa_builder",
    "WildfireMamba",
    "wildfire_mamba_builder",
    "WildfireSpreadTS",
    "wildfirespreadts_builder",
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
    defaults={
        "input_dim": 7,
        "hidden_dim": 64,
        "output_dim": 5,
        "lookback": 12,
        "num_layers": 2,
        "dropout": 0.1,
    },
)

register_model(
    "asufm",
    asufm_builder,
    defaults={
        "input_dim": 7,
        "hidden_dim": 64,
        "output_dim": 5,
        "lookback": 12,
        "dropout": 0.1,
    },
)

register_model(
    "wildfirespreadts",
    wildfirespreadts_builder,
    defaults={
        "history": 4,
        "in_channels": 6,
        "hidden_dim": 32,
        "out_channels": 1,
        "dropout": 0.1,
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
    "logistic_regression",
    logistic_regression_builder,
    defaults={
        "max_iter": 500,
    },
)

register_model(
    "random_forest",
    random_forest_builder,
    defaults={
        "n_estimators": 500,
        "max_depth": None,
    },
)

register_model(
    "xgboost",
    xgboost_builder,
    defaults={
        "max_depth": 8,
        "eta": 0.05,
        "num_boost_round": 800,
    },
)

register_model(
    "lightgbm",
    lightgbm_builder,
    defaults={
        "num_leaves": 63,
        "learning_rate": 0.05,
        "num_boost_round": 800,
    },
)

register_model(
    "unet",
    unet_builder,
    defaults={"in_channels": 1, "base_channels": 16, "out_dim": 1},
)

register_model(
    "resnet18_unet",
    resnet18_unet_builder,
    defaults={"in_channels": 1, "base_channels": 16, "out_dim": 1},
)

register_model(
    "attention_unet",
    attention_unet_builder,
    defaults={"in_channels": 1, "base_channels": 16, "out_dim": 1},
)

register_model(
    "deeplabv3p",
    deeplabv3p_builder,
    defaults={"in_channels": 1, "hidden_dim": 32, "out_dim": 1},
)

register_model(
    "convlstm",
    convlstm_builder,
    defaults={"history": 4, "in_channels": 1, "hidden_dim": 24, "out_dim": 1},
)

register_model(
    "mau",
    mau_builder,
    defaults={"history": 4, "in_channels": 1, "hidden_dim": 24, "out_dim": 1},
)

register_model(
    "predrnn_v2",
    predrnn_v2_builder,
    defaults={"history": 4, "in_channels": 1, "hidden_dim": 24, "out_dim": 1},
)

register_model(
    "rainformer",
    rainformer_builder,
    defaults={"history": 4, "in_channels": 1, "hidden_dim": 24, "out_dim": 1},
)

register_model(
    "earthformer",
    earthformer_builder,
    defaults={"history": 4, "in_channels": 1, "hidden_dim": 32, "out_dim": 1},
)

register_model(
    "swinlstm",
    swinlstm_builder,
    defaults={"history": 4, "in_channels": 1, "hidden_dim": 24, "out_dim": 1},
)

register_model(
    "earthfarseer",
    earthfarseer_builder,
    defaults={"history": 4, "in_channels": 1, "hidden_dim": 24, "out_dim": 1},
)

register_model(
    "convgru_trajgru",
    convgru_trajgru_builder,
    defaults={"history": 4, "in_channels": 1, "hidden_dim": 24, "out_dim": 1},
)

register_model(
    "tcn",
    tcn_builder,
    defaults={"history": 4, "in_channels": 1, "hidden_dim": 24, "out_dim": 1},
)

register_model(
    "utae",
    utae_builder,
    defaults={"history": 4, "in_channels": 1, "hidden_dim": 24, "out_dim": 1},
)

register_model(
    "segformer",
    segformer_builder,
    defaults={"history": 4, "in_channels": 1, "hidden_dim": 32, "out_dim": 1},
)

register_model(
    "swin_unet",
    swin_unet_builder,
    defaults={"history": 4, "in_channels": 1, "base_channels": 16, "out_dim": 1},
)

register_model(
    "vit_segmenter",
    vit_segmenter_builder,
    defaults={"history": 4, "in_channels": 1, "hidden_dim": 32, "out_dim": 1},
)

register_model(
    "deep_ensemble",
    deep_ensemble_builder,
    defaults={"in_channels": 1, "base_channels": 16, "out_dim": 1, "ensemble_size": 3},
)

register_model(
    "firepred",
    firepred_builder,
    defaults={"history": 5, "in_channels": 8, "hidden_dim": 32, "out_channels": 1, "dropout": 0.1},
)

register_model(
    "ts_satfire",
    ts_satfire_builder,
    defaults={"history": 5, "in_channels": 8, "hidden_dim": 32, "out_channels": 1, "dropout": 0.1},
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
