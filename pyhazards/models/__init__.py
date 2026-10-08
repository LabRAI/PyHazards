from .backbones import CNNPatchEncoder, MLPBackbone, TemporalEncoder
from .asufm import ASUFM, asufm_builder
from .attention_unet import AttentionUnet, TemporalAttentionUnet, attention_unet_builder
from .builder import build_model, default_builder
from .classical import EstimatorModule, kondylatos_instance_features, random_forest_builder, xgboost_builder
from .cnn_aspp import WildfireCNNASPP, cnn_aspp_builder
from .earthfarseer import Earthfarseer, EarthfarseerSegmenter, earthfarseer_builder
from .convgru import ConvGRU, ConvGRUCell, ConvGRUSegmenter, convgru_builder
from .convlstm import ConvLSTM, ConvLSTMCell, ConvLSTMSegmenter, convlstm_builder
from .earthformer import CuboidTransformerModel, Earthformer, EarthformerSegmenter, earthformer_builder
from .deeplabv3 import DeepLabV3, deeplabv3_builder
from .deep_ensemble import DeepEnsemble, deep_ensemble_builder
from .eqnet import EQNet, EQNetLoss, eqnet_builder, eqnet_candidate_grid, eqnet_travel_times, shift_and_stack
from .eqtransformer import EQTransformer, eqtransformer_builder
from .firecastnet import FireCastNet, firecastnet_builder
from .floodcast import FloodCast, floodcast_builder
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
from .prithvi import PrithviMAE, PrithviSegmentation, PrithviViT
from .prithvi_burnscars import prithvi_burnscars_builder
from .prithvi_eo_2_tl import prithvi_eo_2_tl_builder
from .rainformer import Rainformer, RainformerSegmenter, rainformer_builder
from .registry import available_models, register_model
from .resnet_unet import ResNetUNet, resnet18_unet_builder
from .saf_net import SAFNet, saf_net_builder
from .segformer import SegFormer, segformer_builder
from .swin_unet import SwinUnet, swin_unet_builder
from .swin_unetr import SwinUNETR, TemporalSwinUNETR, swin_unetr_builder
from .swinlstm import SwinLSTM, SwinLSTMSegmenter, swinlstm_builder
from .tcif_fusion import TCIFFusion, tcif_fusion_builder
from .tcn import TCN, TemporalConvNet, tcn_builder
from .trajgru import TrajGRU, TrajGRUSegmenter, trajgru_builder
from .tropicalcyclone_mlp import TropicalCycloneMLP, load_keras_weights, tropicalcyclone_mlp_builder
from .tropicyclonenet import (
    TrajectoryDiscriminator,
    TropiCycloneNet,
    load_tropicyclonenet_checkpoint,
    tropicyclonenet_builder,
)
from .unet import UNet, unet_builder
from .ts_satfire import TS_SATFIRE_BASELINES, ts_satfire_builder
from .unet3d import TemporalUNet, UNet, unet3d_builder
from .unetr import UNETR, TemporalUNETR, unetr_builder
from .urbanfloodcast import UrbanFloodCast, urbanfloodcast_builder
from .utae import UTAE, utae_builder
from .wavecastnet import (
    ConvLEMCell,
    WaveCastNet,
    WaveCastNetLoss,
    WaveCastNetSparse,
    WavefieldMetrics,
    wavecastnet_builder,
    wavecastnet_station_coords,
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
    "DeepEnsemble",
    "deep_ensemble_builder",
    "EstimatorModule",
    "kondylatos_instance_features",
    "random_forest_builder",
    "xgboost_builder",
    "RegressionHead",
    "SegmentationHead",
    "EQNet",
    "EQNetLoss",
    "eqnet_builder",
    "eqnet_candidate_grid",
    "eqnet_travel_times",
    "shift_and_stack",
    "EQTransformer",
    "eqtransformer_builder",
    "FireCastNet",
    "firecastnet_builder",
    "FloodCast",
    "floodcast_builder",
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
    "PrithviMAE",
    "PrithviSegmentation",
    "PrithviViT",
    "prithvi_burnscars_builder",
    "prithvi_eo_2_tl_builder",
    "SAFNet",
    "saf_net_builder",
    "TCIFFusion",
    "tcif_fusion_builder",
    "TCN",
    "TemporalConvNet",
    "tcn_builder",
    "TropicalCycloneMLP",
    "load_keras_weights",
    "tropicalcyclone_mlp_builder",
    "TrajectoryDiscriminator",
    "TropiCycloneNet",
    "load_tropicyclonenet_checkpoint",
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
    "CuboidTransformerModel",
    "Earthformer",
    "EarthformerSegmenter",
    "earthformer_builder",
    "ConvGRU",
    "ConvGRUCell",
    "ConvGRUSegmenter",
    "convgru_builder",
    "PixelLogisticRegression",
    "logistic_regression_builder",
    "ResNetUNet",
    "resnet18_unet_builder",
    "UNet",
    "unet_builder",
    "DeepLabV3",
    "deeplabv3_builder",
    "SegFormer",
    "segformer_builder",
    "UTAE",
    "utae_builder",
    "SwinUnet",
    "swin_unet_builder",
    "Earthfarseer",
    "EarthfarseerSegmenter",
    "earthfarseer_builder",
    "Rainformer",
    "RainformerSegmenter",
    "rainformer_builder",
    "SwinUNETR",
    "TemporalSwinUNETR",
    "swin_unetr_builder",
    "TS_SATFIRE_BASELINES",
    "ts_satfire_builder",
    "UNet",
    "TemporalUNet",
    "unet3d_builder",
    "UNETR",
    "TemporalUNETR",
    "unetr_builder",
    "SwinLSTM",
    "SwinLSTMSegmenter",
    "swinlstm_builder",
    "TrajGRU",
    "TrajGRUSegmenter",
    "trajgru_builder",
    "ConvLEMCell",
    "WaveCastNet",
    "WaveCastNetLoss",
    "WaveCastNetSparse",
    "WavefieldMetrics",
    "wavecastnet_builder",
    "wavecastnet_station_coords",
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

# Kondylatos et al. (2022) notebooks/RF.ipynb; the daily-tensor layout is a builder default.
register_model(
    "random_forest",
    random_forest_builder,
    defaults={
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "random_state": 123,
    },
)

# Library defaults of xgboost.XGBClassifier (the paper's values are in its unread Supporting Information).
register_model("xgboost", xgboost_builder, defaults={})

register_model(
    "deep_ensemble",
    deep_ensemble_builder,
    defaults={"num_members": 5, "regression_output": "gaussian", "mc_dropout_passes": 0},
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
    "earthfarseer",
    earthfarseer_builder,
    defaults={"in_channels": 1, "history": 10, "img_size": 64, "out_channels": 1},
)

register_model(
    "rainformer",
    rainformer_builder,
    defaults={"in_channels": 1, "history": 9, "img_size": 288, "out_channels": 1},
)

register_model(
    "swinlstm",
    swinlstm_builder,
    defaults={"variant": "d", "in_channels": 1, "img_size": 64, "num_output_frames": 10},
)

register_model(
    "trajgru",
    trajgru_builder,
    defaults={"config": "hko7", "in_channels": 1, "out_channels": 1, "layer_type": "TrajGRU"},
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
    "unet",
    unet_builder,
    defaults={"in_channels": 1, "out_channels": 2, "padding": "valid", "dropout": 0.5, "variant": "paper"},
)

register_model(
    "deeplabv3",
    deeplabv3_builder,
    # Shadrin et al. (2024): smp DeepLabV3, ResNet-18 encoder with three stages, 58 channels for day 3.
    defaults={"in_channels": 58, "out_channels": 1, "encoder_name": "resnet18", "encoder_depth": 3, "encoder_weights": None},
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
    "convgru",
    convgru_builder,
    defaults={"in_channels": 11, "out_channels": 1, "hidden_dim": 128, "kernel_size": 5, "num_layers": 1},
)

register_model(
    "utae",
    utae_builder,
    defaults={"out_channels": 1},
)

register_model(
    "earthformer",
    earthformer_builder,
    # Shapes and hyperparameters come from the preset (official earthformer_sevir_v1.yaml).
    defaults={"config": "sevir", "pretrained": None},
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
    "unet3d",
    unet3d_builder,
    defaults={
        "out_channels": 2,
        "spatial_dims": 3,
        "channels": (64, 128, 256, 512, 1024),
        "strides": None,
        "num_res_units": 0,
        "dropout": 0.0,
        "time_reduction": "mean",
    },
)

register_model(
    "unetr",
    unetr_builder,
    defaults={
        "out_channels": 2,
        "spatial_dims": 3,
        "history": 6,
        "image_size": 256,
        "feature_size": 16,
        "hidden_size": 384,
        "mlp_dim": 1536,
        "num_heads": 12,
        "norm_name": "batch",
        "time_reduction": "mean",
    },
)

register_model(
    "swin_unetr",
    swin_unetr_builder,
    defaults={
        "out_channels": 2,
        "spatial_dims": 3,
        "history": 6,
        "image_size": 256,
        "norm_name": "batch",
        "attn_version": "v1",
        "time_reduction": "mean",
    },
)

register_model(
    "ts_satfire",
    ts_satfire_builder,
    defaults={
        "baseline": "swinunetr3d",
        "in_channels": 43,
        "history": 6,
        "image_size": 256,
        "out_channels": 2,
        "num_heads": 3,
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
    "prithvi_burnscars",
    prithvi_burnscars_builder,
    defaults={"in_channels": 6, "num_classes": 2, "pretrained": False},
)

register_model(
    "prithvi_eo_2_tl",
    prithvi_eo_2_tl_builder,
    defaults={"variant": "300m", "in_channels": 6, "num_classes": 2, "num_frames": 1, "pretrained": False},
)

register_model(
    "firecastnet",
    firecastnet_builder,
    defaults={
        "in_channels": 11,
        "timeseries_len": 24,
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
        "n_dynamic": 5,
        "n_static": 27,
        "hidden_size": 256,
        "n_targets": 1,
        "output_dropout": 0.4,
        "initial_forget_bias": 5.0,
    },
)

register_model(
    "neuralhydrology_ealstm",
    neuralhydrology_ealstm_builder,
    defaults={
        "n_dynamic": 5,
        "n_static": 27,
        "hidden_size": 256,
        "n_targets": 1,
        "output_dropout": 0.4,
        "initial_forget_bias": 5.0,
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
    defaults={"config": "floodhub", "pretrained": False},
)

register_model(
    "phasenet",
    phasenet_builder,
    defaults={"in_channels": 3},
)

register_model(
    "eqtransformer",
    eqtransformer_builder,
    defaults={"in_channels": 3},
)

register_model(
    "gpd",
    gpd_builder,
    defaults={"in_channels": 3},
)

register_model(
    "eqnet",
    eqnet_builder,
    defaults={"in_channels": 3, "sampling_rate": 100.0},
)

register_model(
    "wavecastnet",
    wavecastnet_builder,
    defaults={
        "variant": "dense",
        "in_channels": 3,
        "height": 344,
        "width": 224,
        "future_seq": 30,
        "kernel_size": 3,
        "dt": 1.0,
        "activation": "tanh",
    },
)

register_model(
    "tropicalcyclone_mlp",
    tropicalcyclone_mlp_builder,
    defaults={
        "input_dim": 121,
        "hidden_dims": (2048, 2048),
        "activations": ("sigmoid", "relu"),
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
        "num_sample": 6,
        "official_sample_loop": False,
    },
)

register_model(
    "saf_net",
    saf_net_builder,
    defaults={
        "wide_dim": 96,
        "num_times": 4,
        "num_levels": 4,
        "grid_size": 31,
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
