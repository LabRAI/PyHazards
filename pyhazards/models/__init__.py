from .backbones import CNNPatchEncoder, MLPBackbone, TemporalEncoder
from .asufm import ASUFM, asufm_builder
from .builder import build_model, default_builder
from .cnn_aspp import WildfireCNNASPP, cnn_aspp_builder
from .eqnet import EQNet, eqnet_builder
from .eqtransformer import EQTransformer, eqtransformer_builder
from .firecastnet import FireCastNet, firecastnet_builder
from .firemm_ir import FireMMIR, firemm_ir_builder
from .firepred import FirePred, firepred_builder
from .gemini_25_pro_wildfire_prompted import (
    Gemini25ProWildfirePrompted,
    gemini_25_pro_wildfire_prompted_builder,
)
from .internvl3_wildfire_prompted import (
    InternVL3WildfirePrompted,
    internvl3_wildfire_prompted_builder,
)
from .llama4_wildfire_prompted import (
    Llama4WildfirePrompted,
    llama4_wildfire_prompted_builder,
)
from .modis_active_fire_c61 import MODISActiveFireC61, modis_active_fire_c61_builder
from .prithvi_burnscars import PrithviBurnScars, prithvi_burnscars_builder
from .prithvi_eo_2_tl import PrithviEO2TL, prithvi_eo_2_tl_builder
from .prithvi_wxc import PrithviWxC, prithvi_wxc_builder
from .qwen25_vl_wildfire_prompted import (
    Qwen25VLWildfirePrompted,
    qwen25_vl_wildfire_prompted_builder,
)
from .ts_satfire import TSSatFire, ts_satfire_builder
from .viirs_375m_active_fire import VIIRS375mActiveFire, viirs_375m_active_fire_builder
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
from .wildfire_aspp import TverskyLoss, WildfireASPP, wildfire_aspp_builder
from .wildfire_fpa import WildfireFPA, wildfire_fpa_builder
from .wildfire_mamba import WildfireMamba, wildfire_mamba_builder
from .wildfiregpt import WildfireGPTReasoner, wildfiregpt_builder
from .logistic_regression import LogisticRegressionModel, logistic_regression_builder
from .random_forest import RandomForestModel, random_forest_builder
from .xgboost import XGBoostModel, xgboost_builder
from .lightgbm import LightGBMModel, lightgbm_builder
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
    "FireMMIR",
    "firemm_ir_builder",
    "FirePred",
    "firepred_builder",
    "Gemini25ProWildfirePrompted",
    "gemini_25_pro_wildfire_prompted_builder",
    "InternVL3WildfirePrompted",
    "internvl3_wildfire_prompted_builder",
    "Llama4WildfirePrompted",
    "llama4_wildfire_prompted_builder",
    "MODISActiveFireC61",
    "modis_active_fire_c61_builder",
    "PrithviBurnScars",
    "prithvi_burnscars_builder",
    "PrithviEO2TL",
    "prithvi_eo_2_tl_builder",
    "PrithviWxC",
    "prithvi_wxc_builder",
    "Qwen25VLWildfirePrompted",
    "qwen25_vl_wildfire_prompted_builder",
    "TSSatFire",
    "ts_satfire_builder",
    "VIIRS375mActiveFire",
    "viirs_375m_active_fire_builder",
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
    "WildfireFPA",
    "wildfire_fpa_builder",
    "WildfireMamba",
    "wildfire_mamba_builder",
    "WildfireGPTReasoner",
    "wildfiregpt_builder",
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
        "in_dim": 8,
        "out_dim": 1,
        "input_dim": 7,
        "output_dim": 1,
        "depth": 2,
        "hidden_dim": 64,
        "activation": "relu",
        "dropout": 0.1,
        "latent_dim": 32,
        "num_layers": 2,
        "lookback": 12,
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
    "asufm",
    asufm_builder,
    defaults={
        "image_size": 64,
        "patch_size": 4,
        "in_channels": 6,
        "out_dim": 1,
        "embed_dim": 96,
        "depths": (2, 2, 2, 2),
        "num_heads": (3, 6, 12, 24),
        "window_size": 8,
        "mlp_ratio": 4.0,
        "dropout": 0.0,
        "drop_path_rate": 0.1,
        "focal_window": 3,
        "focal_level": 2,
        "use_focal_modulation": True,
        "spatial_attention": True,
        "skip_num": 3,
        "use_checkpoint": True,
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
    "firepred",
    firepred_builder,
    defaults={
        "history": 5,
        "in_channels": 8,
        "hidden_dim": 32,
        "out_channels": 1,
        "dropout": 0.1,
    },
)

register_model(
    "modis_active_fire_c61",
    modis_active_fire_c61_builder,
    defaults={
        "in_channels": 5,
        "hidden_dim": 24,
        "out_dim": 1,
        "context_kernel": 9,
        "dropout": 0.1,
    },
)

register_model(
    "prithvi_eo_2_tl",
    prithvi_eo_2_tl_builder,
    defaults={
        "image_size": 32,
        "in_channels": 6,
        "out_dim": 1,
        "patch_size": 4,
        "embed_dim": 128,
        "depth": 4,
        "num_heads": 4,
        "mlp_ratio": 4.0,
        "dropout": 0.1,
        "time_dim": 1,
        "location_dim": 2,
        "decoder_channels": 64,
    },
)

register_model(
    "prithvi_burnscars",
    prithvi_burnscars_builder,
    defaults={
        "image_size": 32,
        "in_channels": 6,
        "out_dim": 1,
        "patch_size": 4,
        "embed_dim": 128,
        "depth": 4,
        "num_heads": 4,
        "mlp_ratio": 4.0,
        "dropout": 0.1,
        "time_dim": 1,
        "location_dim": 2,
        "decoder_channels": 64,
    },
)

register_model(
    "prithvi_wxc",
    prithvi_wxc_builder,
    defaults={
        "image_size": 32,
        "in_channels": 8,
        "out_dim": 1,
        "patch_size": 4,
        "embed_dim": 128,
        "depth": 4,
        "num_heads": 4,
        "mlp_ratio": 4.0,
        "dropout": 0.1,
        "lead_time_dim": 1,
        "variable_summary_dim": 8,
        "decoder_channels": 64,
    },
)

register_model(
    "gemini_25_pro_wildfire_prompted",
    gemini_25_pro_wildfire_prompted_builder,
    defaults={
        "in_channels": 6,
        "out_dim": 1,
        "hidden_dim": 96,
        "prompt_dim": 32,
        "num_prompt_tokens": 6,
        "num_heads": 8,
        "dropout": 0.1,
    },
)

register_model(
    "internvl3_wildfire_prompted",
    internvl3_wildfire_prompted_builder,
    defaults={
        "in_channels": 6,
        "out_dim": 1,
        "hidden_dim": 96,
        "prompt_dim": 32,
        "num_prompt_tokens": 5,
        "num_heads": 6,
        "dropout": 0.1,
    },
)

register_model(
    "llama4_wildfire_prompted",
    llama4_wildfire_prompted_builder,
    defaults={
        "in_channels": 6,
        "out_dim": 1,
        "hidden_dim": 80,
        "prompt_dim": 32,
        "num_prompt_tokens": 4,
        "num_heads": 8,
        "dropout": 0.1,
    },
)

register_model(
    "qwen25_vl_wildfire_prompted",
    qwen25_vl_wildfire_prompted_builder,
    defaults={
        "in_channels": 6,
        "out_dim": 1,
        "hidden_dim": 64,
        "prompt_dim": 24,
        "num_prompt_tokens": 4,
        "num_heads": 4,
        "dropout": 0.1,
    },
)

register_model(
    "ts_satfire",
    ts_satfire_builder,
    defaults={
        "history": 5,
        "in_channels": 8,
        "hidden_dim": 32,
        "out_channels": 1,
        "dropout": 0.1,
    },
)

register_model(
    "viirs_375m_active_fire",
    viirs_375m_active_fire_builder,
    defaults={
        "in_channels": 5,
        "hidden_dim": 24,
        "out_dim": 1,
        "context_kernel": 7,
        "dropout": 0.1,
    },
)

register_model(
    "wildfiregpt",
    wildfiregpt_builder,
    defaults={
        "in_channels": 12,
        "out_dim": 1,
        "base_channels": 32,
        "hidden_dim": 64,
        "profile_dim": 8,
        "retrieved_dim": 16,
        "num_heads": 4,
        "dropout": 0.1,
    },
)

register_model(
    "firemm_ir",
    firemm_ir_builder,
    defaults={
        "in_channels": 6,
        "out_dim": 1,
        "hidden_dim": 64,
        "instruction_dim": 16,
        "num_memory_slots": 3,
        "num_heads": 4,
        "dropout": 0.1,
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


__all__.extend([
    "LogisticRegressionModel", "logistic_regression_builder",
    "RandomForestModel", "random_forest_builder",
    "XGBoostModel", "xgboost_builder",
    "LightGBMModel", "lightgbm_builder",
    "TinyUNet", "unet_builder",
    "TinyResNet18UNet", "resnet18_unet_builder",
    "TinyAttentionUNet", "attention_unet_builder",
    "TinyDeepLabV3P", "deeplabv3p_builder",
    "TinyConvLSTM", "convlstm_builder",
    "TinyMAU", "mau_builder",
    "TinyPredRNNv2", "predrnn_v2_builder",
    "TinyRainformer", "rainformer_builder",
    "TinyEarthFormer", "earthformer_builder",
    "TinySwinLSTM", "swinlstm_builder",
    "TinyEarthFarseer", "earthfarseer_builder",
    "TinyConvGRTrajGRU", "convgru_trajgru_builder",
    "TinyTCN", "tcn_builder",
    "TinyUTAE", "utae_builder",
    "TinySegFormer", "segformer_builder",
    "TinySwinUNet", "swin_unet_builder",
    "TinyViTSegmenter", "vit_segmenter_builder",
    "DeepEnsemble", "deep_ensemble_builder",
])


register_model(
    "logistic_regression",
    logistic_regression_builder,
    defaults={
        "solver": "lbfgs",
        "max_iter": 500,
        "class_weight": "balanced",
    },
)

register_model(
    "random_forest",
    random_forest_builder,
    defaults={
        "n_estimators": 500,
        "max_depth": None,
        "class_weight": "balanced_subsample",
    },
)

register_model(
    "xgboost",
    xgboost_builder,
    defaults={
        "max_depth": 8,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "num_boost_round": 800,
    },
)

register_model(
    "lightgbm",
    lightgbm_builder,
    defaults={
        "num_leaves": 63,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "num_boost_round": 800,
    },
)

register_model(
    "unet",
    unet_builder,
    defaults={
        "in_channels": 1,
        "out_dim": 1,
        "base_channels": 16,
    },
)

register_model(
    "resnet18_unet",
    resnet18_unet_builder,
    defaults={
        "in_channels": 1,
        "out_dim": 1,
        "stem_channels": 16,
    },
)

register_model(
    "attention_unet",
    attention_unet_builder,
    defaults={
        "in_channels": 1,
        "out_dim": 1,
        "base_channels": 8,
    },
)

register_model(
    "deeplabv3p",
    deeplabv3p_builder,
    defaults={
        "in_channels": 1,
        "out_dim": 1,
        "base_channels": 16,
    },
)

register_model(
    "convlstm",
    convlstm_builder,
    defaults={
        "in_channels": 1,
        "out_dim": 1,
        "enc_channels": 16,
        "hidden_channels": 16,
        "num_layers": 2,
        "kernel_size": 3,
    },
)

register_model(
    "mau",
    mau_builder,
    defaults={
        "in_channels": 1,
        "out_dim": 1,
        "hidden_channels": 12,
    },
)

register_model(
    "predrnn_v2",
    predrnn_v2_builder,
    defaults={
        "in_channels": 1,
        "out_dim": 1,
        "hidden_channels": 12,
    },
)

register_model(
    "rainformer",
    rainformer_builder,
    defaults={
        "in_channels": 1,
        "out_dim": 1,
        "hidden_channels": 16,
        "num_heads": 4,
        "num_layers": 2,
    },
)

register_model(
    "earthformer",
    earthformer_builder,
    defaults={
        "in_channels": 1,
        "out_dim": 1,
        "hidden_channels": 16,
        "num_heads": 4,
        "num_layers": 2,
    },
)

register_model(
    "swinlstm",
    swinlstm_builder,
    defaults={
        "in_channels": 1,
        "out_dim": 1,
        "embed_dim": 16,
        "hidden_channels": 16,
        "num_heads": 4,
        "window_size": 3,
    },
)

register_model(
    "earthfarseer",
    earthfarseer_builder,
    defaults={
        "in_channels": 1,
        "out_dim": 1,
        "hidden_channels": 16,
        "num_heads": 4,
        "num_layers": 2,
    },
)

register_model(
    "convgru_trajgru",
    convgru_trajgru_builder,
    defaults={
        "in_channels": 1,
        "out_dim": 1,
        "enc_channels": 16,
        "hidden_channels": 16,
        "kernel_size": 3,
    },
)

register_model(
    "tcn",
    tcn_builder,
    defaults={
        "in_channels": 1,
        "out_dim": 1,
        "embed_dim": 16,
        "hidden_channels": 16,
        "kernel_size": 3,
        "num_levels": 3,
        "dropout": 0.1,
    },
)

register_model(
    "utae",
    utae_builder,
    defaults={
        "in_channels": 1,
        "out_dim": 1,
        "hidden_channels": 16,
        "num_heads": 4,
    },
)

register_model(
    "segformer",
    segformer_builder,
    defaults={
        "in_channels": 1,
        "out_dim": 1,
        "embed_dims": (16, 32),
        "num_heads": (1, 2),
        "sr_ratios": (4, 2),
        "mlp_ratio": 2.0,
        "dropout": 0.1,
    },
)

register_model(
    "swin_unet",
    swin_unet_builder,
    defaults={
        "in_channels": 1,
        "out_dim": 1,
        "embed_dims": (16, 32),
        "num_heads": (1, 2),
        "window_size": 3,
        "mlp_ratio": 2.0,
        "dropout": 0.1,
    },
)

register_model(
    "vit_segmenter",
    vit_segmenter_builder,
    defaults={
        "in_channels": 1,
        "out_dim": 1,
        "patch_size": 4,
        "embed_dim": 64,
        "depth": 4,
        "num_heads": 4,
        "mlp_ratio": 2.0,
        "dropout": 0.1,
    },
)

register_model(
    "deep_ensemble",
    deep_ensemble_builder,
    defaults={
        "in_channels": 1,
        "out_dim": 1,
        "base_channels": 8,
        "ensemble_size": 5,
    },
)
