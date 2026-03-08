from .backbones import CNNPatchEncoder, MLPBackbone, TemporalEncoder
from .builder import build_model, default_builder
from .cnn_aspp import WildfireCNNASPP, cnn_aspp_builder
from .eqnet import EQNet, eqnet_builder
from .eqtransformer import EQTransformer, eqtransformer_builder
from .floodcast import FloodCast, floodcast_builder
from .fourcastnet_tc import FourCastNetTC, fourcastnet_tc_builder
from .gpd import GPD, gpd_builder
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
from .wildfire_fpa_dnn import WildfireFPADNN, wildfire_fpa_dnn_builder
from .wildfire_fpa_forecast import WildfireFPAForecast, wildfire_fpa_forecast_builder
from .wildfire_fpa_lstm import WildfireFPALSTM, wildfire_fpa_lstm_builder
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
    "EQNet",
    "eqnet_builder",
    "EQTransformer",
    "eqtransformer_builder",
    "FloodCast",
    "floodcast_builder",
    "FourCastNetTC",
    "fourcastnet_tc_builder",
    "GPD",
    "gpd_builder",
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
    "WildfireFPADNN",
    "wildfire_fpa_dnn_builder",
    "WildfireFPAForecast",
    "wildfire_fpa_forecast_builder",
    "WildfireFPALSTM",
    "wildfire_fpa_lstm_builder",
    "WildfireMamba",
    "wildfire_mamba_builder",
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
