from .base import DataBundle, DataSplit, Dataset, FeatureSpec, LabelSpec
from .earthquake import (
    SeisBenchWaveformDataset,
    SyntheticEarthquakeForecastDataset,
    SyntheticEarthquakeWaveformDataset,
)
from .flood import (
    CamelsUSStreamflowDataset,
    CaravanStreamflowDataset,
    SyntheticFloodInundationDataset,
    SyntheticFloodMeshDataset,
    SyntheticFloodStreamflowDataset,
)
from .fpa_fod import FPAFODTabularDataset, FPAFODWeeklyDataset
from .graph import GraphTemporalDataset, graph_collate
from .registry import available_datasets, load_dataset, register_dataset
from .tc import (
    IBTrACSTropicalCycloneDataset,
    SHIPSXu2021Dataset,
    SyntheticSAFNetDataset,
    SyntheticSHIPSDataset,
    SyntheticTCNDDataset,
    SyntheticTropicalCycloneDataset,
    TropiCycloneNetDataset,
)
from .wildfire import (
    SyntheticWildfireDangerDataset,
    SyntheticWildfireSpreadDataset,
    SyntheticWildfireSpreadTemporalDataset,
    WildfireTrackORasterDataset,
    WildfireTrackOTabularDataset,
    WildfireTrackOTemporalDataset,
)
from .wrf_sfire import WRFSFireSpreadDataset

__all__ = [
    "DataBundle",
    "DataSplit",
    "Dataset",
    "FeatureSpec",
    "LabelSpec",
    "SeisBenchWaveformDataset",
    "SyntheticEarthquakeForecastDataset",
    "SyntheticEarthquakeWaveformDataset",
    "CamelsUSStreamflowDataset",
    "CaravanStreamflowDataset",
    "SyntheticFloodInundationDataset",
    "SyntheticFloodMeshDataset",
    "SyntheticFloodStreamflowDataset",
    "FPAFODTabularDataset",
    "FPAFODWeeklyDataset",
    "available_datasets",
    "load_dataset",
    "register_dataset",
    "GraphTemporalDataset",
    "graph_collate",
    "IBTrACSTropicalCycloneDataset",
    "SHIPSXu2021Dataset",
    "SyntheticSAFNetDataset",
    "SyntheticSHIPSDataset",
    "SyntheticTCNDDataset",
    "SyntheticTropicalCycloneDataset",
    "TropiCycloneNetDataset",
    "SyntheticWildfireDangerDataset",
    "SyntheticWildfireSpreadDataset",
    "SyntheticWildfireSpreadTemporalDataset",
    "WRFSFireSpreadDataset",
    "WildfireTrackORasterDataset",
    "WildfireTrackOTabularDataset",
    "WildfireTrackOTemporalDataset",
]

register_dataset(SyntheticEarthquakeForecastDataset.name, SyntheticEarthquakeForecastDataset)
register_dataset(SyntheticEarthquakeWaveformDataset.name, SyntheticEarthquakeWaveformDataset)
register_dataset(SeisBenchWaveformDataset.name, SeisBenchWaveformDataset)
register_dataset(SyntheticFloodInundationDataset.name, SyntheticFloodInundationDataset)
register_dataset(SyntheticFloodStreamflowDataset.name, SyntheticFloodStreamflowDataset)
register_dataset(SyntheticFloodMeshDataset.name, SyntheticFloodMeshDataset)
register_dataset(CamelsUSStreamflowDataset.name, CamelsUSStreamflowDataset)
register_dataset(CaravanStreamflowDataset.name, CaravanStreamflowDataset)
register_dataset(FPAFODTabularDataset.name, FPAFODTabularDataset)
register_dataset(FPAFODWeeklyDataset.name, FPAFODWeeklyDataset)
register_dataset(SyntheticTropicalCycloneDataset.name, SyntheticTropicalCycloneDataset)
register_dataset(IBTrACSTropicalCycloneDataset.name, IBTrACSTropicalCycloneDataset)
register_dataset(SHIPSXu2021Dataset.name, SHIPSXu2021Dataset)
register_dataset(SyntheticSHIPSDataset.name, SyntheticSHIPSDataset)
register_dataset(SyntheticSAFNetDataset.name, SyntheticSAFNetDataset)
register_dataset(SyntheticTCNDDataset.name, SyntheticTCNDDataset)
register_dataset(TropiCycloneNetDataset.name, TropiCycloneNetDataset)
register_dataset(SyntheticWildfireDangerDataset.name, SyntheticWildfireDangerDataset)
register_dataset(SyntheticWildfireSpreadDataset.name, SyntheticWildfireSpreadDataset)
register_dataset(SyntheticWildfireSpreadTemporalDataset.name, SyntheticWildfireSpreadTemporalDataset)
register_dataset(WRFSFireSpreadDataset.name, WRFSFireSpreadDataset)
register_dataset(WildfireTrackORasterDataset.name, WildfireTrackORasterDataset)
register_dataset(WildfireTrackOTemporalDataset.name, WildfireTrackOTemporalDataset)
register_dataset(WildfireTrackOTabularDataset.name, WildfireTrackOTabularDataset)
