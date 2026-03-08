from .base import DataBundle, DataSplit, Dataset, FeatureSpec, LabelSpec
from .earthquake import SyntheticEarthquakeForecastDataset, SyntheticEarthquakeWaveformDataset
from .flood import SyntheticFloodInundationDataset, SyntheticFloodStreamflowDataset
from .fpa_fod import FPAFODTabularDataset, FPAFODWeeklyDataset
from .graph import GraphTemporalDataset, graph_collate
from .registry import available_datasets, load_dataset, register_dataset
from .tc import SyntheticTropicalCycloneDataset
from .wildfire import SyntheticWildfireSpreadDataset

__all__ = [
    "DataBundle",
    "DataSplit",
    "Dataset",
    "FeatureSpec",
    "LabelSpec",
    "SyntheticEarthquakeForecastDataset",
    "SyntheticEarthquakeWaveformDataset",
    "SyntheticFloodInundationDataset",
    "SyntheticFloodStreamflowDataset",
    "FPAFODTabularDataset",
    "FPAFODWeeklyDataset",
    "available_datasets",
    "load_dataset",
    "register_dataset",
    "GraphTemporalDataset",
    "graph_collate",
    "SyntheticTropicalCycloneDataset",
    "SyntheticWildfireSpreadDataset",
]

register_dataset(SyntheticEarthquakeForecastDataset.name, SyntheticEarthquakeForecastDataset)
register_dataset(SyntheticEarthquakeWaveformDataset.name, SyntheticEarthquakeWaveformDataset)
register_dataset(SyntheticFloodInundationDataset.name, SyntheticFloodInundationDataset)
register_dataset(SyntheticFloodStreamflowDataset.name, SyntheticFloodStreamflowDataset)
register_dataset(FPAFODTabularDataset.name, FPAFODTabularDataset)
register_dataset(FPAFODWeeklyDataset.name, FPAFODWeeklyDataset)
register_dataset(SyntheticTropicalCycloneDataset.name, SyntheticTropicalCycloneDataset)
register_dataset(SyntheticWildfireSpreadDataset.name, SyntheticWildfireSpreadDataset)
