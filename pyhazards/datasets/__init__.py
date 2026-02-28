from .base import DataBundle, DataSplit, Dataset, FeatureSpec, LabelSpec
from .graph import GraphTemporalDataset, graph_collate
from .registry import available_datasets, load_dataset, register_dataset

__all__ = [
    "DataBundle",
    "DataSplit",
    "Dataset",
    "FeatureSpec",
    "LabelSpec",
    "available_datasets",
    "load_dataset",
    "register_dataset",
    "GraphTemporalDataset",
    "graph_collate",
]
from .wildfire_fpa_fod import FPAFODWildfireTabular, FPAFODWildfireWeekly

# Register datasets so framework load_dataset(...) can find them
register_dataset(
    name="wildfire_fpa_fod_tabular",
    builder=lambda **kwargs: FPAFODWildfireTabular(**kwargs),
)

register_dataset(
    name="wildfire_fpa_fod_weekly",
    builder=lambda **kwargs: FPAFODWildfireWeekly(**kwargs),
)
