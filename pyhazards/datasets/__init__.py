from .base import DataBundle, DataSplit, Dataset, FeatureSpec, LabelSpec
from .fpa_fod import FPAFODTabularDataset, FPAFODWeeklyDataset
from .graph import GraphTemporalDataset, graph_collate
from .registry import available_datasets, load_dataset, register_dataset

__all__ = [
    "DataBundle",
    "DataSplit",
    "Dataset",
    "FeatureSpec",
    "LabelSpec",
    "FPAFODTabularDataset",
    "FPAFODWeeklyDataset",
    "available_datasets",
    "load_dataset",
    "register_dataset",
    "GraphTemporalDataset",
    "graph_collate",
]

register_dataset(FPAFODTabularDataset.name, FPAFODTabularDataset)
register_dataset(FPAFODWeeklyDataset.name, FPAFODWeeklyDataset)
