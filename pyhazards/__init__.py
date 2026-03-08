from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pyhazards")
except PackageNotFoundError:
    __version__ = "0.0.0"  # fallback

from .datasets import (
    DataBundle,
    DataSplit,
    Dataset,
    FeatureSpec,
    LabelSpec,
    GraphTemporalDataset,
    graph_collate,
    available_datasets,
    load_dataset,
    register_dataset,
)
from .tasks import HazardTask, available_hazard_tasks, get_hazard_task, has_hazard_task
from .configs import (
    BenchmarkConfig,
    DatasetRef,
    ExperimentConfig,
    ModelRef,
    ReportConfig,
    dump_experiment_config,
    load_experiment_config,
)
from .benchmarks import (
    Benchmark,
    BenchmarkResult,
    BenchmarkRunSummary,
    available_benchmarks,
    build_benchmark,
    get_benchmark,
    register_benchmark,
    run_benchmark,
)
from .models import (
    CNNPatchEncoder,
    ClassificationHead,
    MLPBackbone,
    RegressionHead,
    SegmentationHead,
    TemporalEncoder,
    available_models,
    build_model,
    register_model,
    WildfireMamba,
    wildfire_mamba_builder,
)
from .metrics import ClassificationMetrics, MetricBase, RegressionMetrics, SegmentationMetrics
from .reports import BenchmarkReport, export_report_bundle
from .engine import BenchmarkRunner, Trainer
from .interactive_map import RAI_FIRE_URL, open_interactive_map

__all__ = [
    "__version__",
    "DataBundle",
    "DataSplit",
    "Dataset",
    "FeatureSpec",
    "LabelSpec",
    "GraphTemporalDataset",
    "graph_collate",
    "available_datasets",
    "load_dataset",
    "register_dataset",
    "HazardTask",
    "available_hazard_tasks",
    "get_hazard_task",
    "has_hazard_task",
    "BenchmarkConfig",
    "DatasetRef",
    "ExperimentConfig",
    "ModelRef",
    "ReportConfig",
    "dump_experiment_config",
    "load_experiment_config",
    "Benchmark",
    "BenchmarkResult",
    "BenchmarkRunSummary",
    "available_benchmarks",
    "build_benchmark",
    "get_benchmark",
    "register_benchmark",
    "run_benchmark",
    "CNNPatchEncoder",
    "ClassificationHead",
    "RegressionHead",
    "SegmentationHead",
    "MLPBackbone",
    "TemporalEncoder",
    "available_models",
    "build_model",
    "register_model",
    "WildfireMamba",
    "wildfire_mamba_builder",
    "BenchmarkReport",
    "export_report_bundle",
    "BenchmarkRunner",
    "Trainer",
    "MetricBase",
    "ClassificationMetrics",
    "RegressionMetrics",
    "SegmentationMetrics",
    "RAI_FIRE_URL",
    "open_interactive_map",
]
