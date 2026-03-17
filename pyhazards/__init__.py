from importlib import import_module
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pyhazards")
except PackageNotFoundError:
    __version__ = "0.0.0"  # fallback

_EXPORTS = {
    "DataBundle": ("pyhazards.datasets", "DataBundle"),
    "DataSplit": ("pyhazards.datasets", "DataSplit"),
    "Dataset": ("pyhazards.datasets", "Dataset"),
    "FeatureSpec": ("pyhazards.datasets", "FeatureSpec"),
    "LabelSpec": ("pyhazards.datasets", "LabelSpec"),
    "GraphTemporalDataset": ("pyhazards.datasets", "GraphTemporalDataset"),
    "graph_collate": ("pyhazards.datasets", "graph_collate"),
    "available_datasets": ("pyhazards.datasets", "available_datasets"),
    "load_dataset": ("pyhazards.datasets", "load_dataset"),
    "register_dataset": ("pyhazards.datasets", "register_dataset"),
    "HazardTask": ("pyhazards.tasks", "HazardTask"),
    "available_hazard_tasks": ("pyhazards.tasks", "available_hazard_tasks"),
    "get_hazard_task": ("pyhazards.tasks", "get_hazard_task"),
    "has_hazard_task": ("pyhazards.tasks", "has_hazard_task"),
    "BenchmarkConfig": ("pyhazards.configs", "BenchmarkConfig"),
    "DatasetRef": ("pyhazards.configs", "DatasetRef"),
    "ExperimentConfig": ("pyhazards.configs", "ExperimentConfig"),
    "ModelRef": ("pyhazards.configs", "ModelRef"),
    "ReportConfig": ("pyhazards.configs", "ReportConfig"),
    "dump_experiment_config": ("pyhazards.configs", "dump_experiment_config"),
    "load_experiment_config": ("pyhazards.configs", "load_experiment_config"),
    "Benchmark": ("pyhazards.benchmarks", "Benchmark"),
    "BenchmarkResult": ("pyhazards.benchmarks", "BenchmarkResult"),
    "BenchmarkRunSummary": ("pyhazards.benchmarks", "BenchmarkRunSummary"),
    "available_benchmarks": ("pyhazards.benchmarks", "available_benchmarks"),
    "build_benchmark": ("pyhazards.benchmarks", "build_benchmark"),
    "get_benchmark": ("pyhazards.benchmarks", "get_benchmark"),
    "register_benchmark": ("pyhazards.benchmarks", "register_benchmark"),
    "run_benchmark": ("pyhazards.benchmarks", "run_benchmark"),
    "CNNPatchEncoder": ("pyhazards.models", "CNNPatchEncoder"),
    "ClassificationHead": ("pyhazards.models", "ClassificationHead"),
    "MLPBackbone": ("pyhazards.models", "MLPBackbone"),
    "RegressionHead": ("pyhazards.models", "RegressionHead"),
    "SegmentationHead": ("pyhazards.models", "SegmentationHead"),
    "TemporalEncoder": ("pyhazards.models", "TemporalEncoder"),
    "available_models": ("pyhazards.models", "available_models"),
    "build_model": ("pyhazards.models", "build_model"),
    "register_model": ("pyhazards.models", "register_model"),
    "WildfireMamba": ("pyhazards.models", "WildfireMamba"),
    "wildfire_mamba_builder": ("pyhazards.models", "wildfire_mamba_builder"),
    "MetricBase": ("pyhazards.metrics", "MetricBase"),
    "ClassificationMetrics": ("pyhazards.metrics", "ClassificationMetrics"),
    "RegressionMetrics": ("pyhazards.metrics", "RegressionMetrics"),
    "SegmentationMetrics": ("pyhazards.metrics", "SegmentationMetrics"),
    "BenchmarkReport": ("pyhazards.reports", "BenchmarkReport"),
    "export_report_bundle": ("pyhazards.reports", "export_report_bundle"),
    "BenchmarkRunner": ("pyhazards.engine", "BenchmarkRunner"),
    "Trainer": ("pyhazards.engine", "Trainer"),
    "RAI_FIRE_URL": ("pyhazards.interactive_map", "RAI_FIRE_URL"),
    "open_interactive_map": ("pyhazards.interactive_map", "open_interactive_map"),
}

_SUBMODULES = {
    "datasets": "pyhazards.datasets",
    "tasks": "pyhazards.tasks",
    "configs": "pyhazards.configs",
    "benchmarks": "pyhazards.benchmarks",
    "models": "pyhazards.models",
    "metrics": "pyhazards.metrics",
    "reports": "pyhazards.reports",
    "engine": "pyhazards.engine",
    "interactive_map": "pyhazards.interactive_map",
}


def __getattr__(name: str) -> object:
    # Keep import-time side effects minimal while preserving the package API.
    module_name = _SUBMODULES.get(name)
    if module_name is not None:
        module = import_module(module_name)
        globals()[name] = module
        return module
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError("module 'pyhazards' has no attribute '{name}'".format(name=name)) from exc
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__) | set(_SUBMODULES))

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
