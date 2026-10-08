"""Earthquake datasets: a reader for real SeisBench-format / STEAD waveform datasets and synthetic smoke data."""

from .seisbench import SeisBenchWaveformDataset, parse_trace_name, read_data_format
from .synthetic import (
    SyntheticEarthquakeForecastDataset,
    SyntheticEarthquakeWaveformDataset,
    SyntheticEarthquakeWavefieldDataset,
)

__all__ = [
    "SeisBenchWaveformDataset",
    "SyntheticEarthquakeForecastDataset",
    "SyntheticEarthquakeWaveformDataset",
    "SyntheticEarthquakeWavefieldDataset",
    "parse_trace_name",
    "read_data_format",
]
