from __future__ import annotations

from typing import Callable, Dict, Type

from .base import Benchmark

_BENCHMARK_REGISTRY: Dict[str, Type[Benchmark]] = {}


def register_benchmark(name: str, builder: Type[Benchmark]) -> None:
    key = name.strip().lower()
    if key in _BENCHMARK_REGISTRY:
        raise ValueError("Benchmark '{name}' already registered.".format(name=name))
    _BENCHMARK_REGISTRY[key] = builder


def available_benchmarks():
    return sorted(_BENCHMARK_REGISTRY.keys())


def get_benchmark(name: str):
    return _BENCHMARK_REGISTRY.get(name.strip().lower())


def build_benchmark(name: str) -> Benchmark:
    builder = get_benchmark(name)
    if builder is None:
        raise KeyError(
            "Benchmark '{name}' is not registered. Known: {known}".format(
                name=name,
                known=", ".join(available_benchmarks()),
            )
        )
    return builder()


__all__ = [
    "available_benchmarks",
    "build_benchmark",
    "get_benchmark",
    "register_benchmark",
]
