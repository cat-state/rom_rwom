"""Utilities for Rememberance of Memories experiments."""

from rom_rwom.ngram_hash import NgramHashConfig, NgramHasher

__all__ = [
    "NgramHashConfig",
    "NgramHasher",
    "RomGatedDeltaMemory",
    "RomMemoryOutput",
    "RomStateMemory",
]


def __getattr__(name: str):
    if name in {"RomGatedDeltaMemory", "RomMemoryOutput", "RomStateMemory"}:
        from rom_rwom.torch_state_memory import (
            RomGatedDeltaMemory,
            RomMemoryOutput,
            RomStateMemory,
        )

        return {
            "RomGatedDeltaMemory": RomGatedDeltaMemory,
            "RomMemoryOutput": RomMemoryOutput,
            "RomStateMemory": RomStateMemory,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
