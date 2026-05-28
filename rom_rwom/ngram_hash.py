"""Deterministic suffix n-gram hashing for ROM/RWOM memory addresses.

The first ROM target keeps Engram's addressing primitive intact: token suffixes
map to stable per-layer, per-head table rows. The payload behind those rows can
then evolve from a vector embedding to a recurrent hidden state.
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value == 2:
        return True
    if value % 2 == 0:
        return False
    divisor = 3
    while divisor * divisor <= value:
        if value % divisor == 0:
            return False
        divisor += 2
    return True


def _next_unique_prime(start: int, seen: set[int]) -> int:
    candidate = max(2, start + 1)
    while candidate in seen or not _is_prime(candidate):
        candidate += 1
    seen.add(candidate)
    return candidate


@dataclass(frozen=True)
class NgramHashConfig:
    """Configuration for Engram-style multi-head suffix n-gram hashes."""

    vocab_size: int
    table_size: int | tuple[int, ...]
    layer_ids: tuple[int, ...] = (2,)
    min_ngram: int = 2
    max_ngram: int = 2
    heads_per_ngram: int = 8
    pad_id: int = 0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.min_ngram < 1:
            raise ValueError("min_ngram must be >= 1")
        if self.max_ngram < self.min_ngram:
            raise ValueError("max_ngram must be >= min_ngram")
        if self.heads_per_ngram <= 0:
            raise ValueError("heads_per_ngram must be positive")
        if not self.layer_ids:
            raise ValueError("layer_ids cannot be empty")


class NgramHasher:
    """Stable multiplicative-XOR suffix n-gram hasher.

    `hash_batch` returns nested Python lists with shape
    `[batch][time][(max_ngram - min_ngram + 1) * heads_per_ngram]`.
    """

    _LAYER_SEED_STRIDE = 10007

    def __init__(self, config: NgramHashConfig) -> None:
        self.config = config
        self.ngram_orders = tuple(range(config.min_ngram, config.max_ngram + 1))
        self._multipliers = {
            layer_id: self._build_layer_multipliers(layer_id)
            for layer_id in config.layer_ids
        }
        self.table_sizes = self._build_table_sizes()

    @property
    def heads_per_token(self) -> int:
        return len(self.ngram_orders) * self.config.heads_per_ngram

    def _build_layer_multipliers(self, layer_id: int) -> tuple[int, ...]:
        rng = Random(self.config.seed + self._LAYER_SEED_STRIDE * layer_id)
        max_safe = max(1, (2**63 - 1) // self.config.vocab_size)
        return tuple(
            2 * rng.randrange(max_safe // 2 or 1) + 1
            for _ in range(self.config.max_ngram)
        )

    def _build_table_sizes(self) -> dict[int, tuple[tuple[int, ...], ...]]:
        if isinstance(self.config.table_size, int):
            base_sizes = (self.config.table_size,) * len(self.ngram_orders)
        else:
            base_sizes = self.config.table_size
        if len(base_sizes) != len(self.ngram_orders):
            raise ValueError("table_size tuple must match number of n-gram orders")

        seen: set[int] = set()
        sizes: dict[int, tuple[tuple[int, ...], ...]] = {}
        for layer_id in self.config.layer_ids:
            layer_sizes = []
            for base_size in base_sizes:
                head_sizes = []
                search_start = base_size - 1
                for _ in range(self.config.heads_per_ngram):
                    prime = _next_unique_prime(search_start, seen)
                    head_sizes.append(prime)
                    search_start = prime
                layer_sizes.append(tuple(head_sizes))
            sizes[layer_id] = tuple(layer_sizes)
        return sizes

    def hash_batch(self, input_ids: list[list[int]], layer_id: int) -> list[list[list[int]]]:
        if layer_id not in self._multipliers:
            raise KeyError(f"unknown layer_id {layer_id}")
        if not input_ids:
            return []
        width = len(input_ids[0])
        if any(len(row) != width for row in input_ids):
            raise ValueError("input_ids must be rectangular")

        return [self._hash_row(row, layer_id) for row in input_ids]

    def _hash_row(self, row: list[int], layer_id: int) -> list[list[int]]:
        multipliers = self._multipliers[layer_id]
        layer_sizes = self.table_sizes[layer_id]
        output: list[list[int]] = []

        for t in range(len(row)):
            token_hashes: list[int] = []
            for ngram_idx, ngram_order in enumerate(self.ngram_orders):
                mixed = 0
                for offset in range(ngram_order):
                    source_idx = t - offset
                    token_id = row[source_idx] if source_idx >= 0 else self.config.pad_id
                    mixed ^= token_id * multipliers[offset]
                for table_size in layer_sizes[ngram_idx]:
                    token_hashes.append(mixed % table_size)
            output.append(token_hashes)

        return output
