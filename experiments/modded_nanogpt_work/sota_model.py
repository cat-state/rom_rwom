import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def norm(x: Tensor) -> Tensor:
    return F.rms_norm(x, (x.size(-1),))


def apply_rom_short_conv(
    value: Tensor,
    short_conv_norm: nn.Module | None,
    short_conv_module: nn.Module | None,
) -> Tensor:
    if short_conv_module is None:
        return value
    conv_in = short_conv_norm(value).T.unsqueeze(0)
    conv_out = short_conv_module(conv_in)[..., : value.size(0)].squeeze(0).T
    return value + F.silu(conv_out)


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True


def _next_prime_after(start: int, seen: set[int]) -> int:
    n = max(2, start + 1)
    while n in seen or not _is_prime(n):
        n += 1
    seen.add(n)
    return n


class PerHeadLinear(nn.Module):
    """Independent bias-free linear projection per memory read head."""

    def __init__(self, num_heads: int, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_heads, out_features, in_features))
        for head_weight in self.weight:
            nn.init.kaiming_uniform_(head_weight, a=math.sqrt(5))

    def forward(self, x: Tensor) -> Tensor:
        return torch.einsum("thd,hod->tho", x, self.weight.type_as(x))


class EngramLayerReadoutDelta(nn.Module):
    """Zero-initialized per-layer correction on top of the shared Engram readout."""

    def __init__(self, total_hash_heads: int, head_dim: int, model_dim: int):
        super().__init__()
        self.value_proj = PerHeadLinear(total_hash_heads, head_dim, model_dim)
        self.key_proj = PerHeadLinear(total_hash_heads, head_dim, model_dim)
        for param in self.parameters():
            nn.init.zeros_(param)


@dataclass(frozen=True)
class SotaEngramConfig:
    vocab_size: int
    model_dim: int
    memory_dim: int = 768
    num_heads: int = 1
    max_ngram: int = 3
    seed: int = 0
    pad_id: int = 50256
    token_vocab_size: int = 50257
    hash_seed: int = 0
    init_std: float = 0.01
    ngram_row_factors: tuple[float, ...] = (0.5, 1.5)
    layer_ids: tuple[int, ...] = (2, 8)
    layer_partition_group_ids: tuple[int, ...] = (0, 0)
    superpose_k: int = 2
    superpose_include_base: bool = True
    superpose_aux_scale: float = 0.5
    superpose_normalize: bool = True
    normalize_memory_heads: bool = True
    normalize_readout: bool = True
    short_conv: bool = True
    short_conv_kernel: int = 3
    head_mix: bool = True
    head_mix_init: tuple[float, ...] = (0.5, -0.5)
    readout_delta_scale: float = 1.0
    read_hit_scale_exponent: float = 0.25
    read_hit_scale_offset: float = 1.0
    read_hit_scale_min: float = 0.25
    read_hit_scale_max: float = 4.0
    read_hit_scale_norm_mean: bool = True
    avalanche_hash: bool = True


class SotaEngramMemory(nn.Module):
    """Extracted SOTA Engram readout path.

    This intentionally covers the branch used by the current record configs:
    per-head K=2 superposed ngram memory, shared readout with zero-init
    per-layer deltas, optional head mixing, layer partitions, normalized memory
    heads, and normalized output readout.
    """

    def __init__(self, cfg: SotaEngramConfig):
        super().__init__()
        self.cfg = cfg
        if min(cfg.vocab_size, cfg.model_dim, cfg.memory_dim, cfg.num_heads, cfg.max_ngram) <= 0:
            raise ValueError("Engram dimensions must be positive")
        self.total_hash_heads = (cfg.max_ngram - 1) * cfg.num_heads
        if cfg.memory_dim % self.total_hash_heads != 0:
            raise ValueError("memory_dim must be divisible by (max_ngram - 1) * num_heads")
        if cfg.superpose_k != 2 or not cfg.superpose_include_base:
            raise ValueError("SOTA extraction currently expects K=2 superpose with base included")
        if len(cfg.ngram_row_factors) != cfg.max_ngram - 1:
            raise ValueError("ngram_row_factors must have one entry for each ngram head")
        if len(cfg.layer_partition_group_ids) != len(cfg.layer_ids):
            raise ValueError("layer_partition_group_ids must match layer_ids")

        self.model_dim = cfg.model_dim
        self.memory_dim = cfg.memory_dim
        self.num_heads = cfg.num_heads
        self.max_ngram = cfg.max_ngram
        self.head_dim = cfg.memory_dim // self.total_hash_heads
        self.store_dim = self.head_dim
        self.pad_id = cfg.pad_id
        self.hash_token_vocab_size = cfg.token_vocab_size
        self.hash_pad_id = cfg.pad_id

        unique_group_ids = tuple(sorted(set(int(i) for i in cfg.layer_partition_group_ids)))
        group_index = {group_id: idx for idx, group_id in enumerate(unique_group_ids)}
        partition_group_ids = tuple(group_index[int(group_id)] for group_id in cfg.layer_partition_group_ids)
        self.layer_partition_ids = tuple(int(i) for i in cfg.layer_ids)
        self.layer_partition_group_ids = partition_group_ids
        self.layer_partition_index = {
            layer_id: partition_group_ids[idx] for idx, layer_id in enumerate(self.layer_partition_ids)
        }

        seen_primes: set[int] = set()
        row_factor_norm = (cfg.max_ngram - 1) / sum(cfg.ngram_row_factors)
        partition_count = max(partition_group_ids) + 1
        per_layer_vocab_size = max(2, cfg.vocab_size // partition_count)
        layer_head_mods: list[list[int]] = []
        flat_sizes: list[int] = []
        for _partition_id in range(partition_count):
            head_mods = []
            for ngram_idx, _ngram in enumerate(range(2, cfg.max_ngram + 1)):
                for _head in range(cfg.num_heads):
                    rows_for_head = max(
                        2,
                        int(round(per_layer_vocab_size * cfg.ngram_row_factors[ngram_idx] * row_factor_norm)),
                    )
                    head_mod = _next_prime_after(rows_for_head - 1, seen_primes)
                    head_mods.append(head_mod)
                    flat_sizes.append(head_mod)
            layer_head_mods.append(head_mods)
        flat_offsets = [0]
        for size in flat_sizes[:-1]:
            flat_offsets.append(flat_offsets[-1] + size)
        layer_offsets = []
        cursor = 0
        for _partition_id in range(partition_count):
            layer_offsets.append(flat_offsets[cursor : cursor + self.total_hash_heads])
            cursor += self.total_hash_heads
        self.register_buffer("head_mods", torch.tensor(layer_head_mods[0], dtype=torch.int64), persistent=False)
        self.register_buffer("offsets", torch.tensor(layer_offsets[0], dtype=torch.int64), persistent=False)
        self.register_buffer("layer_head_mods", torch.tensor(layer_head_mods, dtype=torch.int64), persistent=False)
        self.register_buffer("layer_offsets", torch.tensor(layer_offsets, dtype=torch.int64), persistent=False)
        self.num_memory_rows = sum(flat_sizes)

        superpose_salts = torch.arange(1, cfg.superpose_k + 1, dtype=torch.int64) * 0xD1B54A35
        self.register_buffer("superpose_salts", superpose_salts, persistent=False)

        generator = torch.Generator()
        generator.manual_seed(cfg.seed)
        max_multiplier = max(1, (2**31 - 1) // max(1, self.hash_token_vocab_size))
        multipliers = torch.randint(1, max_multiplier, (cfg.max_ngram,), generator=generator, dtype=torch.int64) * 2 + 1
        self.register_buffer("multipliers", multipliers, persistent=False)

        layer_multipliers = []
        for layer_id in cfg.layer_ids:
            layer_generator = torch.Generator()
            layer_generator.manual_seed(cfg.hash_seed + 10007 * int(layer_id))
            layer_multipliers.append(
                torch.randint(1, max_multiplier, (cfg.max_ngram,), generator=layer_generator, dtype=torch.int64) * 2 + 1
            )
        self.register_buffer("layer_multipliers", torch.stack(layer_multipliers), persistent=False)
        self.layer_hash_ids = tuple(int(i) for i in cfg.layer_ids)
        self.layer_hash_index = {layer_id: idx for idx, layer_id in enumerate(self.layer_hash_ids)}

        self.embedding = nn.Embedding(self.num_memory_rows, self.store_dim, sparse=True)
        self.embedding.weight.data.normal_(std=cfg.init_std)
        self.register_buffer("hit_hist", torch.zeros(self.num_memory_rows, dtype=torch.int32), persistent=False)
        self._eval_hit_hist: Tensor | None = None
        self.last_read_hit_scale: Tensor | None = None

        self.value_proj = PerHeadLinear(self.total_hash_heads, self.head_dim, cfg.model_dim)
        self.key_proj = PerHeadLinear(self.total_hash_heads, self.head_dim, cfg.model_dim)
        if cfg.short_conv:
            self.short_conv_norm = nn.RMSNorm(cfg.model_dim)
            self.short_conv = nn.Conv1d(
                cfg.model_dim,
                cfg.model_dim,
                cfg.short_conv_kernel,
                groups=cfg.model_dim,
                bias=False,
                padding=(cfg.short_conv_kernel - 1) * cfg.max_ngram,
                dilation=cfg.max_ngram,
            )
            nn.init.zeros_(self.short_conv.weight)
        else:
            self.short_conv_norm = None
            self.short_conv = None

        self.layer_readout_deltas = nn.ModuleDict({
            str(layer_id): EngramLayerReadoutDelta(self.total_hash_heads, self.head_dim, cfg.model_dim)
            for layer_id in cfg.layer_ids
        })
        self.layer_readout_delta_index = {layer_id: idx for idx, layer_id in enumerate(cfg.layer_ids)}

        if cfg.head_mix:
            if len(cfg.head_mix_init) != self.total_hash_heads:
                raise ValueError("head_mix_init must have one value per hash head")
            self.head_mix_logits = nn.Parameter(torch.tensor(cfg.head_mix_init, dtype=torch.float32))
        else:
            self.head_mix_logits = None

    @staticmethod
    def _avalanche_hash(x: Tensor) -> Tensor:
        x = x ^ (x >> 30)
        x = x * -4658895280553007687
        x = x ^ (x >> 27)
        x = x * -7723592293110705685
        return x ^ (x >> 31)

    def _hash_superpose(self, input_ids: Tensor, layer_id: int | None = None) -> Tensor:
        cfg = self.cfg
        x = input_ids.to(torch.int64)
        multipliers = self.multipliers
        if layer_id is not None:
            layer_idx = self.layer_hash_index.get(int(layer_id))
            if layer_idx is not None:
                multipliers = self.layer_multipliers[layer_idx]
        head_mods = self.head_mods
        offsets = self.offsets
        if layer_id is not None:
            layer_idx = self.layer_partition_index.get(int(layer_id))
            if layer_idx is not None:
                head_mods = self.layer_head_mods[layer_idx]
                offsets = self.layer_offsets[layer_idx]

        shifted = [x]
        for k in range(1, cfg.max_ngram):
            pad = torch.full((k,), self.hash_pad_id, dtype=torch.int64, device=x.device)
            shifted.append(torch.cat([pad, x[:-k]], dim=0))

        superpose_salts = self.superpose_salts.to(device=x.device)
        addresses = []
        head_idx = 0
        for ngram in range(2, cfg.max_ngram + 1):
            mix = shifted[0] * multipliers[0]
            for k in range(1, ngram):
                mix = torch.bitwise_xor(mix, shifted[k] * multipliers[k])
            if cfg.avalanche_hash:
                mix = self._avalanche_hash(mix ^ int(cfg.hash_seed))
            for _head in range(cfg.num_heads):
                head_addresses = [(mix % head_mods[head_idx]) + offsets[head_idx]]
                head_salt = torch.tensor((head_idx + 1) * 0xC2B2AE3D, dtype=torch.int64, device=x.device)
                superpose_mix = self._avalanche_hash(
                    torch.bitwise_xor(mix.unsqueeze(-1), superpose_salts[:1].view(1, -1) + head_salt)
                )
                head_addresses.append((superpose_mix % head_mods[head_idx]) + offsets[head_idx])
                addresses.append(torch.cat([addr.unsqueeze(-1) if addr.ndim == 1 else addr for addr in head_addresses], dim=-1))
                head_idx += 1
        return torch.stack(addresses, dim=1)

    def _lookup_memory_heads(self, addresses: Tensor) -> Tensor:
        flat_addresses = addresses.reshape(-1)
        return self.embedding(flat_addresses).view(*addresses.shape, self.store_dim)

    def _lookup_combined_memory_heads(self, addresses: Tensor) -> Tensor:
        cfg = self.cfg
        accum = None
        for slot in range(addresses.size(-1)):
            memory = self._lookup_memory_heads(addresses[..., slot])
            if slot > 0:
                memory = memory * cfg.superpose_aux_scale
            accum = memory if accum is None else accum + memory
        if accum is None:
            raise RuntimeError("Expected at least one Engram lookup slot")
        if cfg.superpose_normalize:
            denom = math.sqrt(1.0 + (addresses.size(-1) - 1) * cfg.superpose_aux_scale * cfg.superpose_aux_scale)
            return accum / denom
        return accum

    def _active_hit_hist(self) -> Tensor | None:
        return self._eval_hit_hist if self._eval_hit_hist is not None else self.hit_hist

    def read_hit_scale(self, addresses: Tensor) -> Tensor | None:
        cfg = self.cfg
        hist = self._active_hit_hist()
        if cfg.read_hit_scale_exponent == 0 or hist is None:
            self.last_read_hit_scale = None
            return None
        with torch.no_grad():
            counts = hist.index_select(0, addresses.detach().reshape(-1)).view_as(addresses).float()
            scale = (counts + cfg.read_hit_scale_offset).pow(cfg.read_hit_scale_exponent)
            if cfg.read_hit_scale_norm_mean:
                scale = scale / scale.mean().clamp_min(1e-6)
            scale.clamp_(min=cfg.read_hit_scale_min, max=cfg.read_hit_scale_max)
            self.last_read_hit_scale = scale.detach()
        return scale

    def record_hit_hist(self, addresses: Tensor) -> None:
        if self.hit_hist is None or not self.training or not torch.is_grad_enabled():
            return
        with torch.no_grad():
            rows = addresses.detach().reshape(-1)
            if rows.numel() == 0:
                return
            ones = torch.ones(rows.shape, dtype=self.hit_hist.dtype, device=rows.device)
            self.hit_hist.index_add_(0, rows, ones)

    def _head_mix_weights(self, layer_id: int | None, *, dtype: torch.dtype, device: torch.device) -> Tensor | None:
        del layer_id
        if self.head_mix_logits is None:
            return None
        weights = F.softmax(self.head_mix_logits.float(), dim=-1) * math.sqrt(float(self.total_hash_heads))
        return weights.to(device=device, dtype=dtype).view(1, self.total_hash_heads, 1)

    def forward(self, input_ids: Tensor, hidden_states: Tensor, layer_id: int | None = None) -> Tensor:
        assert input_ids.ndim == 1
        assert hidden_states.ndim == 2
        addresses = self._hash_superpose(input_ids, layer_id=layer_id)
        memory_heads = self._lookup_combined_memory_heads(addresses)
        read_hit_scale = self.read_hit_scale(addresses)
        self.record_hit_hist(addresses)

        memory_heads = memory_heads.to(dtype=hidden_states.dtype)
        if self.cfg.normalize_memory_heads:
            memory_heads = norm(memory_heads)

        memory = memory_heads
        readout_key = str(int(layer_id)) if layer_id is not None else ""
        readout_delta = self.layer_readout_deltas[readout_key] if readout_key in self.layer_readout_deltas else None
        readout_delta_value_scale = self.cfg.readout_delta_scale if readout_delta is not None else 0.0
        readout_delta_key_scale = self.cfg.readout_delta_scale if readout_delta is not None else 0.0

        value = self.value_proj(memory)
        if readout_delta is not None:
            value = value + readout_delta_value_scale * readout_delta.value_proj(memory)

        key_raw = self.key_proj(memory)
        if readout_delta is not None:
            key_raw = key_raw + readout_delta_key_scale * readout_delta.key_proj(memory)
        key = norm(key_raw)
        query = norm(hidden_states)
        gate = (key * query.unsqueeze(1)).sum(dim=-1) / math.sqrt(hidden_states.size(-1))
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gated_value = value * torch.sigmoid(gate).unsqueeze(-1)

        if read_hit_scale is not None:
            scale = read_hit_scale
            if scale.ndim == gated_value.ndim:
                scale = scale.mean(dim=-1)
            gated_value = gated_value * scale.to(dtype=gated_value.dtype).unsqueeze(-1)

        head_mix_weights = self._head_mix_weights(layer_id, dtype=gated_value.dtype, device=gated_value.device)
        if head_mix_weights is not None:
            gated_value = gated_value * head_mix_weights
        merged_value = gated_value.sum(dim=1)
        if head_mix_weights is None:
            merged_value = merged_value / math.sqrt(self.total_hash_heads)
        output = apply_rom_short_conv(merged_value, self.short_conv_norm, self.short_conv)
        if self.cfg.normalize_readout:
            return norm(output)
        return output
