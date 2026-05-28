"""Tiny Torch prototype for ROM recurrent-state payloads.

This module is intentionally simple and slow. It models the payload idea before
we commit to a custom FLA/Triton path.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class RomMemoryOutput:
    """Auxiliary tensors from a ROM memory branch."""

    read: torch.Tensor
    residual: torch.Tensor
    next_state: torch.Tensor | None


class RomStateMemory(nn.Module):
    """A keyed table of GDN-style recurrent states.

    The table stores one state matrix per `(row, head)` with shape `[K, V]`.
    Reads are differentiable. Writes return a new table tensor and are intended
    for tiny experiments because collision handling is last-writer-wins.
    """

    def __init__(
        self,
        num_rows: int,
        num_heads: int,
        key_dim: int,
        value_dim: int,
        *,
        trainable: bool = True,
        init_std: float = 0.0,
    ) -> None:
        super().__init__()
        if min(num_rows, num_heads, key_dim, value_dim) <= 0:
            raise ValueError("all dimensions must be positive")

        state = torch.empty(num_rows, num_heads, key_dim, value_dim)
        if init_std == 0.0:
            nn.init.zeros_(state)
        else:
            nn.init.normal_(state, mean=0.0, std=init_std)

        if trainable:
            self.state = nn.Parameter(state)
        else:
            self.register_buffer("state", state)

    def forward(
        self,
        addresses: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        g: torch.Tensor,
        *,
        write: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Read and optionally update memory cells.

        Args:
            addresses: Integer tensor `[B, T, H]`.
            q: Query tensor `[B, T, H, K]`.
            k: Key tensor `[B, T, H, K]`.
            v: Value tensor `[B, T, H, V]`.
            beta: Delta learning-rate gate `[B, T, H]`.
            g: Log decay gate `[B, T, H]`.
            write: If true, return a cloned table with recurrent updates applied.

        Returns:
            output: `[B, T, H, V]`.
            next_state: Updated table when `write=True`, otherwise `None`.
        """
        self._validate_shapes(addresses, q, k, v, beta, g)

        B, T, H, _, V = *q.shape, v.shape[-1]
        head_ids = torch.arange(H, device=addresses.device).view(1, H).expand(B, H)
        table = self.state
        next_table = table.clone() if write else table
        outputs = []

        for t in range(T):
            addr_t = addresses[:, t]
            cell = next_table[addr_t, head_ids]

            cell = cell * g[:, t].exp()[..., None, None]
            prediction = torch.einsum("bhkv,bhk->bhv", cell, k[:, t])
            error = v[:, t] - prediction
            delta_v = beta[:, t][..., None] * error
            updated = cell + k[:, t].unsqueeze(-1) * delta_v.unsqueeze(-2)
            output_t = torch.einsum("bhk,bhkv->bhv", q[:, t], updated)
            outputs.append(output_t)

            if write:
                next_table[addr_t, head_ids] = updated

        return torch.stack(outputs, dim=1), next_table if write else None

    def _validate_shapes(
        self,
        addresses: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        g: torch.Tensor,
    ) -> None:
        if addresses.ndim != 3:
            raise ValueError("addresses must have shape [B, T, H]")
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            raise ValueError("q, k, and v must have shape [B, T, H, D]")
        if beta.shape != addresses.shape or g.shape != addresses.shape:
            raise ValueError("beta and g must match addresses shape [B, T, H]")
        if q.shape != k.shape:
            raise ValueError("q and k must have the same shape")
        if q.shape[:3] != addresses.shape or v.shape[:3] != addresses.shape:
            raise ValueError("q, k, v, and addresses must agree on [B, T, H]")
        if q.shape[-1] != self.state.shape[-2]:
            raise ValueError("q/k key_dim must match table key_dim")
        if v.shape[-1] != self.state.shape[-1]:
            raise ValueError("v value_dim must match table value_dim")
        if addresses.min() < 0 or addresses.max() >= self.state.shape[0]:
            raise ValueError("addresses are outside the table range")


class RomGatedDeltaMemory(nn.Module):
    """A small GDN-style ROM branch for transformer hidden states.

    This is a prototype layer, not the final FLA kernel path. It keeps Engram's
    deterministic address lookup but stores a recurrent delta-rule state matrix
    at each memory row. The residual projection is zero-initialized so the branch
    starts as a no-op when attached to an existing model.
    """

    def __init__(
        self,
        hidden_size: int,
        num_rows: int,
        num_heads: int,
        key_dim: int,
        value_dim: int,
        *,
        trainable_state: bool = True,
        normalize_qk: bool = True,
    ) -> None:
        super().__init__()
        if min(hidden_size, num_rows, num_heads, key_dim, value_dim) <= 0:
            raise ValueError("all dimensions must be positive")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.normalize_qk = normalize_qk

        self.q_proj = nn.Linear(hidden_size, num_heads * key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_heads * key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_heads * value_dim, bias=False)
        self.beta_proj = nn.Linear(hidden_size, num_heads, bias=True)
        self.decay_proj = nn.Linear(hidden_size, num_heads, bias=True)
        self.write_gate_proj = nn.Linear(hidden_size, num_heads, bias=True)
        self.out_proj = nn.Linear(num_heads * value_dim, hidden_size, bias=False)
        nn.init.zeros_(self.out_proj.weight)

        self.memory = RomStateMemory(
            num_rows=num_rows,
            num_heads=num_heads,
            key_dim=key_dim,
            value_dim=value_dim,
            trainable=trainable_state,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        addresses: torch.Tensor,
        *,
        write: bool = False,
    ) -> tuple[torch.Tensor, RomMemoryOutput]:
        """Apply the ROM branch and return hidden states plus diagnostics.

        Args:
            hidden_states: `[B, T, D]`.
            addresses: `[B, T]` shared across heads or `[B, T, H]` per-head.
            write: If true, return a candidate next state table with gated
                writes applied. The module parameter is not mutated.
        """
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [B, T, D]")
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError("hidden_states hidden dimension does not match layer")

        B, T, _ = hidden_states.shape
        addresses = self._expand_addresses(addresses, B, T, hidden_states.device)

        q = self._project_heads(self.q_proj(hidden_states), self.key_dim)
        k = self._project_heads(self.k_proj(hidden_states), self.key_dim)
        v = self._project_heads(self.v_proj(hidden_states), self.value_dim)
        if self.normalize_qk:
            q = F.normalize(q, p=2, dim=-1)
            k = F.normalize(k, p=2, dim=-1)

        beta = torch.sigmoid(self.beta_proj(hidden_states))
        decay = -F.softplus(self.decay_proj(hidden_states))
        write_gate = torch.sigmoid(self.write_gate_proj(hidden_states))

        read, candidate_state = self.memory(
            addresses,
            q,
            k,
            v,
            beta,
            decay,
            write=write,
        )
        next_state = None
        if candidate_state is not None:
            next_state = self._blend_written_rows(
                addresses=addresses,
                candidate_state=candidate_state,
                write_gate=write_gate,
            )

        residual = self.out_proj(read.reshape(B, T, self.num_heads * self.value_dim))
        return hidden_states + residual, RomMemoryOutput(
            read=read,
            residual=residual,
            next_state=next_state,
        )

    def _project_heads(self, x: torch.Tensor, head_dim: int) -> torch.Tensor:
        B, T, _ = x.shape
        return x.reshape(B, T, self.num_heads, head_dim)

    def _expand_addresses(
        self,
        addresses: torch.Tensor,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        if addresses.ndim == 2:
            if addresses.shape != (batch_size, seq_len):
                raise ValueError("2D addresses must have shape [B, T]")
            addresses = addresses.unsqueeze(-1).expand(batch_size, seq_len, self.num_heads)
        elif addresses.ndim == 3:
            if addresses.shape != (batch_size, seq_len, self.num_heads):
                raise ValueError("3D addresses must have shape [B, T, H]")
        else:
            raise ValueError("addresses must have shape [B, T] or [B, T, H]")
        return addresses.to(device=device, dtype=torch.long)

    def _blend_written_rows(
        self,
        addresses: torch.Tensor,
        candidate_state: torch.Tensor,
        write_gate: torch.Tensor,
    ) -> torch.Tensor:
        next_state = self.memory.state.clone()
        B, T, H = addresses.shape
        head_ids = torch.arange(H, device=addresses.device).view(1, 1, H).expand(B, T, H)
        gate = write_gate[..., None, None]
        old_rows = next_state[addresses, head_ids]
        new_rows = candidate_state[addresses, head_ids]
        next_state[addresses, head_ids] = gate * new_rows + (1.0 - gate) * old_rows
        return next_state
