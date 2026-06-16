#!/usr/bin/env python3
"""Tiny local equivalence test for the extracted SOTA Engram readout."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import types

import torch
import torch.nn.functional as F
from torch import Tensor, nn


ROOT = Path(__file__).resolve().parents[1]


def _source_slice(text: str, start: str, end: str) -> str:
    start_idx = text.index(start)
    end_idx = text.index(end, start_idx)
    return text[start_idx:end_idx]


def load_legacy_engram_module() -> types.ModuleType:
    """Execute only the original Engram definitions needed for the test."""
    source = (ROOT / "train_gpt.py").read_text()
    snippets = [
        _source_slice(source, "def norm(x: Tensor):", "\ndef debug_check_finite"),
        _source_slice(source, "def apply_rom_short_conv", "\ndef apply_rom_ema_smooth"),
        _source_slice(source, "def _is_prime", "\n\nENGRAM_ANALYSIS_CHUNKS"),
        _source_slice(source, "class PerHeadLinear", "\n\nclass RomTokenBigramMemory"),
    ]

    module = types.ModuleType("legacy_engram_slice")
    g = module.__dict__
    g.update(
        {
            "math": math,
            "torch": torch,
            "F": F,
            "Tensor": Tensor,
            "nn": nn,
            "ThreadPoolExecutor": None,
            "Future": object,
            "dist": types.SimpleNamespace(is_available=lambda: False, is_initialized=lambda: False),
            "is_torch_compiling": lambda: False,
            "maybe_disable_engram_compile": lambda fn: fn,
            "debug_check_finite": lambda *args, **kwargs: None,
            "debug_record_engram_addresses": lambda *args, **kwargs: None,
            "debug_report_engram_backward": lambda _label, grad, _addresses=None: grad,
            "record_engram_analysis": lambda *args, **kwargs: None,
            "record_engram_output_grad_metrics": lambda *args, **kwargs: None,
            "coalesce_row_sparse_grad": lambda grad: grad,
            "row_rms_stable": lambda x: x.norm(dim=1) / (max(1, x.shape[1]) ** 0.5),
            "rms_no_alloc": lambda x: x.norm() / (x.numel() ** 0.5),
            "build_canonical_token_lookup": lambda _vocab_size: (_raise_canonical(), 0),
        }
    )
    g.update(_legacy_flags())
    exec("\n\n".join(snippets), g)
    return module


def _raise_canonical() -> None:
    raise RuntimeError("canonicalization is disabled in this local tiny equivalence test")


def _legacy_flags() -> dict[str, object]:
    false_names = [
        "engram_latent",
        "engram_canonicalize",
        "engram_offload",
        "engram_offload_lazy_moments",
        "engram_sparse_adam",
        "engram_sparse_vector_adam",
        "engram_sparse_row_adagrad",
        "engram_sparse_grad_coalesce_hook",
        "engram_init_zero",
        "engram_freeze_memory",
        "engram_shadow_grad",
        "engram_cache_readout",
        "engram_static_gate",
        "engram_layer_head_mix",
        "engram_layer_head_mix_delta",
        "engram_head_mix_freeze",
        "engram_sketch_slot_mix",
        "engram_sketch_combine_mix",
        "engram_sketch_aux_learned_scale",
        "engram_debug_backward",
        "rom_debug_nan",
        "engram_mask_unhit_eval",
        "engram_mask_hit_invert_eval",
        "engram_hot_split",
        "engram_hot_split_value_only",
        "engram_sketch_hit_hist_base_only",
        "engram_shadow_only",
        "engram_layer_signs",
        "engram_layer_row_signs",
        "engram_layer_row_signs_aux_only",
        "engram_detach_value_memory",
        "engram_detach_key_memory",
        "engram_fixed_half_gate",
        "engram_output_dropout_current",
        "engram_head_dropout_current",
        "engram_sketch_slot_readout",
        "engram_latent_aux_readout",
        "engram_hit_dropout_invert_scale",
        "engram_hit_dropout_decay_steps",
        "engram_hit_dropout_schedule_steps",
        "engram_ngram_read_scale_norm",
            "engram_layer_readout_delta_learned_scale",
        ]
    flags = {name: False for name in false_names}
    flags.update(
        {
            "engram_store_dim": 0,
            "engram_per_head": True,
            "engram_untied_proj": True,
            "engram_ngram_row_factors": (0.5, 1.5),
            "engram_sketch_k": 1,
            "engram_sketch_dim_signs": False,
            "engram_sketch_include_base": True,
            "engram_sketch_scalar_signs": False,
            "engram_sketch_aux_scale": 1.0,
            "engram_sketch_aux_scale_final": 1.0,
            "engram_sketch_aux_scale_schedule_steps": 0,
            "engram_sketch_aux_scale_schedule_start": 0,
            "engram_sketch_aux_learned_scale_max": 1.0,
            "engram_superpose_k": 2,
            "engram_superpose_include_base": True,
            "engram_superpose_aux_scale": 0.5,
            "engram_superpose_aux_scale_final": 0.5,
            "engram_superpose_aux_scale_schedule_steps": 0,
            "engram_superpose_aux_scale_schedule_start": 0,
            "engram_superpose_normalize": True,
            "engram_hash_seed": 0,
            "engram_avalanche_hash": True,
            "engram_init_std": 0.01,
            "engram_sparse_scalar_adam": True,
            "engram_short_conv": True,
            "engram_short_conv_kernel": 3,
            "engram_hit_hist": True,
            "engram_hit_hist_kinds": {"lm"},
            "engram_layer_hashes": True,
            "engram_normalize_memory_heads": True,
            "engram_normalize_readout": True,
            "engram_detach_memory_layers": set(),
            "engram_cache_detach_memory": False,
            "engram_head_mix": True,
            "engram_head_mix_init": (0.5, -0.5),
            "engram_layer_readout_delta_scale": 1.0,
            "engram_layer_readout_delta_scale_final": 1.0,
            "engram_layer_readout_delta_scale_schedule_steps": 0,
            "engram_layer_readout_delta_scale_schedule_start": 0,
            "engram_layer_readout_delta_learned_scale_init": 1.0,
            "engram_layer_readout_delta_learned_scale_max": 1.0,
            "engram_layer_sign_aux_scale": 0.0,
            "engram_layer_sign_aux_scale_final": 0.0,
            "engram_layer_sign_aux_scale_schedule_steps": 0,
            "engram_layer_sign_aux_scale_schedule_start": 0,
            "engram_read_hit_scale_exponent": 0.25,
            "engram_read_hit_scale_offset": 1.0,
            "engram_read_hit_scale_min": 0.25,
            "engram_read_hit_scale_max": 4.0,
            "engram_read_hit_scale_norm_mean": True,
            "engram_mask_hit_min_eval": 0,
            "engram_mask_hit_max_eval": 0,
            "engram_mask_unhit_eval_mode": "zero",
            "engram_eval_hit_scale": 1.0,
            "engram_eval_hit_scale_min": 0,
            "engram_eval_hit_scale_max": 0,
            "engram_eval_hit_scale_invert": False,
            "engram_hot_split_aux_scale": 0.0,
            "engram_hot_split_aux_scale_schedule_steps": 0,
            "engram_hot_split_aux_scale_schedule_start": 0,
            "engram_hot_split_aux_scale_final": 0.0,
            "engram_hot_split_ramp_steps": 0,
            "engram_hit_dropout": 0.0,
            "engram_hit_dropout_final": 0.0,
            "engram_hit_dropout_min_hits": 0,
            "engram_hit_dropout_schedule_start": 0,
            "engram_hit_dropout_decay_start": 0,
            "engram_hit_dropout_decay_final": 0.0,
            "engram_ngram_read_scales": (),
            "engram_ngram_read_scales_final": (),
            "engram_ngram_read_scale_schedule_steps": 0,
            "rom_debug_nan_current_step": 0,
            "rom_debug_nan_min_step": 0,
            "rom_output_scale": 1.0,
        }
    )
    return flags


def build_pair():
    import sys

    sys.path.insert(0, str(ROOT))
    from sota_model import SotaEngramConfig, SotaEngramMemory

    cfg = SotaEngramConfig(
        vocab_size=64,
        token_vocab_size=128,
        model_dim=16,
        memory_dim=8,
        max_ngram=3,
        layer_ids=(2, 8),
        layer_partition_group_ids=(0, 0),
        pad_id=0,
        hash_seed=0,
    )
    legacy_mod = load_legacy_engram_module()
    torch.manual_seed(1234)
    legacy = legacy_mod.EngramBigramMemory(
        cfg.vocab_size,
        cfg.model_dim,
        cfg.memory_dim,
        cfg.num_heads,
        cfg.max_ngram,
        seed=cfg.seed,
        pad_id=cfg.pad_id,
        token_vocab_size=cfg.token_vocab_size,
        layer_hash_ids=cfg.layer_ids,
        layer_readout_delta_ids=cfg.layer_ids,
        layer_partition_ids=cfg.layer_ids,
        layer_partition_group_ids=cfg.layer_partition_group_ids,
    )
    torch.manual_seed(1234)
    extracted = SotaEngramMemory(cfg)
    extracted.load_state_dict(legacy.state_dict(), strict=False)
    legacy.eval()
    extracted.eval()
    return legacy, extracted


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=17)
    args = parser.parse_args()
    if args.tokens < 3:
        raise ValueError("Use --tokens >= 3; the original SOTA ngram hash expects at least max_ngram tokens")
    legacy, extracted = build_pair()
    torch.manual_seed(20260616)
    input_ids = torch.randint(0, 64, (args.tokens,), dtype=torch.int64)
    hidden_states = torch.randn(args.tokens, 16)

    max_diff = 0.0
    for layer_id in (2, 8):
        with torch.no_grad():
            legacy_out = legacy(input_ids, hidden_states, layer_id=layer_id)
            extracted_out = extracted(input_ids, hidden_states, layer_id=layer_id)
        diff = (legacy_out - extracted_out).abs().max().item()
        max_diff = max(max_diff, diff)
        torch.testing.assert_close(legacy_out, extracted_out, rtol=1e-6, atol=1e-6)
        print(f"layer {layer_id}: ok max_abs_diff={diff:.3e} shape={tuple(extracted_out.shape)}")
    print(f"sota extract equivalence passed max_abs_diff={max_diff:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
