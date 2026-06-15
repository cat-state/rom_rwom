import os
import sys

if "--run_id" in sys.argv:
    run_id_arg_index = sys.argv.index("--run_id")
    if run_id_arg_index + 1 >= len(sys.argv):
        raise ValueError("--run_id requires a value")
    os.environ.setdefault("RUN_ID", sys.argv[run_id_arg_index + 1])

# Read the current file and the kernels file code ASAP, for logging
with open(sys.argv[0], 'r') as f:
    code = f.read()
with open(os.path.join(os.path.dirname(sys.argv[0]), 'triton_kernels.py'), 'r') as f:
    code += f"\n\n{'-'*40}\n# triton_kernels.py\n{'-'*40}\n\n"
    code += f.read()

import copy
import glob
import html as html_lib
import json
import math
import re
import threading
import time
import unicodedata
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, fields
from itertools import accumulate, pairwise
from pathlib import Path
import gc

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import torch
import triton
import numpy as np

torch.empty(
    1, device=f"cuda:{os.environ['LOCAL_RANK']}", requires_grad=True
).backward()  # prevents a bug on some systems
import torch._dynamo as dynamo
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention

# torch._inductor.config.coordinate_descent_tuning = True # we have banned this flag for new records because it causes compilation to take 30min
from kernels import get_kernel
from torch import Tensor, nn

from triton_kernels import XXT, XTX, ba_plus_cAA, FusedLinearReLUSquareFunction, FusedSoftcappedCrossEntropy, transpose_add, transpose_copy
# Fused triton kernel: relu(x @ W1.T)^2 @ W2.T
# https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
ReLUSqrdMLP = FusedLinearReLUSquareFunction.apply


@torch.compile(dynamic=True)
def rms_no_alloc(x: Tensor) -> Tensor:
    return x.norm() / (x.numel() ** 0.5)


@torch.compile(dynamic=True)
def table_rms_no_alloc(x: Tensor, table_numel: int) -> Tensor:
    return x.norm() / (table_numel ** 0.5)


def row_rms_stable(x: Tensor) -> Tensor:
    return x.norm(dim=1) / (max(1, x.shape[1]) ** 0.5)


engram_debug_address_records: list[tuple[int, str, Tensor]] = []


def debug_record_engram_addresses(label: str, addresses: Tensor) -> None:
    if not engram_debug_backward or not rom_debug_nan or rom_debug_nan_current_step < rom_debug_nan_min_step:
        return
    if is_torch_compiling():
        return
    engram_debug_address_records.append((rom_debug_nan_current_step, label, addresses.detach().reshape(-1).to(device="cpu", dtype=torch.long)))


def debug_report_engram_bad_row_provenance(name: str, bad_rows: Tensor) -> None:
    if not engram_debug_address_records or bad_rows.numel() == 0:
        return
    bad_rows_cpu = bad_rows.detach().reshape(-1).to(device="cpu", dtype=torch.long)
    print(
        f"ROM_DEBUG_ENGRAM_ROW_PROVENANCE {name}: bad_rows={int(bad_rows_cpu.numel())}"
        f" records={len(engram_debug_address_records)}",
        flush=True,
    )
    for record_step, label, flat_rows in engram_debug_address_records:
        matches = torch.isin(flat_rows, bad_rows_cpu)
        hit_count = int(matches.sum().item())
        if hit_count == 0:
            continue
        matched_rows = torch.unique(flat_rows[matches])
        print(
            f"ROM_DEBUG_ENGRAM_ROW_SOURCE step={record_step} {label}:"
            f" hits={hit_count} unique_bad_rows={int(matched_rows.numel())}"
            f" first_row={int(matched_rows[0].item())}",
            flush=True,
        )


def debug_raise_nonfinite(name: str, x: Tensor, rows: Tensor | None = None) -> None:
    if not rom_debug_nan or rom_debug_nan_current_step < rom_debug_nan_min_step:
        return
    finite = torch.isfinite(x)
    if bool(finite.all().item()):
        return
    nonfinite = ~finite
    nonfinite_count = int(nonfinite.sum().item())
    row_info = ""
    if rows is not None and x.ndim >= 2:
        bad_row_mask = nonfinite.flatten(start_dim=1).any(dim=1)
        bad_pos = torch.nonzero(bad_row_mask, as_tuple=False).flatten()
        if bad_pos.numel() > 0:
            first_pos = int(bad_pos[0].item())
            row_info = f" bad_rows={int(bad_pos.numel())} first_pos={first_pos} first_row={int(rows[first_pos].item())}"
            debug_report_engram_bad_row_provenance(name, rows[bad_pos])
    x_float = x.detach().float()
    absmax = torch.nan_to_num(x_float, nan=0.0, posinf=0.0, neginf=0.0).abs().amax().item()
    finite_count = x.numel() - nonfinite_count
    raise RuntimeError(
        f"ROM_DEBUG_NAN nonfinite {name}: shape={tuple(x.shape)} dtype={x.dtype}"
        f" finite_count={finite_count} nonfinite_count={nonfinite_count}"
        f" finite_frac={finite_count / max(1, x.numel()):.9f}"
        f" absmax_finite={absmax:.6g}{row_info}"
    )


def sanitize_sparse_grad_values_(grad_values: Tensor) -> Tensor:
    grad_values.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
    # fp32 square overflows above ~1.8e19; keep Adam moments finite.
    grad_values.clamp_(-1.0e19, 1.0e19)
    return grad_values


def _sparse_coo_tensor_coalesced(indices: Tensor, values: Tensor, shape: torch.Size | tuple[int, ...], *, device: torch.device, dtype: torch.dtype) -> Tensor:
    try:
        return torch.sparse_coo_tensor(indices, values, shape, device=device, dtype=dtype, is_coalesced=True)
    except TypeError:
        sparse = torch.sparse_coo_tensor(indices, values, shape, device=device, dtype=dtype)
        sparse._coalesced_(True)
        return sparse


def coalesce_row_sparse_grad(grad: Tensor) -> Tensor:
    if not engram_manual_sparse_coalesce or rom_debug_nan_current_step < engram_manual_sparse_coalesce_start:
        return grad.coalesce()
    if grad.is_coalesced():
        return grad
    indices = grad._indices()
    values = grad._values()
    if indices.ndim != 2 or indices.size(0) != 1:
        return grad.coalesce()
    rows = indices[0]
    if rows.numel() == 0:
        return _sparse_coo_tensor_coalesced(indices, values, grad.shape, device=grad.device, dtype=grad.dtype)
    unique_rows, inverse = torch.unique(rows, sorted=True, return_inverse=True)
    summed = torch.zeros((unique_rows.numel(), *values.shape[1:]), dtype=torch.float32, device=values.device)
    summed.index_add_(0, inverse, values.float())
    return _sparse_coo_tensor_coalesced(unique_rows.unsqueeze(0), summed.to(dtype=values.dtype), grad.shape, device=grad.device, dtype=grad.dtype)


def scheduled_scalar(start: float, final: float, steps: int, schedule_start: int, step: int) -> float:
    if steps <= 0:
        return start
    if step <= schedule_start:
        return start
    progress = min(1.0, max(0.0, (step - schedule_start) / steps))
    return start + (final - start) * progress


def index_copy_cast_chunked_(dst: Tensor, dim: int, index: Tensor, src: Tensor, *, max_bytes: int = 256 * 1024 * 1024) -> None:
    if src.dtype == dst.dtype:
        dst.index_copy_(dim, index, src)
        return
    if index.numel() == 0:
        return
    row_bytes = max(1, src[0].numel() * torch.empty((), dtype=dst.dtype, device="cpu").element_size())
    chunk_rows = max(1, max_bytes // row_bytes)
    for start in range(0, index.numel(), chunk_rows):
        end = min(index.numel(), start + chunk_rows)
        dst.index_copy_(dim, index[start:end], src[start:end].to(dst.dtype))


def indexed_rows_rms_no_alloc(param: Tensor, index: Tensor, *, max_bytes: int = 256 * 1024 * 1024) -> Tensor:
    if index.numel() == 0:
        return torch.zeros((), dtype=torch.float32, device=param.device)
    row_width = max(1, int(param[0].numel()))
    row_bytes = row_width * torch.empty((), dtype=torch.float32, device="cpu").element_size()
    chunk_rows = max(1, max_bytes // row_bytes)
    total = torch.zeros((), dtype=torch.float32, device=param.device)
    count = 0
    for start in range(0, index.numel(), chunk_rows):
        end = min(index.numel(), start + chunk_rows)
        rows = param.index_select(0, index[start:end]).float()
        total = total + rows.norm().square()
        count += rows.numel()
    return (total / max(1, count)).sqrt()


dynamo.config.recompile_limit = int(os.environ.get("TORCH_RECOMPILE_LIMIT", "256"))

# -----------------------------------------------------------------------------
# Distributed training setup
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
assert 8 % world_size == 0, "world_size must be a divisor of 8"
grad_accum_steps = int(os.environ.get("GRAD_ACCUM_STEPS", str(8 // world_size)))
assert grad_accum_steps >= 1, "GRAD_ACCUM_STEPS must be positive"
grad_scale = 1 / grad_accum_steps # consistent grad magnitudes between different num_devices
assert torch.cuda.is_available()
device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(device)
dist.init_process_group(backend="cuda:nccl,cpu:gloo", device_id=device)
dist.barrier()
master_process = (rank == 0) # this process will do logging, checkpointing etc.
device_capability = torch.cuda.get_device_capability(device)

def env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return bool(int(value))

is_sm120_or_newer = device_capability[0] >= 12
compile_model = env_flag("COMPILE_MODEL", not is_sm120_or_newer)
compile_backend = os.environ.get("COMPILE_BACKEND", "").strip() or None
compile_layer_modules = env_flag("COMPILE_LAYER_MODULES", False)
compile_dense_layer_body = env_flag("COMPILE_DENSE_LAYER_BODY", False)
plain_mlp_train = env_flag("PLAIN_MLP_TRAIN", False)
fused_ce_eval = env_flag("FUSED_CE_EVAL", not is_sm120_or_newer)
final_smear_mtp = env_flag("FINAL_SMEAR_MTP", False)
final_smear_mtp_init = float(os.environ.get("FINAL_SMEAR_MTP_INIT", "0.0"))
snoo_outer = env_flag("SNOO_OUTER", False)
snoo_lr = float(os.environ.get("SNOO_LR", "0.68"))
snoo_momentum = float(os.environ.get("SNOO_MOMENTUM", "0.37"))
snoo_k = int(os.environ.get("SNOO_K", "28"))
snoo_include_engram = env_flag("SNOO_INCLUDE_ENGRAM", False)
normuon_update_smoothing = float(os.environ.get("NORMUON_UPDATE_SMOOTHING", "0.0"))
skip_kernel_warmup = env_flag("SKIP_KERNEL_WARMUP", False)
adam_embed_vector_adam = env_flag("ADAM_EMBED_VECTOR_ADAM", False)
adam_embed_scalar_adam = env_flag("ADAM_EMBED_SCALAR_ADAM", False)
adam_embed_vecadam_labels = {"embed", "value_embeds", "bigram_embed", "bigram_embed.embedding"}
wandb_enabled = env_flag("WANDB", False) or env_flag("WANDB_ENABLE", False)
wandb_project = os.environ.get("WANDB_PROJECT", "rom-rwom")
wandb_entity = os.environ.get("WANDB_ENTITY", "").strip() or None
wandb_group = os.environ.get("WANDB_GROUP", "").strip() or None
wandb_name = os.environ.get("WANDB_NAME", "").strip()
wandb_mode = os.environ.get("WANDB_MODE", "").strip() or None
wandb_tags = tuple(tag.strip() for tag in os.environ.get("WANDB_TAGS", "").replace(",", " ").split() if tag.strip())
wandb_log_code = env_flag("WANDB_LOG_CODE", False)
wandb_dir = os.environ.get("WANDB_DIR", "logs/wandb")
wandb_histograms = env_flag("WANDB_HISTOGRAMS", True)
wandb_hist_every = int(os.environ.get("WANDB_HIST_EVERY", "250"))
wandb_hist_rows = int(os.environ.get("WANDB_HIST_ROWS", "65536"))
wandb_hist_seed = int(os.environ.get("WANDB_HIST_SEED", "12345"))
engram_hist_nan_assert = env_flag("ENGRAM_HIST_NAN_ASSERT", False)
model_seed_raw = os.environ.get("MODEL_SEED", "").strip()
model_seed = int(model_seed_raw) if model_seed_raw else None
train_data_seed = int(os.environ.get("TRAIN_DATA_SEED", "0"))
rom_bigram = env_flag("ROM_BIGRAM", False)
engram_bigram = env_flag("ENGRAM_BIGRAM", False)
rom_single_token = env_flag("ROM_SINGLE_TOKEN", False)
if rom_single_token:
    rom_bigram = True
rom_token = env_flag("ROM_TOKEN", False)
rom_engram_gate = env_flag("ROM_ENGRAM_GATE", False)
rom_write = env_flag("ROM_WRITE", False)
rom_debug_nan = env_flag("ROM_DEBUG_NAN", False)
rom_debug_nan_min_step = int(os.environ.get("ROM_DEBUG_NAN_MIN_STEP", "0"))
rom_debug_nan_current_step = 0
engram_debug_backward = env_flag("ENGRAM_DEBUG_BACKWARD", False)
disable_embed_split = env_flag("DISABLE_EMBED_SPLIT", False)
rom_heads = int(os.environ.get("ROM_HEADS", "6"))
rom_key_dim = int(os.environ.get("ROM_KEY_DIM", "4"))
rom_value_dim = int(os.environ.get("ROM_VALUE_DIM", "16"))
rom_mqa = env_flag("ROM_MQA", False)
rom_layer_only = int(os.environ.get("ROM_LAYER_ONLY", "-1"))
rom_layers_raw = os.environ.get("ROM_LAYERS", "").strip()
rom_layers = tuple(int(x) for x in rom_layers_raw.replace(",", " ").split()) if rom_layers_raw else ()
rom_output_scale = float(os.environ.get("ROM_OUTPUT_SCALE", "1.0"))
rom_normalize_readout = env_flag("ROM_NORMALIZE_READOUT", False)
rom_readout_rms = float(os.environ.get("ROM_READOUT_RMS", "1.0"))
rom_read_mlp = env_flag("ROM_READ_MLP", False)
rom_read_mlp_hidden_mult = float(os.environ.get("ROM_READ_MLP_HIDDEN_MULT", "2.0"))
rom_read_mlp_lr_mul = float(os.environ.get("ROM_READ_MLP_LR_MUL", "0.5"))
rom_short_conv = env_flag("ROM_SHORT_CONV", False)
rom_short_conv_kernel = int(os.environ.get("ROM_SHORT_CONV_KERNEL", "4"))
rom_ema_smooth = env_flag("ROM_EMA_SMOOTH", False)
rom_ema_alpha = float(os.environ.get("ROM_EMA_ALPHA", "0.5"))
rom_ema_kernel = int(os.environ.get("ROM_EMA_KERNEL", "8"))
rom_state_sparse_embedding = env_flag("ROM_STATE_SPARSE_EMBEDDING", False)
rom_state_sparse_adam = env_flag("ROM_STATE_SPARSE_ADAM", False)
rom_state_sparse_sgd = env_flag("ROM_STATE_SPARSE_SGD", False)
rom_state_normwrite = env_flag("ROM_STATE_NORMWRITE", False)
rom_state_recovered_normwrite = env_flag("ROM_STATE_RECOVERED_NORMWRITE", False)
rom_sparse_sanitize = env_flag("ROM_SPARSE_SANITIZE", False)
rom_state_init_std = float(os.environ.get("ROM_STATE_INIT_STD", "0.0"))
rom_state_diag_init = env_flag("ROM_STATE_DIAG_INIT", False)
rom_state_frob_norm = float(os.environ.get("ROM_STATE_FROB_NORM", "0.0"))
rom_state_row_rms_cap = float(os.environ.get("ROM_STATE_ROW_RMS_CAP", str(rom_state_frob_norm)))
rom_state_write_rms = float(os.environ.get("ROM_STATE_WRITE_RMS", "0.001"))
rom_state_hit_rms_low = float(os.environ.get("ROM_STATE_HIT_RMS_LOW", "0.0"))
rom_state_hit_rms_high = float(os.environ.get("ROM_STATE_HIT_RMS_HIGH", "0.0"))
rom_state_hit_rms_knee = float(os.environ.get("ROM_STATE_HIT_RMS_KNEE", "4.0"))
rom_sparse_adam_lr_mul = float(os.environ.get("ROM_SPARSE_ADAM_LR_MUL", "5.0"))
rom_sparse_adam_beta2 = float(os.environ.get("ROM_SPARSE_ADAM_BETA2", "0.95"))
rom_sparse_row_scalar_adam = env_flag("ROM_SPARSE_ROW_SCALAR_ADAM", False)
rom_sparse_sgd_lr_mul = float(os.environ.get("ROM_SPARSE_SGD_LR_MUL", str(rom_sparse_adam_lr_mul)))
rom_sparse_sgd_momentum = float(os.environ.get("ROM_SPARSE_SGD_MOMENTUM", "0.9"))
rom_table_nonemb_mult = float(os.environ.get("ROM_TABLE_NONEMB_MULT", "0"))
engram_dim = int(os.environ.get("ENGRAM_DIM", "0"))
engram_heads = int(os.environ.get("ENGRAM_HEADS", "8"))
engram_max_ngram = int(os.environ.get("ENGRAM_MAX_NGRAM", "3"))
engram_ngram_row_factors_raw = os.environ.get("ENGRAM_NGRAM_ROW_FACTORS", "").strip()
engram_ngram_row_factors = tuple(float(x) for x in engram_ngram_row_factors_raw.replace(",", " ").split()) if engram_ngram_row_factors_raw else ()
engram_ngram_read_scales_raw = os.environ.get("ENGRAM_NGRAM_READ_SCALES", "").strip()
engram_ngram_read_scales = tuple(float(x) for x in engram_ngram_read_scales_raw.replace(",", " ").split()) if engram_ngram_read_scales_raw else ()
engram_ngram_read_scales_final_raw = os.environ.get("ENGRAM_NGRAM_READ_SCALES_FINAL", "").strip()
engram_ngram_read_scales_final = tuple(float(x) for x in engram_ngram_read_scales_final_raw.replace(",", " ").split()) if engram_ngram_read_scales_final_raw else engram_ngram_read_scales
engram_ngram_read_scale_schedule_steps = int(os.environ.get("ENGRAM_NGRAM_READ_SCALE_SCHEDULE_STEPS", "0"))
engram_ngram_read_scale_norm = env_flag("ENGRAM_NGRAM_READ_SCALE_NORM", True)
engram_hash_seed = int(os.environ.get("ENGRAM_HASH_SEED", "0"))
engram_avalanche_hash = env_flag("ENGRAM_AVALANCHE_HASH", False)
engram_layer_hashes = env_flag("ENGRAM_LAYER_HASHES", False)
engram_layer_readouts = env_flag("ENGRAM_LAYER_READOUTS", False)
engram_layer_readout_delta = env_flag("ENGRAM_LAYER_READOUT_DELTA", False)
engram_layer_readout_delta_scale = float(os.environ.get("ENGRAM_LAYER_READOUT_DELTA_SCALE", "1.0"))
engram_layer_readout_delta_scale_final = float(os.environ.get("ENGRAM_LAYER_READOUT_DELTA_SCALE_FINAL", str(engram_layer_readout_delta_scale)))
engram_layer_readout_delta_scale_schedule_steps = int(os.environ.get("ENGRAM_LAYER_READOUT_DELTA_SCALE_SCHEDULE_STEPS", "0"))
engram_layer_readout_delta_scale_schedule_start = int(os.environ.get("ENGRAM_LAYER_READOUT_DELTA_SCALE_SCHEDULE_START", "0"))
engram_layer_readout_delta_learned_scale = env_flag("ENGRAM_LAYER_READOUT_DELTA_LEARNED_SCALE", False)
engram_layer_readout_delta_learned_scale_init = float(os.environ.get("ENGRAM_LAYER_READOUT_DELTA_LEARNED_SCALE_INIT", "0.5"))
engram_layer_readout_delta_learned_scale_max = float(os.environ.get("ENGRAM_LAYER_READOUT_DELTA_LEARNED_SCALE_MAX", "1.0"))
engram_layer_readout_delta_learned_scale_lr_mul = float(os.environ.get("ENGRAM_LAYER_READOUT_DELTA_LEARNED_SCALE_LR_MUL", "0.25"))
engram_layer_partitions = env_flag("ENGRAM_LAYER_PARTITIONS", False)
engram_layer_signs = env_flag("ENGRAM_LAYER_SIGNS", False)
engram_layer_sign_scale = float(os.environ.get("ENGRAM_LAYER_SIGN_SCALE", "1.0"))
engram_layer_sign_scale_final = float(os.environ.get("ENGRAM_LAYER_SIGN_SCALE_FINAL", str(engram_layer_sign_scale)))
engram_layer_sign_scale_schedule_steps = int(os.environ.get("ENGRAM_LAYER_SIGN_SCALE_SCHEDULE_STEPS", "0"))
engram_layer_sign_scale_schedule_start = int(os.environ.get("ENGRAM_LAYER_SIGN_SCALE_SCHEDULE_START", "0"))
engram_layer_sign_aux_scale = float(os.environ.get("ENGRAM_LAYER_SIGN_AUX_SCALE", "0.0"))
engram_layer_sign_aux_scale_final = float(os.environ.get("ENGRAM_LAYER_SIGN_AUX_SCALE_FINAL", str(engram_layer_sign_aux_scale)))
engram_layer_sign_aux_scale_schedule_steps = int(os.environ.get("ENGRAM_LAYER_SIGN_AUX_SCALE_SCHEDULE_STEPS", "0"))
engram_layer_sign_aux_scale_schedule_start = int(os.environ.get("ENGRAM_LAYER_SIGN_AUX_SCALE_SCHEDULE_START", "0"))
engram_layer_row_signs = env_flag("ENGRAM_LAYER_ROW_SIGNS", False)
engram_layer_row_signs_aux_only = env_flag("ENGRAM_LAYER_ROW_SIGNS_AUX_ONLY", False)
engram_layer_row_sign_scale = float(os.environ.get("ENGRAM_LAYER_ROW_SIGN_SCALE", "1.0"))
engram_layer_row_sign_scale_final = float(os.environ.get("ENGRAM_LAYER_ROW_SIGN_SCALE_FINAL", str(engram_layer_row_sign_scale)))
engram_layer_row_sign_scale_schedule_steps = int(os.environ.get("ENGRAM_LAYER_ROW_SIGN_SCALE_SCHEDULE_STEPS", "0"))
engram_layer_row_sign_scale_schedule_start = int(os.environ.get("ENGRAM_LAYER_ROW_SIGN_SCALE_SCHEDULE_START", "0"))
engram_layer_partition_groups = int(os.environ.get("ENGRAM_LAYER_PARTITION_GROUPS", "0"))
engram_short_conv_kernel = int(os.environ.get("ENGRAM_SHORT_CONV_KERNEL", "4"))
engram_short_conv = env_flag("ENGRAM_SHORT_CONV", True)
engram_normalize_readout = env_flag("ENGRAM_NORMALIZE_READOUT", False)
engram_normalize_memory_heads = env_flag("ENGRAM_NORMALIZE_MEMORY_HEADS", False)
engram_detach_key_memory = env_flag("ENGRAM_DETACH_KEY_MEMORY", False)
engram_detach_value_memory = env_flag("ENGRAM_DETACH_VALUE_MEMORY", False)
engram_detach_memory_layers_raw = os.environ.get("ENGRAM_DETACH_MEMORY_LAYERS", "").strip()
engram_detach_memory_layers = tuple(int(x) for x in engram_detach_memory_layers_raw.replace(",", " ").split()) if engram_detach_memory_layers_raw else ()
engram_fixed_half_gate = env_flag("ENGRAM_FIXED_HALF_GATE", False)
engram_static_gate = env_flag("ENGRAM_STATIC_GATE", False)
engram_static_gate_init = float(os.environ.get("ENGRAM_STATIC_GATE_INIT", "0.0"))
engram_head_mix = env_flag("ENGRAM_HEAD_MIX", False)
engram_layer_head_mix = env_flag("ENGRAM_LAYER_HEAD_MIX", False)
engram_layer_head_mix_delta = env_flag("ENGRAM_LAYER_HEAD_MIX_DELTA", False)
engram_head_mix_init_raw = os.environ.get("ENGRAM_HEAD_MIX_INIT", "").strip()
engram_head_mix_init = tuple(float(x) for x in engram_head_mix_init_raw.replace(",", " ").split()) if engram_head_mix_init_raw else ()
engram_head_mix_freeze = env_flag("ENGRAM_HEAD_MIX_FREEZE", False)
engram_sketch_k = int(os.environ.get("ENGRAM_SKETCH_K", "1"))
engram_sketch_dim_signs = env_flag("ENGRAM_SKETCH_DIM_SIGNS", False)
engram_sketch_dim_sign_mode = os.environ.get("ENGRAM_SKETCH_DIM_SIGN_MODE", "random").strip().lower()
engram_sketch_scalar_signs = env_flag("ENGRAM_SKETCH_SCALAR_SIGNS", True)
engram_sketch_scalar_sign_mode = os.environ.get("ENGRAM_SKETCH_SCALAR_SIGN_MODE", "random").strip().lower()
engram_sketch_include_base = env_flag("ENGRAM_SKETCH_INCLUDE_BASE", False)
engram_sketch_aux_scale = float(os.environ.get("ENGRAM_SKETCH_AUX_SCALE", "1.0"))
engram_sketch_aux_scale_final = float(os.environ.get("ENGRAM_SKETCH_AUX_SCALE_FINAL", str(engram_sketch_aux_scale)))
engram_sketch_aux_scale_schedule_steps = int(os.environ.get("ENGRAM_SKETCH_AUX_SCALE_SCHEDULE_STEPS", "0"))
engram_sketch_aux_scale_schedule_start = int(os.environ.get("ENGRAM_SKETCH_AUX_SCALE_SCHEDULE_START", "0"))
engram_sketch_aux_learned_scale = env_flag("ENGRAM_SKETCH_AUX_LEARNED_SCALE", False)
engram_sketch_aux_learned_scale_init = float(os.environ.get("ENGRAM_SKETCH_AUX_LEARNED_SCALE_INIT", "1.0"))
engram_sketch_aux_learned_scale_max = float(os.environ.get("ENGRAM_SKETCH_AUX_LEARNED_SCALE_MAX", "2.0"))
engram_sketch_aux_learned_scale_lr_mul = float(os.environ.get("ENGRAM_SKETCH_AUX_LEARNED_SCALE_LR_MUL", "0.25"))
engram_sketch_slot_readout = env_flag("ENGRAM_SKETCH_SLOT_READOUT", False)
engram_sketch_slot_attention = env_flag("ENGRAM_SKETCH_SLOT_ATTENTION", False)
engram_sketch_slot_mix = env_flag("ENGRAM_SKETCH_SLOT_MIX", False)
engram_sketch_combine_mix = env_flag("ENGRAM_SKETCH_COMBINE_MIX", False)
engram_sketch_combine_mix_mode = os.environ.get("ENGRAM_SKETCH_COMBINE_MIX_MODE", "softmax").strip().lower()
engram_sketch_combine_mix_max_dev = float(os.environ.get("ENGRAM_SKETCH_COMBINE_MIX_MAX_DEV", "0.1"))
engram_sketch_mix_lr_mul = float(os.environ.get("ENGRAM_SKETCH_MIX_LR_MUL", "0.5"))
engram_sketch_hit_hist_base_only = env_flag("ENGRAM_SKETCH_HIT_HIST_BASE_ONLY", False)
engram_superpose_k = int(os.environ.get("ENGRAM_SUPERPOSE_K", "1"))
engram_superpose_include_base = env_flag("ENGRAM_SUPERPOSE_INCLUDE_BASE", False)
engram_superpose_aux_scale = float(os.environ.get("ENGRAM_SUPERPOSE_AUX_SCALE", "1.0"))
engram_superpose_aux_scale_final = float(os.environ.get("ENGRAM_SUPERPOSE_AUX_SCALE_FINAL", str(engram_superpose_aux_scale)))
engram_superpose_aux_scale_schedule_steps = int(os.environ.get("ENGRAM_SUPERPOSE_AUX_SCALE_SCHEDULE_STEPS", "0"))
engram_superpose_aux_scale_schedule_start = int(os.environ.get("ENGRAM_SUPERPOSE_AUX_SCALE_SCHEDULE_START", "0"))
engram_superpose_normalize = env_flag("ENGRAM_SUPERPOSE_NORMALIZE", True)
engram_manual_sparse_coalesce = env_flag("ENGRAM_MANUAL_SPARSE_COALESCE", False)
engram_manual_sparse_coalesce_start = int(os.environ.get("ENGRAM_MANUAL_SPARSE_COALESCE_START", "0"))
engram_head_dropout = float(os.environ.get("ENGRAM_HEAD_DROPOUT", "0.0"))
engram_head_dropout_final = float(os.environ.get("ENGRAM_HEAD_DROPOUT_FINAL", str(engram_head_dropout)))
engram_head_dropout_schedule_steps = int(os.environ.get("ENGRAM_HEAD_DROPOUT_SCHEDULE_STEPS", "0"))
engram_head_dropout_current = engram_head_dropout
engram_output_dropout = float(os.environ.get("ENGRAM_OUTPUT_DROPOUT", "0.0"))
engram_output_dropout_final = float(os.environ.get("ENGRAM_OUTPUT_DROPOUT_FINAL", str(engram_output_dropout)))
engram_output_dropout_schedule_steps = int(os.environ.get("ENGRAM_OUTPUT_DROPOUT_SCHEDULE_STEPS", "0"))
engram_output_dropout_schedule_start = int(os.environ.get("ENGRAM_OUTPUT_DROPOUT_SCHEDULE_START", "0"))
engram_output_dropout_current = engram_output_dropout
engram_output_grad_metrics = env_flag("ENGRAM_OUTPUT_GRAD_METRICS", False)
engram_read_hit_scale_exponent = float(os.environ.get("ENGRAM_READ_HIT_SCALE_EXPONENT", "0.0"))
engram_read_hit_scale_offset = float(os.environ.get("ENGRAM_READ_HIT_SCALE_OFFSET", "1.0"))
engram_read_hit_scale_min = float(os.environ.get("ENGRAM_READ_HIT_SCALE_MIN", "0.0"))
engram_read_hit_scale_max = float(os.environ.get("ENGRAM_READ_HIT_SCALE_MAX", "inf"))
engram_read_hit_scale_norm_mean = env_flag("ENGRAM_READ_HIT_SCALE_NORM_MEAN", False)
engram_hit_dropout = float(os.environ.get("ENGRAM_HIT_DROPOUT", "0.0"))
engram_hit_dropout_final = float(os.environ.get("ENGRAM_HIT_DROPOUT_FINAL", str(engram_hit_dropout)))
engram_hit_dropout_schedule_steps = int(os.environ.get("ENGRAM_HIT_DROPOUT_SCHEDULE_STEPS", "0"))
engram_hit_dropout_schedule_start = int(os.environ.get("ENGRAM_HIT_DROPOUT_SCHEDULE_START", "0"))
engram_hit_dropout_decay_final = float(os.environ.get("ENGRAM_HIT_DROPOUT_DECAY_FINAL", str(engram_hit_dropout_final)))
engram_hit_dropout_decay_steps = int(os.environ.get("ENGRAM_HIT_DROPOUT_DECAY_STEPS", "0"))
engram_hit_dropout_decay_start = int(os.environ.get("ENGRAM_HIT_DROPOUT_DECAY_START", "0"))
engram_hit_dropout_min_hits = int(os.environ.get("ENGRAM_HIT_DROPOUT_MIN_HITS", "0"))
engram_hit_dropout_invert_scale = env_flag("ENGRAM_HIT_DROPOUT_INVERT_SCALE", True)
engram_hot_split = env_flag("ENGRAM_HOT_SPLIT", False)
engram_hot_split_value_only = env_flag("ENGRAM_HOT_SPLIT_VALUE_ONLY", False)
engram_hot_split_min_hits = int(os.environ.get("ENGRAM_HOT_SPLIT_MIN_HITS", "1024"))
engram_hot_split_aux_scale = float(os.environ.get("ENGRAM_HOT_SPLIT_AUX_SCALE", "1.0"))
engram_hot_split_aux_scale_final = float(os.environ.get("ENGRAM_HOT_SPLIT_AUX_SCALE_FINAL", str(engram_hot_split_aux_scale)))
engram_hot_split_aux_scale_schedule_steps = int(os.environ.get("ENGRAM_HOT_SPLIT_AUX_SCALE_SCHEDULE_STEPS", "0"))
engram_hot_split_aux_scale_schedule_start = int(os.environ.get("ENGRAM_HOT_SPLIT_AUX_SCALE_SCHEDULE_START", "0"))
engram_hot_split_aux_slots = int(os.environ.get("ENGRAM_HOT_SPLIT_AUX_SLOTS", "1"))
engram_hot_split_ramp_steps = int(os.environ.get("ENGRAM_HOT_SPLIT_RAMP_STEPS", "0"))
engram_hot_split_detach_aux = env_flag("ENGRAM_HOT_SPLIT_DETACH_AUX", False)
engram_hot_split_dedup_aux = env_flag("ENGRAM_HOT_SPLIT_DEDUP_AUX", False)
engram_hot_split_train_only = env_flag("ENGRAM_HOT_SPLIT_TRAIN_ONLY", False)
engram_hit_hist_kinds_raw = os.environ.get("ENGRAM_HIT_HIST_KINDS", "lm").strip().lower()
engram_hit_hist_kinds = tuple(x for x in engram_hit_hist_kinds_raw.replace(",", " ").split() if x)
engram_pad_id = int(os.environ.get("ENGRAM_PAD_ID", str(BOS_ID if "BOS_ID" in globals() else 50256)))
engram_per_head = env_flag("ENGRAM_PER_HEAD", False)
engram_canonicalize = env_flag("ENGRAM_CANONICALIZE", False)
engram_latent = env_flag("ENGRAM_LATENT", False)
engram_latent_quantizer = os.environ.get("ENGRAM_LATENT_QUANTIZER", "fsq").lower()
engram_latent_fsq_levels_raw = os.environ.get("ENGRAM_LATENT_FSQ_LEVELS", "8,8,8,8,8,8,8,3")
engram_latent_fsq_levels = tuple(int(x) for x in engram_latent_fsq_levels_raw.replace(",", " ").split() if x)
engram_latent_fsq_eps = float(os.environ.get("ENGRAM_LATENT_FSQ_EPS", "1e-3"))
engram_latent_bsq_bits = int(os.environ.get("ENGRAM_LATENT_BSQ_BITS", "0"))
engram_latent_rows_per_head = int(os.environ.get("ENGRAM_LATENT_ROWS_PER_HEAD", "0"))
engram_latent_input_scale = float(os.environ.get("ENGRAM_LATENT_INPUT_SCALE", "1.0"))
engram_latent_ste_scale = float(os.environ.get("ENGRAM_LATENT_STE_SCALE", "0.0"))
engram_latent_mix_ngram = env_flag("ENGRAM_LATENT_MIX_NGRAM", False)
engram_latent_aux_readout = env_flag("ENGRAM_LATENT_AUX_READOUT", False)
engram_latent_aux_scale = float(os.environ.get("ENGRAM_LATENT_AUX_SCALE", "1.0"))
engram_latent_aux_scale_final = float(os.environ.get("ENGRAM_LATENT_AUX_SCALE_FINAL", str(engram_latent_aux_scale)))
engram_latent_aux_scale_schedule_steps = int(os.environ.get("ENGRAM_LATENT_AUX_SCALE_SCHEDULE_STEPS", "0"))
engram_latent_aux_scale_schedule_start = int(os.environ.get("ENGRAM_LATENT_AUX_SCALE_SCHEDULE_START", "0"))
engram_latent_pkm_subkeys = int(os.environ.get("ENGRAM_LATENT_PKM_SUBKEYS", "256"))
engram_latent_pkm_key_dim = int(os.environ.get("ENGRAM_LATENT_PKM_KEY_DIM", "32"))
engram_latent_pkm_topk = int(os.environ.get("ENGRAM_LATENT_PKM_TOPK", "2"))
engram_cache_recon = env_flag("ENGRAM_CACHE_RECON", False)
engram_cache_recon_source_layer = int(os.environ.get("ENGRAM_CACHE_RECON_SOURCE_LAYER", "2"))
engram_cache_recon_target_layer = int(os.environ.get("ENGRAM_CACHE_RECON_TARGET_LAYER", "8"))
engram_cache_recon_weight = float(os.environ.get("ENGRAM_CACHE_RECON_WEIGHT", "1.0"))
engram_cache_recon_mode = os.environ.get("ENGRAM_CACHE_RECON_MODE", "mse").lower()
engram_cache_readout = env_flag("ENGRAM_CACHE_READOUT", False)
engram_cache_cfg_scale = float(os.environ.get("ENGRAM_CACHE_CFG_SCALE", "0.0"))
engram_cache_detach_memory = env_flag("ENGRAM_CACHE_DETACH_MEMORY", False)
engram_cache_learned_cfg = env_flag("ENGRAM_CACHE_LEARNED_CFG", False)
engram_mhc = env_flag("ENGRAM_MHC", False)
engram_mhc_init = float(os.environ.get("ENGRAM_MHC_INIT", "0.95"))
engram_mhc_streams = int(os.environ.get("ENGRAM_MHC_STREAMS", "2"))
engram_mhc_identity = env_flag("ENGRAM_MHC_IDENTITY", False)
engram_mhc_dynamic = env_flag("ENGRAM_MHC_DYNAMIC", True)
engram_mhc_delta = env_flag("ENGRAM_MHC_DELTA", True)
engram_attnres_merge = env_flag("ENGRAM_ATTNRES_MERGE", False)
engram_attnres_merge_gain = float(os.environ.get("ENGRAM_ATTNRES_MERGE_GAIN", "2.0"))
engram_attnres_gain_warmup_steps = int(os.environ.get("ENGRAM_ATTNRES_GAIN_WARMUP_STEPS", "0"))
engram_attnres_merge_gain_current = engram_attnres_merge_gain
engram_attnres_metrics = env_flag("ENGRAM_ATTNRES_METRICS", True)
engram_attnres_direct_residual = env_flag("ENGRAM_ATTNRES_DIRECT_RESIDUAL", False)
engram_attnres_direct_init = float(os.environ.get("ENGRAM_ATTNRES_DIRECT_INIT", "0.0"))
engram_attnres_direct_layers_raw = os.environ.get("ENGRAM_ATTNRES_DIRECT_LAYERS", "").strip()
engram_attnres_direct_layers = tuple(int(x) for x in engram_attnres_direct_layers_raw.replace(",", " ").split() if x)
engram_attnres_layer_gain = env_flag("ENGRAM_ATTNRES_LAYER_GAIN", False)
engram_attnres_layer_gain_init = float(os.environ.get("ENGRAM_ATTNRES_LAYER_GAIN_INIT", "0.0"))
engram_attnres_extra_source_layer = int(os.environ.get("ENGRAM_ATTNRES_EXTRA_SOURCE_LAYER", "-1"))
engram_attnres_extra_target_layer = int(os.environ.get("ENGRAM_ATTNRES_EXTRA_TARGET_LAYER", "-1"))
engram_attnres_extra_bias_init = float(os.environ.get("ENGRAM_ATTNRES_EXTRA_BIAS_INIT", "0.0"))
engram_attnres_extra_scale = float(os.environ.get("ENGRAM_ATTNRES_EXTRA_SCALE", "1.0"))
engram_attnres_extra_scale_final = float(os.environ.get("ENGRAM_ATTNRES_EXTRA_SCALE_FINAL", str(engram_attnres_extra_scale)))
engram_attnres_extra_scale_schedule_steps = int(os.environ.get("ENGRAM_ATTNRES_EXTRA_SCALE_SCHEDULE_STEPS", "0"))
engram_attnres_extra_scale_schedule_start = int(os.environ.get("ENGRAM_ATTNRES_EXTRA_SCALE_SCHEDULE_START", "0"))
engram_bank_attnres = env_flag("ENGRAM_BANK_ATTNRES", False)
engram_attnres_delta = env_flag("ENGRAM_ATTNRES_DELTA", False)
engram_store_dim = int(os.environ.get("ENGRAM_STORE_DIM", "0"))
engram_untied_proj = env_flag("ENGRAM_UNTIED_PROJ", False)
engram_init_std = float(os.environ.get("ENGRAM_INIT_STD", "1.0"))
engram_init_zero = env_flag("ENGRAM_INIT_ZERO", False)
engram_freeze_memory = env_flag("ENGRAM_FREEZE_MEMORY", False)
engram_shadow_grad = env_flag("ENGRAM_SHADOW_GRAD", False)
engram_shadow_scale = float(os.environ.get("ENGRAM_SHADOW_SCALE", "0.05"))
engram_shadow_write_rms = float(os.environ.get("ENGRAM_SHADOW_WRITE_RMS", "0.05"))
engram_shadow_write_alpha = float(os.environ.get("ENGRAM_SHADOW_WRITE_ALPHA", "1.0"))
engram_shadow_decay = float(os.environ.get("ENGRAM_SHADOW_DECAY", "1.0"))
engram_shadow_row_rms_cap = float(os.environ.get("ENGRAM_SHADOW_ROW_RMS_CAP", "8.0"))
engram_shadow_hit_max = int(os.environ.get("ENGRAM_SHADOW_HIT_MAX", "0"))
engram_shadow_only = env_flag("ENGRAM_SHADOW_ONLY", False)
engram_adam_every_step = env_flag("ENGRAM_ADAM_EVERY_STEP", False)
engram_lr_mul = float(os.environ.get("ENGRAM_LR_MUL", "5.0"))
engram_lr_floor = float(os.environ.get("ENGRAM_LR_FLOOR", "0.0"))
engram_update_metrics = env_flag("ENGRAM_UPDATE_METRICS", False)
engram_update_metrics_every = int(os.environ.get("ENGRAM_UPDATE_METRICS_EVERY", "1"))
engram_touched_update_metrics = env_flag("ENGRAM_TOUCHED_UPDATE_METRICS", False)
engram_touched_update_metrics_chunk = int(os.environ.get("ENGRAM_TOUCHED_UPDATE_METRICS_CHUNK", "1048576"))
engram_disable_compile_region = env_flag("ENGRAM_DISABLE_COMPILE_REGION", False)
profile_events = env_flag("PROFILE_EVENTS", False)
profile_events_start = int(os.environ.get("PROFILE_EVENTS_START", "10"))
profile_events_steps = int(os.environ.get("PROFILE_EVENTS_STEPS", "3"))
torch_profiler = env_flag("TORCH_PROFILER", False)
torch_profiler_start = int(os.environ.get("TORCH_PROFILER_START", "70"))
torch_profiler_steps = int(os.environ.get("TORCH_PROFILER_STEPS", "1"))
torch_profiler_record_shapes = env_flag("TORCH_PROFILER_RECORD_SHAPES", False)
torch_profiler_profile_memory = env_flag("TORCH_PROFILER_PROFILE_MEMORY", False)
torch_profiler_with_stack = env_flag("TORCH_PROFILER_WITH_STACK", False)
engram_sparse_adam = env_flag("ENGRAM_SPARSE_ADAM", False)
engram_sparse_vector_adam = env_flag("ENGRAM_SPARSE_VECTOR_ADAM", False)
engram_sparse_scalar_adam = env_flag("ENGRAM_SPARSE_SCALAR_ADAM", False)
engram_sparse_row_adagrad = env_flag("ENGRAM_SPARSE_ROW_ADAGRAD", False)
engram_sparse_adam_tail_steps = int(os.environ.get("ENGRAM_SPARSE_ADAM_TAIL_STEPS", "0"))
engram_sparse_adam_tail_scale = float(os.environ.get("ENGRAM_SPARSE_ADAM_TAIL_SCALE", "1.0"))
engram_sparse_adam_beta1 = float(os.environ.get("ENGRAM_SPARSE_ADAM_BETA1", "0.75"))
engram_sparse_adam_beta2 = float(os.environ.get("ENGRAM_SPARSE_ADAM_BETA2", "0.95"))
engram_sparse_weight_decay = float(os.environ.get("ENGRAM_SPARSE_WEIGHT_DECAY", "0.0"))
engram_sparse_extra_steps = int(os.environ.get("ENGRAM_SPARSE_EXTRA_STEPS", "0"))
engram_sparse_hit_lr = env_flag("ENGRAM_SPARSE_HIT_LR", False)
engram_sparse_fal = env_flag("ENGRAM_SPARSE_FAL", False)
engram_sparse_ifal = env_flag("ENGRAM_SPARSE_IFAL", False)
engram_sparse_batch_freq_norm = env_flag("ENGRAM_SPARSE_BATCH_FREQ_NORM", False)
engram_sparse_hit_lr_exponent = float(os.environ.get("ENGRAM_SPARSE_HIT_LR_EXPONENT", "0.5"))
engram_sparse_hit_lr_min = float(os.environ.get("ENGRAM_SPARSE_HIT_LR_MIN", "0.0"))
engram_sparse_hit_lr_max = float(os.environ.get("ENGRAM_SPARSE_HIT_LR_MAX", "inf"))
engram_sparse_hit_lr_blend = float(os.environ.get("ENGRAM_SPARSE_HIT_LR_BLEND", "1.0"))
engram_sparse_hit_lr_blend_final = float(os.environ.get("ENGRAM_SPARSE_HIT_LR_BLEND_FINAL", str(engram_sparse_hit_lr_blend)))
engram_sparse_hit_lr_blend_schedule_steps = int(os.environ.get("ENGRAM_SPARSE_HIT_LR_BLEND_SCHEDULE_STEPS", "0"))
engram_sparse_hit_lr_blend_schedule_start = int(os.environ.get("ENGRAM_SPARSE_HIT_LR_BLEND_SCHEDULE_START", "0"))
engram_sparse_sanitize = env_flag("ENGRAM_SPARSE_SANITIZE", False)
engram_sparse_param_clamp = float(os.environ.get("ENGRAM_SPARSE_PARAM_CLAMP", "0.0"))
engram_sparse_row_rms_cap = float(os.environ.get("ENGRAM_SPARSE_ROW_RMS_CAP", "0.0"))
engram_sparse_row_rms_floor = float(os.environ.get("ENGRAM_SPARSE_ROW_RMS_FLOOR", "0.0"))
engram_sparse_row_rms_floor_hit_max = float(os.environ.get("ENGRAM_SPARSE_ROW_RMS_FLOOR_HIT_MAX", "0.0"))
engram_sparse_row_rms_norm = float(os.environ.get("ENGRAM_SPARSE_ROW_RMS_NORM", "0.0"))
engram_sparse_row_decay = float(os.environ.get("ENGRAM_SPARSE_ROW_DECAY", "0.0"))
engram_sparse_grad_coalesce_hook = env_flag("ENGRAM_SPARSE_GRAD_COALESCE_HOOK", False)
engram_sparse_grad_coalesce_hook_start = int(os.environ.get("ENGRAM_SPARSE_GRAD_COALESCE_HOOK_START", "0"))
engram_memory_fp32 = env_flag("ENGRAM_MEMORY_FP32", False)
engram_offload = env_flag("ENGRAM_OFFLOAD", False)
engram_offload_pin = env_flag("ENGRAM_OFFLOAD_PIN", False)
engram_offload_pin_staging = env_flag("ENGRAM_OFFLOAD_PIN_STAGING", False)
engram_offload_prefetch = env_flag("ENGRAM_OFFLOAD_PREFETCH", True)
engram_offload_async_adam = env_flag("ENGRAM_OFFLOAD_ASYNC_ADAM", False)
engram_offload_gpu_adam = env_flag("ENGRAM_OFFLOAD_GPU_ADAM", False)
engram_offload_prefetch_moments = env_flag("ENGRAM_OFFLOAD_PREFETCH_MOMENTS", False)
engram_offload_merge_pending = env_flag("ENGRAM_OFFLOAD_MERGE_PENDING", True)
engram_offload_lazy_moments = env_flag("ENGRAM_OFFLOAD_LAZY_MOMENTS", False)
engram_offload_second_moment = env_flag("ENGRAM_OFFLOAD_SECOND_MOMENT", True)
engram_offload_moment_dtype_name = os.environ.get("ENGRAM_OFFLOAD_MOMENT_DTYPE", "float32").lower()
if engram_offload_moment_dtype_name in ("float", "fp32", "float32"):
    engram_offload_moment_dtype = torch.float32
    engram_offload_moment_dtype_name = "float32"
elif engram_offload_moment_dtype_name in ("bfloat16", "bf16"):
    engram_offload_moment_dtype = torch.bfloat16
    engram_offload_moment_dtype_name = "bfloat16"
else:
    raise ValueError(f"Unsupported ENGRAM_OFFLOAD_MOMENT_DTYPE={engram_offload_moment_dtype_name!r}")
engram_offload_seed = int(os.environ.get("ENGRAM_OFFLOAD_SEED", "1234"))
if engram_offload:
    compile_model = False
engram_analyze = env_flag("ENGRAM_ANALYZE", False)
engram_analyze_ckpt = os.environ.get("ENGRAM_ANALYZE_CKPT", "")
engram_analyze_tokens = int(os.environ.get("ENGRAM_ANALYZE_TOKENS", str(256 * 1024)))
engram_analyze_topk = int(os.environ.get("ENGRAM_ANALYZE_TOPK", "20"))
engram_analyze_out = os.environ.get("ENGRAM_ANALYZE_OUT", "")
engram_analyze_prompts_file = os.environ.get("ENGRAM_ANALYZE_PROMPTS_FILE", "").strip()
engram_analyze_html = os.environ.get("ENGRAM_ANALYZE_HTML", "").strip()
engram_hit_hist = env_flag("ENGRAM_HIT_HIST", False)
engram_save_hit_hist = env_flag("ENGRAM_SAVE_HIT_HIST", False)
engram_hit_hist_load = os.environ.get("ENGRAM_HIT_HIST_LOAD", "").strip()
engram_mask_unhit_eval = env_flag("ENGRAM_MASK_UNHIT_EVAL", False)
engram_mask_unhit_eval_mode = os.environ.get("ENGRAM_MASK_UNHIT_EVAL_MODE", "zero").lower()
engram_mask_hit_min_eval = int(os.environ.get("ENGRAM_MASK_HIT_MIN_EVAL", "0"))
engram_mask_hit_max_eval = int(os.environ.get("ENGRAM_MASK_HIT_MAX_EVAL", "0"))
engram_mask_hit_invert_eval = env_flag("ENGRAM_MASK_HIT_INVERT_EVAL", False)
engram_eval_hit_scale = float(os.environ.get("ENGRAM_EVAL_HIT_SCALE", "1.0"))
engram_eval_hit_scale_min = int(os.environ.get("ENGRAM_EVAL_HIT_SCALE_MIN", "0"))
engram_eval_hit_scale_max = int(os.environ.get("ENGRAM_EVAL_HIT_SCALE_MAX", "0"))
engram_eval_hit_scale_invert = env_flag("ENGRAM_EVAL_HIT_SCALE_INVERT", False)
engram_eval_ckpt = os.environ.get("ENGRAM_EVAL_CKPT", "").strip()
engram_eval_out = os.environ.get("ENGRAM_EVAL_OUT", "").strip()
rom_debug_grad_steps = {int(x) for x in os.environ.get("ROM_DEBUG_GRAD_STEPS", "").replace(",", " ").split() if x}
if engram_bigram and rom_bigram:
    raise ValueError("ENGRAM_BIGRAM and ROM_BIGRAM are mutually exclusive")
if adam_embed_vector_adam and adam_embed_scalar_adam:
    raise ValueError("Choose at most one of ADAM_EMBED_VECTOR_ADAM or ADAM_EMBED_SCALAR_ADAM")
if rom_single_token and rom_token:
    raise ValueError("ROM_SINGLE_TOKEN and ROM_TOKEN are mutually exclusive")
if rom_token and not rom_bigram:
    raise ValueError("ROM_TOKEN requires ROM_BIGRAM=1")
if engram_bigram and engram_max_ngram < 2:
    raise ValueError("ENGRAM_MAX_NGRAM must be >= 2")
if engram_ngram_row_factors:
    if len(engram_ngram_row_factors) != max(0, engram_max_ngram - 1):
        raise ValueError("ENGRAM_NGRAM_ROW_FACTORS must have one positive value per n-gram head")
    if min(engram_ngram_row_factors) <= 0:
        raise ValueError("ENGRAM_NGRAM_ROW_FACTORS entries must be positive")
if engram_ngram_read_scales:
    if len(engram_ngram_read_scales) != max(0, engram_max_ngram - 1):
        raise ValueError("ENGRAM_NGRAM_READ_SCALES must have one non-negative value per n-gram head")
    if min(engram_ngram_read_scales) < 0:
        raise ValueError("ENGRAM_NGRAM_READ_SCALES entries must be non-negative")
    if len(engram_ngram_read_scales_final) != len(engram_ngram_read_scales):
        raise ValueError("ENGRAM_NGRAM_READ_SCALES_FINAL must match ENGRAM_NGRAM_READ_SCALES length")
    if min(engram_ngram_read_scales_final) < 0:
        raise ValueError("ENGRAM_NGRAM_READ_SCALES_FINAL entries must be non-negative")
    if max(engram_ngram_read_scales) == 0 or max(engram_ngram_read_scales_final) == 0:
        raise ValueError("ENGRAM_NGRAM_READ_SCALES cannot be all zero")
    if not engram_per_head:
        raise ValueError("ENGRAM_NGRAM_READ_SCALES requires ENGRAM_PER_HEAD=1")
if engram_ngram_read_scale_schedule_steps < 0:
    raise ValueError("ENGRAM_NGRAM_READ_SCALE_SCHEDULE_STEPS must be non-negative")
if engram_bigram and engram_heads <= 0:
    raise ValueError("ENGRAM_HEADS must be positive")
if engram_head_mix_init and not (engram_head_mix or engram_layer_head_mix):
    raise ValueError("ENGRAM_HEAD_MIX_INIT requires ENGRAM_HEAD_MIX=1 or ENGRAM_LAYER_HEAD_MIX=1")
if engram_head_mix_freeze and not (engram_head_mix or engram_layer_head_mix):
    raise ValueError("ENGRAM_HEAD_MIX_FREEZE requires ENGRAM_HEAD_MIX=1 or ENGRAM_LAYER_HEAD_MIX=1")
if engram_layer_head_mix_delta:
    if not engram_head_mix:
        raise ValueError("ENGRAM_LAYER_HEAD_MIX_DELTA requires ENGRAM_HEAD_MIX=1")
    if engram_layer_head_mix:
        raise ValueError("ENGRAM_LAYER_HEAD_MIX_DELTA is mutually exclusive with ENGRAM_LAYER_HEAD_MIX")
if engram_latent and not engram_bigram:
    raise ValueError("ENGRAM_LATENT requires ENGRAM_BIGRAM=1")
if engram_latent_mix_ngram and not engram_latent:
    raise ValueError("ENGRAM_LATENT_MIX_NGRAM requires ENGRAM_LATENT=1")
if engram_latent_quantizer not in ("fsq", "bsq", "pkm"):
    raise ValueError("ENGRAM_LATENT_QUANTIZER must be 'fsq', 'bsq', or 'pkm'")
if engram_latent and engram_latent_quantizer == "fsq" and not engram_latent_fsq_levels:
    raise ValueError("ENGRAM_LATENT_FSQ_LEVELS must specify at least one level")
if engram_latent and engram_latent_quantizer == "fsq" and any(level < 2 for level in engram_latent_fsq_levels):
    raise ValueError("ENGRAM_LATENT_FSQ_LEVELS entries must be >= 2")
if engram_latent_fsq_eps < 0 or engram_latent_fsq_eps >= 1:
    raise ValueError("ENGRAM_LATENT_FSQ_EPS must be in [0, 1)")
if engram_latent_bsq_bits < 0:
    raise ValueError("ENGRAM_LATENT_BSQ_BITS must be non-negative")
if engram_latent_rows_per_head < 0:
    raise ValueError("ENGRAM_LATENT_ROWS_PER_HEAD must be non-negative")
if engram_latent_input_scale <= 0:
    raise ValueError("ENGRAM_LATENT_INPUT_SCALE must be positive")
if engram_latent_ste_scale < 0:
    raise ValueError("ENGRAM_LATENT_STE_SCALE must be non-negative")
if engram_latent_aux_readout and not engram_latent:
    raise ValueError("ENGRAM_LATENT_AUX_READOUT requires ENGRAM_LATENT=1")
if engram_latent_aux_scale < 0:
    raise ValueError("ENGRAM_LATENT_AUX_SCALE must be non-negative")
if engram_latent_aux_scale_final < 0:
    raise ValueError("ENGRAM_LATENT_AUX_SCALE_FINAL must be non-negative")
if engram_latent_aux_scale_schedule_steps < 0:
    raise ValueError("ENGRAM_LATENT_AUX_SCALE_SCHEDULE_STEPS must be non-negative")
if engram_latent_aux_scale_schedule_start < 0:
    raise ValueError("ENGRAM_LATENT_AUX_SCALE_SCHEDULE_START must be non-negative")
if engram_latent_pkm_subkeys <= 1:
    raise ValueError("ENGRAM_LATENT_PKM_SUBKEYS must be > 1")
if engram_latent_pkm_key_dim <= 0:
    raise ValueError("ENGRAM_LATENT_PKM_KEY_DIM must be positive")
if engram_latent_pkm_topk <= 0 or engram_latent_pkm_topk > engram_latent_pkm_subkeys:
    raise ValueError("ENGRAM_LATENT_PKM_TOPK must be in [1, ENGRAM_LATENT_PKM_SUBKEYS]")
if engram_cache_recon and not engram_bigram:
    raise ValueError("ENGRAM_CACHE_RECON requires ENGRAM_BIGRAM=1")
if engram_cache_recon_mode not in ("mse", "cosine", "direction_mse"):
    raise ValueError("ENGRAM_CACHE_RECON_MODE must be one of: mse, cosine, direction_mse")
if engram_cache_readout and not engram_bigram:
    raise ValueError("ENGRAM_CACHE_READOUT requires ENGRAM_BIGRAM=1")
if engram_cache_readout and not engram_cache_recon:
    raise ValueError("ENGRAM_CACHE_READOUT currently requires ENGRAM_CACHE_RECON=1")
if engram_mhc and not engram_bigram:
    raise ValueError("ENGRAM_MHC requires ENGRAM_BIGRAM=1")
if engram_attnres_extra_scale < 0:
    raise ValueError("ENGRAM_ATTNRES_EXTRA_SCALE must be non-negative")
if engram_attnres_extra_scale_final < 0:
    raise ValueError("ENGRAM_ATTNRES_EXTRA_SCALE_FINAL must be non-negative")
if engram_attnres_extra_scale_schedule_steps < 0:
    raise ValueError("ENGRAM_ATTNRES_EXTRA_SCALE_SCHEDULE_STEPS must be non-negative")
if engram_attnres_extra_scale_schedule_start < 0:
    raise ValueError("ENGRAM_ATTNRES_EXTRA_SCALE_SCHEDULE_START must be non-negative")
if engram_mhc_init <= 0.0 or engram_mhc_init >= 1.0:
    raise ValueError("ENGRAM_MHC_INIT must be in (0, 1)")
if engram_mhc and engram_mhc_streams < 2:
    raise ValueError("ENGRAM_MHC_STREAMS must be >= 2")
if engram_mhc_identity and not engram_mhc:
    raise ValueError("ENGRAM_MHC_IDENTITY requires ENGRAM_MHC=1")
if engram_attnres_merge and not (engram_bigram or rom_single_token):
    raise ValueError("ENGRAM_ATTNRES_MERGE requires ENGRAM_BIGRAM=1 or ROM_SINGLE_TOKEN=1")
if engram_attnres_merge and engram_mhc:
    raise ValueError("ENGRAM_ATTNRES_MERGE is intended to replace ENGRAM_MHC; disable ENGRAM_MHC")
if engram_attnres_merge_gain <= 0:
    raise ValueError("ENGRAM_ATTNRES_MERGE_GAIN must be positive")
if engram_attnres_gain_warmup_steps < 0:
    raise ValueError("ENGRAM_ATTNRES_GAIN_WARMUP_STEPS must be non-negative")
if engram_attnres_direct_residual and not engram_attnres_merge:
    raise ValueError("ENGRAM_ATTNRES_DIRECT_RESIDUAL requires ENGRAM_ATTNRES_MERGE=1")
if engram_bank_attnres and not engram_attnres_merge:
    raise ValueError("ENGRAM_BANK_ATTNRES requires ENGRAM_ATTNRES_MERGE=1")
if engram_bank_attnres and not engram_per_head:
    raise ValueError("ENGRAM_BANK_ATTNRES requires ENGRAM_PER_HEAD=1")
if engram_attnres_delta and not engram_attnres_merge:
    raise ValueError("ENGRAM_ATTNRES_DELTA requires ENGRAM_ATTNRES_MERGE=1")
if engram_head_dropout < 0 or engram_head_dropout >= 1:
    raise ValueError("ENGRAM_HEAD_DROPOUT must be in [0, 1)")
if engram_head_dropout_final < 0 or engram_head_dropout_final >= 1:
    raise ValueError("ENGRAM_HEAD_DROPOUT_FINAL must be in [0, 1)")
if engram_head_dropout_schedule_steps < 0:
    raise ValueError("ENGRAM_HEAD_DROPOUT_SCHEDULE_STEPS must be non-negative")
if engram_output_dropout < 0 or engram_output_dropout >= 1:
    raise ValueError("ENGRAM_OUTPUT_DROPOUT must be in [0, 1)")
if engram_output_dropout_final < 0 or engram_output_dropout_final >= 1:
    raise ValueError("ENGRAM_OUTPUT_DROPOUT_FINAL must be in [0, 1)")
if engram_output_dropout_schedule_steps < 0:
    raise ValueError("ENGRAM_OUTPUT_DROPOUT_SCHEDULE_STEPS must be non-negative")
if engram_output_dropout_schedule_start < 0:
    raise ValueError("ENGRAM_OUTPUT_DROPOUT_SCHEDULE_START must be non-negative")
if engram_layer_partition_groups < 0:
    raise ValueError("ENGRAM_LAYER_PARTITION_GROUPS must be non-negative")
if engram_layer_partition_groups > 0 and not engram_layer_partitions:
    raise ValueError("ENGRAM_LAYER_PARTITION_GROUPS requires ENGRAM_LAYER_PARTITIONS=1")
if engram_layer_readout_delta and engram_layer_readouts:
    raise ValueError("ENGRAM_LAYER_READOUT_DELTA is a residual alternative to ENGRAM_LAYER_READOUTS")
if engram_layer_readout_delta_scale < 0:
    raise ValueError("ENGRAM_LAYER_READOUT_DELTA_SCALE must be non-negative")
if engram_layer_readout_delta_scale_final < 0:
    raise ValueError("ENGRAM_LAYER_READOUT_DELTA_SCALE_FINAL must be non-negative")
if engram_layer_readout_delta_scale_schedule_steps < 0:
    raise ValueError("ENGRAM_LAYER_READOUT_DELTA_SCALE_SCHEDULE_STEPS must be non-negative")
if engram_layer_readout_delta_scale_schedule_start < 0:
    raise ValueError("ENGRAM_LAYER_READOUT_DELTA_SCALE_SCHEDULE_START must be non-negative")
if engram_layer_readout_delta_learned_scale and not engram_layer_readout_delta:
    raise ValueError("ENGRAM_LAYER_READOUT_DELTA_LEARNED_SCALE requires ENGRAM_LAYER_READOUT_DELTA=1")
if engram_layer_readout_delta_learned_scale_max <= 0:
    raise ValueError("ENGRAM_LAYER_READOUT_DELTA_LEARNED_SCALE_MAX must be positive")
if not (0.0 <= engram_layer_readout_delta_learned_scale_init <= engram_layer_readout_delta_learned_scale_max):
    raise ValueError("ENGRAM_LAYER_READOUT_DELTA_LEARNED_SCALE_INIT must be in [0, max]")
if engram_layer_readout_delta_learned_scale_lr_mul < 0:
    raise ValueError("ENGRAM_LAYER_READOUT_DELTA_LEARNED_SCALE_LR_MUL must be non-negative")
if not (0.0 <= engram_layer_sign_scale <= 1.0):
    raise ValueError("ENGRAM_LAYER_SIGN_SCALE must be in [0, 1]")
if not (0.0 <= engram_layer_sign_scale_final <= 1.0):
    raise ValueError("ENGRAM_LAYER_SIGN_SCALE_FINAL must be in [0, 1]")
if engram_layer_sign_scale_schedule_steps < 0:
    raise ValueError("ENGRAM_LAYER_SIGN_SCALE_SCHEDULE_STEPS must be non-negative")
if engram_layer_sign_scale_schedule_start < 0:
    raise ValueError("ENGRAM_LAYER_SIGN_SCALE_SCHEDULE_START must be non-negative")
if engram_layer_sign_aux_scale < 0:
    raise ValueError("ENGRAM_LAYER_SIGN_AUX_SCALE must be non-negative")
if engram_layer_sign_aux_scale_final < 0:
    raise ValueError("ENGRAM_LAYER_SIGN_AUX_SCALE_FINAL must be non-negative")
if engram_layer_sign_aux_scale_schedule_steps < 0:
    raise ValueError("ENGRAM_LAYER_SIGN_AUX_SCALE_SCHEDULE_STEPS must be non-negative")
if engram_layer_sign_aux_scale_schedule_start < 0:
    raise ValueError("ENGRAM_LAYER_SIGN_AUX_SCALE_SCHEDULE_START must be non-negative")
if engram_layer_signs and (engram_layer_sign_aux_scale > 0 or engram_layer_sign_aux_scale_final > 0):
    raise ValueError("ENGRAM_LAYER_SIGN_AUX_SCALE is an alternative to ENGRAM_LAYER_SIGNS")
if engram_layer_row_signs_aux_only and not engram_layer_row_signs:
    raise ValueError("ENGRAM_LAYER_ROW_SIGNS_AUX_ONLY requires ENGRAM_LAYER_ROW_SIGNS=1")
if not (0.0 <= engram_layer_row_sign_scale <= 1.0):
    raise ValueError("ENGRAM_LAYER_ROW_SIGN_SCALE must be in [0, 1]")
if not (0.0 <= engram_layer_row_sign_scale_final <= 1.0):
    raise ValueError("ENGRAM_LAYER_ROW_SIGN_SCALE_FINAL must be in [0, 1]")
if engram_layer_row_sign_scale_schedule_steps < 0:
    raise ValueError("ENGRAM_LAYER_ROW_SIGN_SCALE_SCHEDULE_STEPS must be non-negative")
if engram_layer_row_sign_scale_schedule_start < 0:
    raise ValueError("ENGRAM_LAYER_ROW_SIGN_SCALE_SCHEDULE_START must be non-negative")
if engram_fixed_half_gate and engram_static_gate:
    raise ValueError("ENGRAM_FIXED_HALF_GATE and ENGRAM_STATIC_GATE are mutually exclusive")
if min(engram_detach_memory_layers, default=0) < 0:
    raise ValueError("ENGRAM_DETACH_MEMORY_LAYERS entries must be non-negative")
if engram_sketch_k <= 0:
    raise ValueError("ENGRAM_SKETCH_K must be positive")
if engram_sketch_dim_signs and engram_sketch_k == 1:
    raise ValueError("ENGRAM_SKETCH_DIM_SIGNS requires ENGRAM_SKETCH_K > 1")
if engram_sketch_dim_sign_mode not in ("random", "hadamard", "balanced"):
    raise ValueError("ENGRAM_SKETCH_DIM_SIGN_MODE must be 'random', 'hadamard', or 'balanced'")
if engram_sketch_scalar_sign_mode not in ("random", "balanced"):
    raise ValueError("ENGRAM_SKETCH_SCALAR_SIGN_MODE must be 'random' or 'balanced'")
if engram_sketch_include_base and engram_sketch_k == 1:
    raise ValueError("ENGRAM_SKETCH_INCLUDE_BASE requires ENGRAM_SKETCH_K > 1")
if engram_sketch_aux_scale < 0:
    raise ValueError("ENGRAM_SKETCH_AUX_SCALE must be non-negative")
if engram_sketch_aux_scale_final < 0:
    raise ValueError("ENGRAM_SKETCH_AUX_SCALE_FINAL must be non-negative")
if engram_sketch_aux_scale_schedule_steps < 0:
    raise ValueError("ENGRAM_SKETCH_AUX_SCALE_SCHEDULE_STEPS must be non-negative")
if engram_sketch_aux_scale_schedule_start < 0:
    raise ValueError("ENGRAM_SKETCH_AUX_SCALE_SCHEDULE_START must be non-negative")
if engram_sketch_aux_learned_scale and not engram_sketch_include_base:
    raise ValueError("ENGRAM_SKETCH_AUX_LEARNED_SCALE requires ENGRAM_SKETCH_INCLUDE_BASE=1")
if engram_sketch_aux_learned_scale_max <= 0:
    raise ValueError("ENGRAM_SKETCH_AUX_LEARNED_SCALE_MAX must be positive")
if not (0.0 <= engram_sketch_aux_learned_scale_init <= engram_sketch_aux_learned_scale_max):
    raise ValueError("ENGRAM_SKETCH_AUX_LEARNED_SCALE_INIT must be in [0, max]")
if engram_sketch_aux_learned_scale_lr_mul < 0:
    raise ValueError("ENGRAM_SKETCH_AUX_LEARNED_SCALE_LR_MUL must be non-negative")
if engram_sketch_slot_readout and engram_sketch_k == 1:
    raise ValueError("ENGRAM_SKETCH_SLOT_READOUT requires ENGRAM_SKETCH_K > 1")
if engram_sketch_slot_readout and not engram_per_head:
    raise ValueError("ENGRAM_SKETCH_SLOT_READOUT requires ENGRAM_PER_HEAD=1")
if engram_sketch_slot_attention and not engram_sketch_slot_readout:
    raise ValueError("ENGRAM_SKETCH_SLOT_ATTENTION requires ENGRAM_SKETCH_SLOT_READOUT=1")
if engram_sketch_slot_attention and (engram_fixed_half_gate or engram_static_gate):
    raise ValueError("ENGRAM_SKETCH_SLOT_ATTENTION requires dynamic key/query gating")
if engram_sketch_slot_mix and not engram_sketch_slot_readout:
    raise ValueError("ENGRAM_SKETCH_SLOT_MIX requires ENGRAM_SKETCH_SLOT_READOUT=1")
if engram_sketch_slot_attention and engram_sketch_slot_mix:
    raise ValueError("ENGRAM_SKETCH_SLOT_ATTENTION and ENGRAM_SKETCH_SLOT_MIX are mutually exclusive")
if engram_sketch_combine_mix and engram_sketch_k == 1:
    raise ValueError("ENGRAM_SKETCH_COMBINE_MIX requires ENGRAM_SKETCH_K > 1")
if engram_sketch_combine_mix_mode not in ("softmax", "bounded"):
    raise ValueError("ENGRAM_SKETCH_COMBINE_MIX_MODE must be 'softmax' or 'bounded'")
if engram_sketch_combine_mix_max_dev < 0:
    raise ValueError("ENGRAM_SKETCH_COMBINE_MIX_MAX_DEV must be non-negative")
if engram_sketch_mix_lr_mul < 0:
    raise ValueError("ENGRAM_SKETCH_MIX_LR_MUL must be non-negative")
if engram_superpose_k <= 0:
    raise ValueError("ENGRAM_SUPERPOSE_K must be positive")
if engram_superpose_include_base and engram_superpose_k <= 1:
    raise ValueError("ENGRAM_SUPERPOSE_INCLUDE_BASE requires ENGRAM_SUPERPOSE_K > 1")
if engram_superpose_aux_scale < 0:
    raise ValueError("ENGRAM_SUPERPOSE_AUX_SCALE must be non-negative")
if engram_superpose_aux_scale_final < 0:
    raise ValueError("ENGRAM_SUPERPOSE_AUX_SCALE_FINAL must be non-negative")
if engram_superpose_aux_scale_schedule_steps < 0:
    raise ValueError("ENGRAM_SUPERPOSE_AUX_SCALE_SCHEDULE_STEPS must be non-negative")
if engram_superpose_aux_scale_schedule_start < 0:
    raise ValueError("ENGRAM_SUPERPOSE_AUX_SCALE_SCHEDULE_START must be non-negative")
if engram_sketch_k > 1 and engram_superpose_k > 1:
    raise ValueError("ENGRAM_SKETCH_K and ENGRAM_SUPERPOSE_K are mutually exclusive")
if engram_read_hit_scale_offset <= 0:
    raise ValueError("ENGRAM_READ_HIT_SCALE_OFFSET must be positive")
if engram_read_hit_scale_min < 0 or engram_read_hit_scale_max <= 0 or engram_read_hit_scale_min > engram_read_hit_scale_max:
    raise ValueError("Invalid ENGRAM_READ_HIT_SCALE_MIN/MAX")
if engram_hit_dropout < 0 or engram_hit_dropout >= 1:
    raise ValueError("ENGRAM_HIT_DROPOUT must be in [0, 1)")
if engram_hit_dropout_final < 0 or engram_hit_dropout_final >= 1:
    raise ValueError("ENGRAM_HIT_DROPOUT_FINAL must be in [0, 1)")
if engram_hit_dropout_decay_final < 0 or engram_hit_dropout_decay_final >= 1:
    raise ValueError("ENGRAM_HIT_DROPOUT_DECAY_FINAL must be in [0, 1)")
if engram_hit_dropout_schedule_steps < 0:
    raise ValueError("ENGRAM_HIT_DROPOUT_SCHEDULE_STEPS must be non-negative")
if engram_hit_dropout_schedule_start < 0:
    raise ValueError("ENGRAM_HIT_DROPOUT_SCHEDULE_START must be non-negative")
if engram_hit_dropout_decay_steps < 0:
    raise ValueError("ENGRAM_HIT_DROPOUT_DECAY_STEPS must be non-negative")
if engram_hit_dropout_decay_start < 0:
    raise ValueError("ENGRAM_HIT_DROPOUT_DECAY_START must be non-negative")
if engram_hit_dropout_min_hits < 0:
    raise ValueError("ENGRAM_HIT_DROPOUT_MIN_HITS must be non-negative")
if engram_hot_split_min_hits < 0:
    raise ValueError("ENGRAM_HOT_SPLIT_MIN_HITS must be non-negative")
if engram_hot_split_aux_scale < 0:
    raise ValueError("ENGRAM_HOT_SPLIT_AUX_SCALE must be non-negative")
if engram_hot_split_aux_scale_final < 0:
    raise ValueError("ENGRAM_HOT_SPLIT_AUX_SCALE_FINAL must be non-negative")
if engram_hot_split_aux_scale_schedule_steps < 0:
    raise ValueError("ENGRAM_HOT_SPLIT_AUX_SCALE_SCHEDULE_STEPS must be non-negative")
if engram_hot_split_aux_scale_schedule_start < 0:
    raise ValueError("ENGRAM_HOT_SPLIT_AUX_SCALE_SCHEDULE_START must be non-negative")
if engram_hot_split_aux_slots <= 0:
    raise ValueError("ENGRAM_HOT_SPLIT_AUX_SLOTS must be positive")
if engram_hot_split_ramp_steps < 0:
    raise ValueError("ENGRAM_HOT_SPLIT_RAMP_STEPS must be non-negative")
if any(kind not in ("lm", "cache") for kind in engram_hit_hist_kinds):
    raise ValueError("ENGRAM_HIT_HIST_KINDS entries must be 'lm' or 'cache'")
if (engram_attnres_extra_source_layer >= 0 or engram_attnres_extra_target_layer >= 0) and not engram_attnres_merge:
    raise ValueError("ENGRAM_ATTNRES_EXTRA_* requires ENGRAM_ATTNRES_MERGE")
if (engram_attnres_extra_source_layer >= 0) != (engram_attnres_extra_target_layer >= 0):
    raise ValueError("Set both ENGRAM_ATTNRES_EXTRA_SOURCE_LAYER and ENGRAM_ATTNRES_EXTRA_TARGET_LAYER, or neither")
if engram_attnres_extra_source_layer >= 0 and engram_attnres_extra_source_layer >= engram_attnres_extra_target_layer:
    raise ValueError("ENGRAM_ATTNRES_EXTRA_SOURCE_LAYER must be before ENGRAM_ATTNRES_EXTRA_TARGET_LAYER")
if engram_init_std < 0:
    raise ValueError("ENGRAM_INIT_STD must be non-negative")
if engram_shadow_grad and (engram_shadow_scale <= 0 or engram_shadow_write_rms < 0 or engram_shadow_write_alpha < 0 or engram_shadow_decay < 0 or engram_shadow_row_rms_cap < 0 or engram_shadow_hit_max < 0):
    raise ValueError("Invalid ENGRAM_SHADOW_* values")
if engram_shadow_only and not engram_shadow_grad:
    raise ValueError("ENGRAM_SHADOW_ONLY requires ENGRAM_SHADOW_GRAD=1")
if engram_cache_recon and engram_cache_recon_target_layer <= engram_cache_recon_source_layer:
    raise ValueError("ENGRAM_CACHE_RECON_TARGET_LAYER must be greater than ENGRAM_CACHE_RECON_SOURCE_LAYER")
if engram_cache_recon and engram_cache_recon_source_layer < 0:
    raise ValueError("ENGRAM_CACHE_RECON_SOURCE_LAYER must be non-negative")
if engram_cache_recon_weight < 0:
    raise ValueError("ENGRAM_CACHE_RECON_WEIGHT must be non-negative")
if engram_analyze and not engram_bigram:
    raise ValueError("ENGRAM_ANALYZE requires ENGRAM_BIGRAM=1")
if engram_analyze and not engram_analyze_ckpt:
    raise ValueError("ENGRAM_ANALYZE requires ENGRAM_ANALYZE_CKPT")
if engram_sparse_adam and not engram_bigram:
    raise ValueError("ENGRAM_SPARSE_ADAM requires ENGRAM_BIGRAM=1")
if engram_sparse_adam and engram_offload:
    raise ValueError("ENGRAM_SPARSE_ADAM is for GPU-resident Engram tables")
if engram_sparse_vector_adam and not engram_bigram:
    raise ValueError("ENGRAM_SPARSE_VECTOR_ADAM requires ENGRAM_BIGRAM=1")
if engram_sparse_vector_adam and engram_offload:
    raise ValueError("ENGRAM_SPARSE_VECTOR_ADAM is for GPU-resident Engram tables")
if engram_sparse_scalar_adam and not engram_bigram:
    raise ValueError("ENGRAM_SPARSE_SCALAR_ADAM requires ENGRAM_BIGRAM=1")
if engram_sparse_scalar_adam and engram_offload:
    raise ValueError("ENGRAM_SPARSE_SCALAR_ADAM is for GPU-resident Engram tables")
if engram_sparse_row_adagrad and not engram_bigram:
    raise ValueError("ENGRAM_SPARSE_ROW_ADAGRAD requires ENGRAM_BIGRAM=1")
if engram_sparse_row_adagrad and engram_offload:
    raise ValueError("ENGRAM_SPARSE_ROW_ADAGRAD is for GPU-resident Engram tables")
if engram_sparse_grad_coalesce_hook_start < 0:
    raise ValueError("ENGRAM_SPARSE_GRAD_COALESCE_HOOK_START must be non-negative")
if engram_manual_sparse_coalesce_start < 0:
    raise ValueError("ENGRAM_MANUAL_SPARSE_COALESCE_START must be non-negative")
if sum((engram_sparse_adam, engram_sparse_vector_adam, engram_sparse_scalar_adam, engram_sparse_row_adagrad)) > 1:
    raise ValueError("Choose only one sparse Engram optimizer")
if engram_freeze_memory and not engram_bigram:
    raise ValueError("ENGRAM_FREEZE_MEMORY requires ENGRAM_BIGRAM=1")
if engram_freeze_memory and engram_offload:
    raise ValueError("ENGRAM_FREEZE_MEMORY currently supports GPU-resident Engram tables")
if engram_freeze_memory and (engram_sparse_adam or engram_sparse_vector_adam or engram_sparse_scalar_adam or engram_sparse_row_adagrad):
    raise ValueError("ENGRAM_FREEZE_MEMORY should not be combined with sparse Engram optimizers")
if engram_sparse_adam_tail_steps < 0:
    raise ValueError("ENGRAM_SPARSE_ADAM_TAIL_STEPS must be non-negative")
if engram_sparse_adam_tail_scale < 0:
    raise ValueError("ENGRAM_SPARSE_ADAM_TAIL_SCALE must be non-negative")
if engram_sparse_extra_steps < 0:
    raise ValueError("ENGRAM_SPARSE_EXTRA_STEPS must be non-negative")
if engram_sparse_weight_decay < 0:
    raise ValueError("ENGRAM_SPARSE_WEIGHT_DECAY must be non-negative")
if engram_sparse_adam_tail_steps > 0 and not engram_sparse_adam:
    raise ValueError("ENGRAM_SPARSE_ADAM_TAIL_STEPS requires ENGRAM_SPARSE_ADAM=1")
if not (0.0 <= engram_sparse_adam_beta1 < 1.0):
    raise ValueError("ENGRAM_SPARSE_ADAM_BETA1 must be in [0, 1)")
if not (0.0 <= engram_sparse_adam_beta2 < 1.0):
    raise ValueError("ENGRAM_SPARSE_ADAM_BETA2 must be in [0, 1)")
if (engram_sparse_hit_lr or engram_sparse_fal or engram_sparse_ifal) and not (engram_sparse_adam or engram_sparse_scalar_adam):
    raise ValueError("ENGRAM_SPARSE_HIT_LR/ENGRAM_SPARSE_FAL/ENGRAM_SPARSE_IFAL require ENGRAM_SPARSE_ADAM=1 or ENGRAM_SPARSE_SCALAR_ADAM=1")
if sum((engram_sparse_hit_lr, engram_sparse_fal, engram_sparse_ifal)) > 1:
    raise ValueError("Choose only one sparse Engram hit-frequency LR mode")
if (engram_sparse_hit_lr or engram_sparse_fal or engram_sparse_ifal) and engram_sparse_vector_adam:
    raise ValueError("Sparse hit-frequency LR modes are currently implemented for standard sparse Adam or scalar sparse Adam")
if (engram_sparse_hit_lr or engram_sparse_fal or engram_sparse_ifal) and engram_sparse_adam_tail_steps > 0:
    raise ValueError("Sparse hit-frequency LR modes are currently incompatible with ENGRAM_SPARSE_ADAM_TAIL_STEPS")
if engram_sparse_hit_lr_min < 0 or engram_sparse_hit_lr_max <= 0 or engram_sparse_hit_lr_min > engram_sparse_hit_lr_max:
    raise ValueError("Invalid ENGRAM_SPARSE_HIT_LR_MIN/MAX")
if engram_sparse_hit_lr_blend < 0 or engram_sparse_hit_lr_blend_final < 0:
    raise ValueError("ENGRAM_SPARSE_HIT_LR_BLEND values must be non-negative")
if engram_sparse_hit_lr_blend_schedule_steps < 0:
    raise ValueError("ENGRAM_SPARSE_HIT_LR_BLEND_SCHEDULE_STEPS must be non-negative")
if engram_sparse_hit_lr_blend_schedule_start < 0:
    raise ValueError("ENGRAM_SPARSE_HIT_LR_BLEND_SCHEDULE_START must be non-negative")
if engram_sparse_param_clamp < 0:
    raise ValueError("ENGRAM_SPARSE_PARAM_CLAMP must be non-negative")
if engram_sparse_row_rms_cap < 0:
    raise ValueError("ENGRAM_SPARSE_ROW_RMS_CAP must be non-negative")
if engram_sparse_row_rms_floor < 0:
    raise ValueError("ENGRAM_SPARSE_ROW_RMS_FLOOR must be non-negative")
if engram_sparse_row_rms_floor_hit_max < 0:
    raise ValueError("ENGRAM_SPARSE_ROW_RMS_FLOOR_HIT_MAX must be non-negative")
if engram_sparse_row_rms_norm < 0:
    raise ValueError("ENGRAM_SPARSE_ROW_RMS_NORM must be non-negative")
if engram_sparse_row_decay < 0:
    raise ValueError("ENGRAM_SPARSE_ROW_DECAY must be non-negative")
if engram_mask_unhit_eval_mode not in ("zero", "random"):
    raise ValueError("ENGRAM_MASK_UNHIT_EVAL_MODE must be 'zero' or 'random'")
if engram_mask_hit_min_eval < 0:
    raise ValueError("ENGRAM_MASK_HIT_MIN_EVAL must be non-negative")
if engram_mask_hit_max_eval < 0:
    raise ValueError("ENGRAM_MASK_HIT_MAX_EVAL must be non-negative")
if engram_mask_hit_min_eval > 0 and engram_mask_hit_max_eval > 0 and engram_mask_hit_min_eval > engram_mask_hit_max_eval:
    raise ValueError("ENGRAM_MASK_HIT_MIN_EVAL must be <= ENGRAM_MASK_HIT_MAX_EVAL when both are set")
if engram_eval_hit_scale < 0:
    raise ValueError("ENGRAM_EVAL_HIT_SCALE must be non-negative")
if engram_eval_hit_scale_min < 0:
    raise ValueError("ENGRAM_EVAL_HIT_SCALE_MIN must be non-negative")
if engram_eval_hit_scale_max < 0:
    raise ValueError("ENGRAM_EVAL_HIT_SCALE_MAX must be non-negative")
if engram_eval_hit_scale_min > 0 and engram_eval_hit_scale_max > 0 and engram_eval_hit_scale_min > engram_eval_hit_scale_max:
    raise ValueError("ENGRAM_EVAL_HIT_SCALE_MIN must be <= ENGRAM_EVAL_HIT_SCALE_MAX when both are set")
if rom_short_conv_kernel <= 0:
    raise ValueError("ROM_SHORT_CONV_KERNEL must be positive")
if rom_ema_kernel <= 0:
    raise ValueError("ROM_EMA_KERNEL must be positive")
if not (0.0 <= rom_ema_alpha < 1.0):
    raise ValueError("ROM_EMA_ALPHA must be in [0, 1)")
if rom_read_mlp and rom_read_mlp_hidden_mult <= 0:
    raise ValueError("ROM_READ_MLP_HIDDEN_MULT must be positive")
if engram_short_conv and engram_short_conv_kernel <= 0:
    raise ValueError("ENGRAM_SHORT_CONV_KERNEL must be positive when ENGRAM_SHORT_CONV=1")
if rom_state_sparse_adam and not rom_state_sparse_embedding:
    raise ValueError("ROM_STATE_SPARSE_ADAM requires ROM_STATE_SPARSE_EMBEDDING=1")
if rom_state_sparse_sgd and not rom_state_sparse_embedding:
    raise ValueError("ROM_STATE_SPARSE_SGD requires ROM_STATE_SPARSE_EMBEDDING=1")
if rom_state_normwrite and not rom_state_sparse_embedding:
    raise ValueError("ROM_STATE_NORMWRITE requires ROM_STATE_SPARSE_EMBEDDING=1")
if rom_state_recovered_normwrite and not rom_state_sparse_embedding:
    raise ValueError("ROM_STATE_RECOVERED_NORMWRITE requires ROM_STATE_SPARSE_EMBEDDING=1")
if rom_state_recovered_normwrite and rom_mqa:
    raise ValueError("ROM_STATE_RECOVERED_NORMWRITE currently requires ROM_MQA=0")
if sum(int(x) for x in (rom_state_sparse_adam, rom_state_sparse_sgd, rom_state_normwrite, rom_state_recovered_normwrite)) > 1:
    raise ValueError("Choose only one sparse ROM optimizer/write rule")
if (rom_state_hit_rms_low > 0 or rom_state_hit_rms_high > 0) and not rom_state_recovered_normwrite:
    raise ValueError("ROM_STATE_HIT_RMS_* requires ROM_STATE_RECOVERED_NORMWRITE=1")
if (rom_state_hit_rms_low > 0) != (rom_state_hit_rms_high > 0):
    raise ValueError("ROM_STATE_HIT_RMS_LOW and ROM_STATE_HIT_RMS_HIGH must both be positive or both be zero")
if rom_state_hit_rms_knee <= 0:
    raise ValueError("ROM_STATE_HIT_RMS_KNEE must be positive")
if rom_sparse_sanitize and not rom_state_sparse_adam:
    raise ValueError("ROM_SPARSE_SANITIZE requires ROM_STATE_SPARSE_ADAM=1")
if rom_sparse_row_scalar_adam and not rom_state_sparse_adam:
    raise ValueError("ROM_SPARSE_ROW_SCALAR_ADAM requires ROM_STATE_SPARSE_ADAM=1")
if rom_state_sparse_embedding and rom_token:
    raise ValueError("ROM_STATE_SPARSE_EMBEDDING currently supports hashed bigram ROM only")
if rom_state_sparse_embedding and not rom_bigram:
    raise ValueError("ROM_STATE_SPARSE_EMBEDDING requires ROM_BIGRAM=1")
if rom_mqa and rom_engram_gate:
    raise ValueError("ROM_MQA currently uses ROM's learned per-query gate; disable ROM_ENGRAM_GATE")
if rom_mqa and rom_token:
    raise ValueError("ROM_MQA currently supports hashed bigram ROM only")

# -----------------------------------------------------------------------------
# Custom operators: FP8 matmul by @YouJiacheng
# Transposed layout by @ChrisJMcCormick allows for faster gradient accumulation.

@torch.library.custom_op("nanogpt::mm_t", mutates_args=())
def mm_t_op(x: Tensor, w: Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[Tensor, Tensor, Tensor]:
    """Computes y = x @ w with F8 weights stored as (in_features, out_features)."""
    @torch.compile
    def impl(x: Tensor, w: Tensor):
        assert x.is_contiguous() and w.is_contiguous()
        assert x.shape[1] == w.shape[0]  # x: (batch, in), w: (in, out)

        x_f8 = x.div(x_s).to(torch.float8_e4m3fn)
        w_f8 = w.div(w_s).to(torch.float8_e4m3fn)

        # _scaled_mm requires column-major B. w_f8 is row-major (in, out).
        # .T.contiguous().T creates a column-major view without changing logical shape.
        w_f8_col_major = w_f8.T.contiguous().T

        out = torch._scaled_mm(
            x_f8,
            w_f8_col_major,
            out_dtype=torch.bfloat16,
            scale_a=x.new_tensor(x_s, dtype=torch.float32),
            scale_b=x.new_tensor(w_s, dtype=torch.float32),
            use_fast_accum=True,
        )
        return out, x_f8, w_f8

    return impl(x, w)

@mm_t_op.register_fake
def _(x: Tensor, w: Tensor, *_):
    assert x.ndim == w.ndim == 2
    assert x.shape[1] == w.shape[0]
    assert x.device == w.device
    assert x.is_contiguous() and w.is_contiguous()
    return x @ w, x.to(torch.float8_e4m3fn), w.to(torch.float8_e4m3fn)

@torch.library.custom_op("nanogpt::mm_t_backward", mutates_args=())
def mm_t_backward_op(g: Tensor, x_f8: Tensor, w_f8: Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[Tensor, Tensor]:
    @torch.compile
    def impl(grad: Tensor, x_f8: Tensor, w_f8: Tensor):
        assert grad.is_contiguous()

        x_scale = grad.new_tensor(x_s, dtype=torch.float32)
        w_scale = grad.new_tensor(w_s, dtype=torch.float32)
        grad_scale = grad.new_tensor(grad_s, dtype=torch.float32)
        grad_f8 = grad.div(grad_s).to(torch.float8_e5m2)

        # grad_x = grad @ w.T
        grad_x = torch._scaled_mm(
            grad_f8,
            w_f8.T,
            out_dtype=torch.bfloat16,
            scale_a=grad_scale,
            scale_b=w_scale,
            use_fast_accum=False,
        )

        # grad_w = x.T @ grad
        # Result is (in, out), naturally matching weight storage. No final .T needed.
        grad_w = torch._scaled_mm(
            x_f8.T.contiguous(),
            grad_f8.T.contiguous().T,
            out_dtype=torch.float32,
            scale_a=x_scale,
            scale_b=grad_scale,
            use_fast_accum=False,
        )

        return grad_x, grad_w

    grad_x, grad_w = impl(g, x_f8, w_f8)

    return grad_x, grad_w

@mm_t_backward_op.register_fake
def _(g: Tensor, x_f8: Tensor, w_f8: Tensor, *_):
    return x_f8.to(torch.bfloat16), w_f8.to(torch.float32)

def backward_t(ctx, grad_out: Tensor, *_):
    x_f8, w_f8 = ctx.saved_tensors
    x_s, w_s, grad_s = ctx.scales
    grad_x, grad_w = torch.ops.nanogpt.mm_t_backward(
        grad_out, x_f8, w_f8, x_s, w_s, grad_s
    )
    return grad_x, grad_w, None, None, None

def setup_context_t(ctx: torch.autograd.function.FunctionCtx, inputs, output):
    *_, x_s, w_s, grad_s = inputs
    _, x_f8, w_f8 = output
    ctx.save_for_backward(x_f8, w_f8)
    ctx.scales = x_s, w_s, grad_s
    ctx.set_materialize_grads(False)

mm_t_op.register_autograd(backward_t, setup_context=setup_context_t)

# -----------------------------------------------------------------------------
# Polar Express

# Computed for num_iters=5, safety_factor=2e-2, cushion=2
polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323)
]

@torch.compile(dynamic=False, fullgraph=True) # Must use dynamic=False or else it's much slower
def polar_express(grad_chunk: torch.Tensor, momentum_buffer: torch.Tensor, momentum_t: torch.Tensor,
                  split_baddbmm: bool = False):
    """
    Fused Nesterov momentum + Polar Express Sign Method.
    Nesterov momentum is applied in FP32, then the result is cast to BF16 for polar express
    orthogonalization, avoiding materialization of the FP32 intermediate between graph breaks.

    Polar Express: https://arxiv.org/pdf/2505.16932
    by Noah Amsel, David Persson, Christopher Musco, Robert M. Gower.

    momentum_t is a 0-D CPU tensor to avoid triggering graph recompilations when the value changes.
    """
    # Nesterov momentum (in FP32)
    momentum = momentum_t.to(grad_chunk.dtype)
    momentum_buffer.lerp_(grad_chunk, 1 - momentum)
    g = grad_chunk.lerp_(momentum_buffer, momentum)

    X = g.bfloat16()
    is_tall = g.size(-2) > g.size(-1)

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * (1 + 2e-2) + 1e-6)

    X = X.contiguous()

    if is_tall:
        # Tall: use Triton kernels with X^T @ X (small) and right multiplication
        A = torch.empty((*X.shape[:-2], X.size(-1), X.size(-1)), device=X.device, dtype=X.dtype)
        B = torch.empty_like(A)
        C = torch.empty_like(X)

        # Select batched vs unbatched
        if split_baddbmm:
            XB_matmul = torch.bmm if X.ndim > 2 else torch.mm
        else:
            aX_plus_XB = torch.baddbmm if X.ndim > 2 else torch.addmm

        # Perform the iterations
        for a, b, c in polar_express_coeffs:
            XTX(X, out=A)  # A = X.T @ X
            ba_plus_cAA(A, alpha=c, beta=b, out=B)  # B = b*A + c*(A@A)

            # Referencing X twice causes pytorch to make a defensive copy,
            # resulting in a cudaMemcpyAsync in baddbmm.
            # For large matrices (i.e., the mlp weights), it's faster to split
            # the operation into two kernels to avoid this.
            if split_baddbmm:
                XB_matmul(X, B, out=C)  # C = X @ B
                C.add_(X, alpha=a)      # C = C + a*X  (in-place, X only read)
            else:
                aX_plus_XB(X, X, B, beta=a, out=C)  # C = a * X + X @ B

            X, C = C, X  # Swap references to avoid unnecessary copies
    else:
        # Wide: use Triton kernels with X @ X^T (small) and left multiplication
        A = torch.empty((*X.shape[:-1], X.size(-2)), device=X.device, dtype=X.dtype)
        B = torch.empty_like(A)
        C = torch.empty_like(X)

        # Select batched vs unbatched
        if split_baddbmm:
            BX_matmul = torch.bmm if X.ndim > 2 else torch.mm
        else:
            aX_plus_BX = torch.baddbmm if X.ndim > 2 else torch.addmm

        # Perform the iterations
        for a, b, c in polar_express_coeffs:
            XXT(X, out=A)  # A = X @ X.mT
            ba_plus_cAA(A, alpha=c, beta=b, out=B)  # B = b * A + c * A @ A

            if split_baddbmm:
                BX_matmul(B, X, out=C)  # C = B @ X
                C.add_(X, alpha=a)      # C = C + a*X  (in-place, X only read)
            else:
                aX_plus_BX(X, B, X, beta=a, out=C)  # C = a * X + B @ X

            X, C = C, X  # Swap references to avoid unnecessary copies

    return X

# -----------------------------------------------------------------------------
# Sparse Comms for bigram embedding gradient reduce-scatter
def _sparse_comms_active():
    # we count on this in order for sparse communication to be worthwhile
    return world_size == 8 and grad_accum_steps == 1

@torch.no_grad
def sparse_comms_start(idxes_np, N, rank, world, send_idxes_buffer):
    rows_per_rank = N // world

    # queue upload of indexes to gpu
    send_idxes = send_idxes_buffer[:idxes_np.shape[0]]
    send_idxes.copy_(torch.from_numpy(idxes_np))
    send_idxes = send_idxes.to(device, non_blocking=True)

    # calculate how many gradient rows we will send to every rank
    insertion_points = np.searchsorted(
        idxes_np,
        np.arange(0, rows_per_rank * (world + 1), rows_per_rank, dtype=np.int32),
    )
    send_counts = torch.from_numpy(insertion_points[1:] - insertion_points[:-1])
    # zero-out own send-count - we won't send our own gradient rows to ourselves as it's a waste:
    # in sparse_comms_merge_gradients, we'll use the slice of the gradient that already includes them as the base tensor
    send_counts[rank] = 0

    # remove indexes owned by our rank from the send list
    send_idxes = torch.cat([send_idxes[: insertion_points[rank]], send_idxes[insertion_points[rank + 1] :]])

    # share the send counts so that each rank will know how many rows
    # to expect from every other rank
    recv_counts = torch.empty_like(send_counts)
    recv_counts_fut = dist.all_to_all_single(recv_counts, send_counts, async_op=True).get_future()
    return send_idxes, send_counts, recv_counts, recv_counts_fut

@torch.no_grad
def sparse_comms_share_indexes(send_idxes, send_counts, recv_counts):
    # cpu tensors, so these ops are cheap and don't force a host<->device sync
    total_recv_count = recv_counts.sum().item()
    recv_counts = recv_counts.tolist()
    send_counts = send_counts.tolist()

    # queue sharing of row indexes
    recv_idxes = torch.empty(total_recv_count, dtype=torch.int32, device=device)
    idxes_fut = dist.all_to_all_single(
        recv_idxes,
        send_idxes,
        output_split_sizes=recv_counts,
        input_split_sizes=send_counts,
        async_op=True,
    ).get_future()

    sparse_state = {
        "send_idxes": send_idxes,
        "send_counts": send_counts,
        "recv_counts": recv_counts, # list for sharing
    }
    return recv_idxes, sparse_state, idxes_fut

@torch.compile
@torch.no_grad
def sparse_comms_share_gradients(grad, idxes, send_counts, recv_counts):
    # gather the rows that we want to send
    send_vals = grad[idxes]

    d = grad.shape[1]

    send_sizes = [i*d for i in send_counts]
    recv_sizes = [i*d for i in recv_counts]

    recv_vals = torch.empty(sum(recv_sizes), device=send_vals.device, dtype=grad.dtype)

    val_fut = dist.all_to_all_single(
        recv_vals,
        send_vals.view(-1),
        input_split_sizes=send_sizes,
        output_split_sizes=recv_sizes,
        async_op=True,
    ).get_future()

    return recv_vals, val_fut

@torch.no_grad
def sparse_comms_merge_gradients(grad, recv_idx, recv_vals, rank, world):
    d = grad.shape[1]
    rows_per_rank = grad.shape[0] // world

    grad.index_add_(0, recv_idx, recv_vals.view(-1, d))

    # return the slice of the gradient for parameters our rank updates
    return grad[rows_per_rank * rank : rows_per_rank * (rank + 1)].mul_((1 / world))


# -----------------------------------------------------------------------------
# Combined NorMuon + Adam Optimizer

class Snoo:
    """Sparse Nesterov outer optimizer wrapper from the medium-track speedrun records."""
    @torch.no_grad()
    def __init__(self, params: list[nn.Parameter], lr: float, momentum: float, k: int) -> None:
        if k <= 0:
            raise ValueError(f"Invalid Snoo k: {k}")
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.k = k
        self.current_step = 0
        self.outer_buf = [p.detach().clone() for p in self.params]
        self.optimizer = torch.optim.SGD(
            self.params,
            lr=lr,
            momentum=momentum,
            nesterov=True,
            fused=True,
        )

    @torch.no_grad()
    def step(self) -> None:
        if self.current_step % self.k == 0:
            for p_new, p_old in zip(self.params, self.outer_buf):
                p_new.grad = p_old.data - p_new.data
                p_new.copy_(p_old, non_blocking=True)

            self.optimizer.step()

            for p_new, p_old in zip(self.params, self.outer_buf):
                p_old.copy_(p_new, non_blocking=True)
                p_new.grad = None
        self.current_step += 1

    def state_dict(self):
        return {
            "current_step": self.current_step,
            "lr": self.lr,
            "momentum": self.momentum,
            "k": self.k,
            "outer_buf": self.outer_buf,
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict) -> None:
        self.current_step = state_dict["current_step"]
        for target, source in zip(self.outer_buf, state_dict["outer_buf"]):
            target.copy_(source, non_blocking=True)
        self.optimizer.load_state_dict(state_dict["optimizer"])


class SparseSGDMomentum(torch.optim.Optimizer):
    """SGD with dense momentum state, but sparse row updates for embedding grads."""
    def __init__(self, params, lr: float, momentum: float = 0.9, weight_decay: float = 0.0):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0:
            raise ValueError(f"Invalid momentum: {momentum}")
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue
                if not grad.is_sparse:
                    raise RuntimeError("SparseSGDMomentum requires sparse gradients")

                grad = coalesce_row_sparse_grad(grad)
                idx = grad.indices()[0]
                grad_values = grad.values()
                if weight_decay != 0:
                    grad_values = grad_values + weight_decay * param.index_select(0, idx).to(grad_values.dtype)

                if momentum != 0:
                    state = self.state[param]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(param, dtype=torch.float32)
                    buf = state["momentum_buffer"]
                    update = buf.index_select(0, idx)
                    update.mul_(momentum).add_(grad_values.float())
                    buf.index_copy_(0, idx, update)
                else:
                    update = grad_values.float()

                param.index_add_(0, idx, update.to(param.dtype), alpha=-lr)

        return loss


class SparseNormalizedWrite(torch.optim.Optimizer):
    """Sparse row update that writes a fixed-RMS normalized gradient into each touched row."""
    def __init__(self, params, write_rms: float = 0.001, row_cap: float = 0.0):
        if write_rms < 0:
            raise ValueError(f"Invalid write_rms: {write_rms}")
        if row_cap < 0:
            raise ValueError(f"Invalid row_cap: {row_cap}")
        defaults = dict(write_rms=write_rms, row_cap=row_cap)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            write_rms = group["write_rms"]
            row_cap = group["row_cap"]
            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue
                if not grad.is_sparse:
                    raise RuntimeError("SparseNormalizedWrite requires sparse gradients")

                grad = coalesce_row_sparse_grad(grad)
                idx = grad.indices()[0]
                grad_values = torch.nan_to_num(grad.values().float(), nan=0.0, posinf=0.0, neginf=0.0)
                if idx.numel() == 0 or write_rms == 0:
                    continue
                flat = grad_values.flatten(start_dim=1)
                rms = row_rms_stable(flat).unsqueeze(1).clamp_min_(1e-12)
                update = -grad_values * (write_rms / rms).view(-1, *([1] * (grad_values.ndim - 1)))
                param.index_add_(0, idx, update.to(dtype=param.dtype))
                if row_cap > 0:
                    idx = torch.unique(idx)
                    rows = param.index_select(0, idx).float()
                    rows = torch.nan_to_num(rows, nan=0.0, posinf=0.0, neginf=0.0)
                    row_rms = row_rms_stable(rows.flatten(start_dim=1)).unsqueeze(1)
                    scale = (row_cap / row_rms.clamp_min(1e-12)).clamp_max(1.0)
                    rows.mul_(scale.view(-1, *([1] * (rows.ndim - 1))))
                    param.index_copy_(0, idx, rows.to(dtype=param.dtype))

        return loss


class SparseRowScalarAdam(torch.optim.Optimizer):
    """Sparse Adam with vector first moments and one second-moment scalar per row."""
    def __init__(self, params, lr: float, betas: tuple[float, float] = (0.75, 0.95),
                 eps: float = 1e-10, weight_decay: float = 0.0):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        beta1, beta2 = betas
        if not 0 <= beta1 < 1:
            raise ValueError(f"Invalid beta1: {beta1}")
        if not 0 <= beta2 < 1:
            raise ValueError(f"Invalid beta2: {beta2}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.last_update_metrics: dict[str, float | int] = {}

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.last_update_metrics = {}
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue
                if not grad.is_sparse:
                    raise RuntimeError("SparseRowScalarAdam requires sparse gradients")

                grad = coalesce_row_sparse_grad(grad)
                idx = grad.indices()[0]
                grad_values = grad.values().float()
                if idx.numel() == 0:
                    continue
                debug_raise_nonfinite("sparse_row_scalar_adam.grad_values", grad_values, idx)
                if engram_sparse_sanitize:
                    sanitize_sparse_grad_values_(grad_values)

                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param, dtype=torch.float32, device=param.device)
                    state["exp_avg_sq_row"] = torch.zeros(param.shape[0], dtype=torch.float32, device=param.device)

                state["step"] += 1
                step = int(state["step"])

                exp_avg_rows = state["exp_avg"].index_select(0, idx)
                exp_avg_sq_row = state["exp_avg_sq_row"].index_select(0, idx)
                param_rows = param.index_select(0, idx).float()
                debug_raise_nonfinite("sparse_row_scalar_adam.exp_avg_before", exp_avg_rows, idx)
                debug_raise_nonfinite("sparse_row_scalar_adam.exp_avg_sq_before", exp_avg_sq_row)
                debug_raise_nonfinite("sparse_row_scalar_adam.param_rows_before", param_rows, idx)
                if engram_sparse_sanitize:
                    exp_avg_rows.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                    exp_avg_sq_row.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                    exp_avg_sq_row.clamp_min_(0.0)
                    param_rows.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

                exp_avg_rows.mul_(beta1).add_(grad_values, alpha=1 - beta1)
                grad_row_rms = row_rms_stable(grad_values)
                grad_row_second = grad_row_rms.square()
                exp_avg_sq_row.mul_(beta2).add_(grad_row_second, alpha=1 - beta2)
                debug_raise_nonfinite("sparse_row_scalar_adam.exp_avg_after_moment", exp_avg_rows, idx)
                debug_raise_nonfinite("sparse_row_scalar_adam.exp_avg_sq_after_moment", exp_avg_sq_row)
                if engram_sparse_sanitize:
                    exp_avg_rows.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                    exp_avg_sq_row.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                    exp_avg_sq_row.clamp_min_(0.0)

                bias1 = 1 - beta1 ** step
                bias2 = 1 - beta2 ** step
                step_size = lr * (bias2 ** 0.5 / bias1)
                update = exp_avg_rows / exp_avg_sq_row.sqrt().add_(eps).unsqueeze(1)
                update.mul_(step_size)
                debug_raise_nonfinite("sparse_row_scalar_adam.update", update, idx)
                if engram_sparse_sanitize:
                    update.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                if weight_decay:
                    mask = (update * param_rows) > 0
                    update.addcmul_(param_rows, mask, value=lr * lr * weight_decay)

                if engram_update_metrics and engram_update_metrics_every > 0 and step % engram_update_metrics_every == 0:
                    update_rms = rms_no_alloc(update)
                    param_rms = rms_no_alloc(param_rows)
                    grad_rms = rms_no_alloc(grad_values)
                    table_numel = max(1, param.numel())
                    table_grad_rms = table_rms_no_alloc(grad_values, table_numel)
                    table_update_rms = table_rms_no_alloc(update, table_numel)
                    self.last_update_metrics = {
                        "adam_step": step,
                        "rows": int(idx.numel()),
                        "touched_rows": int(idx.numel()),
                        "table_rows": int(param.shape[0]),
                        "table_numel": int(param.numel()),
                        "grad_rms": float(grad_rms.item()),
                        "update_rms": float(update_rms.item()),
                        "param_rms": float(param_rms.item()),
                        "touched_grad_rms": float(grad_rms.item()),
                        "touched_update_rms": float(update_rms.item()),
                        "touched_param_rms": float(param_rms.item()),
                        "table_grad_rms": float(table_grad_rms.item()),
                        "table_update_rms": float(table_update_rms.item()),
                        "lr": float(lr),
                        "step_size": float(step_size),
                    }

                param_rows.add_(update, alpha=-1.0)
                debug_raise_nonfinite("sparse_row_scalar_adam.param_rows_after", param_rows, idx)
                if engram_sparse_sanitize:
                    param_rows.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                index_copy_cast_chunked_(param, 0, idx, param_rows)
                state["exp_avg"].index_copy_(0, idx, exp_avg_rows)
                state["exp_avg_sq_row"].index_copy_(0, idx, exp_avg_sq_row)

        return loss


class SparseScalarAdam(torch.optim.Optimizer):
    """Sparse Adam with one first-moment magnitude and one second-moment scalar per row.

    A scalar first moment has no direction by itself, so the current gradient supplies
    the update direction while the row scalar momentum smooths the update magnitude.
    """
    def __init__(self, params, lr: float, betas: tuple[float, float] = (0.75, 0.95),
                 eps: float = 1e-10, weight_decay: float = 0.0):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        beta1, beta2 = betas
        if not 0 <= beta1 < 1:
            raise ValueError(f"Invalid beta1: {beta1}")
        if not 0 <= beta2 < 1:
            raise ValueError(f"Invalid beta2: {beta2}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.last_update_metrics: dict[str, float | int] = {}

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.last_update_metrics = {}
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue
                if not grad.is_sparse:
                    raise RuntimeError("SparseScalarAdam requires sparse gradients")

                hit_lr_mode = engram_sparse_hit_lr or engram_sparse_fal or engram_sparse_ifal
                raw_idx = grad._indices()[0] if hit_lr_mode else None
                grad = coalesce_row_sparse_grad(grad)
                idx = grad.indices()[0]
                grad_values = grad.values().float()
                if idx.numel() == 0:
                    continue
                debug_raise_nonfinite("sparse_scalar_adam.grad_values", grad_values, idx)
                if engram_sparse_sanitize:
                    sanitize_sparse_grad_values_(grad_values)

                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg_row"] = torch.zeros(param.shape[0], dtype=torch.float32, device=param.device)
                    state["exp_avg_sq_row"] = torch.zeros(param.shape[0], dtype=torch.float32, device=param.device)
                    if hit_lr_mode:
                        state["hit_count"] = torch.zeros(param.shape[0], dtype=torch.float32, device=param.device)

                state["step"] += 1
                step = int(state["step"])

                lr_scale = None
                hit_count_rows = None
                hit_lr_blend = 1.0
                if hit_lr_mode:
                    if "hit_count" not in state:
                        state["hit_count"] = torch.zeros(param.shape[0], dtype=torch.float32, device=param.device)
                    assert raw_idx is not None
                    if raw_idx.numel() == idx.numel():
                        hit_delta = torch.ones(idx.numel(), dtype=torch.float32, device=param.device)
                    else:
                        raw_unique, raw_counts = torch.unique(raw_idx, sorted=True, return_counts=True)
                        hit_delta = raw_counts.to(dtype=torch.float32).index_select(0, torch.searchsorted(raw_unique, idx))
                    hit_count_rows = state["hit_count"].index_select(0, idx)
                    hit_count_rows.add_(hit_delta)
                    if engram_sparse_fal:
                        max_hit = torch.maximum(state["hit_count"].max(), hit_count_rows.max()).clamp_min(1.0)
                        denom = torch.log1p(max_hit)
                        lr_scale = torch.log1p(hit_count_rows) / denom
                        lr_scale.clamp_(min=engram_sparse_hit_lr_min, max=engram_sparse_hit_lr_max)
                    elif engram_sparse_ifal:
                        lr_scale = torch.log1p(hit_count_rows).reciprocal()
                        lr_scale.div_(lr_scale.mean().clamp_min(1e-12))
                        lr_scale.clamp_(min=engram_sparse_hit_lr_min, max=engram_sparse_hit_lr_max)
                    elif engram_sparse_hit_lr_exponent == 0:
                        lr_scale = torch.ones_like(hit_count_rows)
                    else:
                        lr_scale = hit_count_rows.pow(engram_sparse_hit_lr_exponent)
                        lr_scale.clamp_(min=engram_sparse_hit_lr_min, max=engram_sparse_hit_lr_max)
                    hit_lr_blend = scheduled_scalar(
                        engram_sparse_hit_lr_blend,
                        engram_sparse_hit_lr_blend_final,
                        engram_sparse_hit_lr_blend_schedule_steps,
                        engram_sparse_hit_lr_blend_schedule_start,
                        step,
                    )
                    if hit_lr_blend != 1.0:
                        lr_scale.sub_(1.0).mul_(hit_lr_blend).add_(1.0)

                exp_avg_row = state["exp_avg_row"].index_select(0, idx)
                exp_avg_sq_row = state["exp_avg_sq_row"].index_select(0, idx)
                debug_raise_nonfinite("sparse_scalar_adam.exp_avg_row_before", exp_avg_row)
                debug_raise_nonfinite("sparse_scalar_adam.exp_avg_sq_before", exp_avg_sq_row)
                if engram_sparse_sanitize:
                    exp_avg_row.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                    exp_avg_sq_row.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                    exp_avg_sq_row.clamp_min_(0.0)

                grad_row_rms = row_rms_stable(grad_values)
                exp_avg_row.mul_(beta1).add_(grad_row_rms, alpha=1 - beta1)
                exp_avg_sq_row.mul_(beta2).add_(grad_row_rms.square(), alpha=1 - beta2)
                debug_raise_nonfinite("sparse_scalar_adam.exp_avg_row_after_moment", exp_avg_row)
                debug_raise_nonfinite("sparse_scalar_adam.exp_avg_sq_after_moment", exp_avg_sq_row)
                if engram_sparse_sanitize:
                    exp_avg_row.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                    exp_avg_sq_row.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                    exp_avg_sq_row.clamp_min_(0.0)

                bias1 = 1 - beta1 ** step
                bias2 = 1 - beta2 ** step
                step_size = lr * (bias2 ** 0.5 / bias1)
                grad_rms_metric = None
                table_grad_rms_metric = None
                if engram_update_metrics and engram_update_metrics_every > 0 and step % engram_update_metrics_every == 0:
                    grad_rms_metric = rms_no_alloc(grad_values)
                    table_grad_rms_metric = table_rms_no_alloc(grad_values, max(1, param.numel()))

                # Reuse the sparse grad value buffer as the update buffer. Wide/banked
                # tables can fit the scalar moments but not an extra rows-by-dim
                # temporary for the update.
                update = grad_values
                update.div_(grad_row_rms.clamp_min(eps).unsqueeze(1))
                update.mul_((step_size * exp_avg_row / exp_avg_sq_row.sqrt().add_(eps)).unsqueeze(1))
                if lr_scale is not None:
                    update.mul_(lr_scale.unsqueeze(1))
                debug_raise_nonfinite("sparse_scalar_adam.update", update, idx)
                if engram_sparse_sanitize:
                    update.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

                if engram_update_metrics and engram_update_metrics_every > 0 and step % engram_update_metrics_every == 0:
                    update_rms = rms_no_alloc(update)
                    param_rms = indexed_rows_rms_no_alloc(param, idx)
                    table_numel = max(1, param.numel())
                    table_update_rms = table_rms_no_alloc(update, table_numel)
                    assert grad_rms_metric is not None
                    assert table_grad_rms_metric is not None
                    self.last_update_metrics = {
                        "adam_step": step,
                        "rows": int(idx.numel()),
                        "touched_rows": int(idx.numel()),
                        "table_rows": int(param.shape[0]),
                        "table_numel": int(param.numel()),
                        "grad_rms": float(grad_rms_metric.item()),
                        "update_rms": float(update_rms.item()),
                        "param_rms": float(param_rms.item()),
                        "touched_grad_rms": float(grad_rms_metric.item()),
                        "touched_update_rms": float(update_rms.item()),
                        "touched_param_rms": float(param_rms.item()),
                        "table_grad_rms": float(table_grad_rms_metric.item()),
                        "table_update_rms": float(table_update_rms.item()),
                        "lr": float(lr),
                        "step_size": float(step_size),
                    }
                    if lr_scale is not None and hit_count_rows is not None:
                        self.last_update_metrics.update({
                            "hit_lr_scale_mean": float(lr_scale.mean().item()),
                            "hit_lr_scale_min": float(lr_scale.min().item()),
                            "hit_lr_scale_max": float(lr_scale.max().item()),
                            "hit_lr_blend": float(hit_lr_blend),
                            "hit_count_mean": float(hit_count_rows.mean().item()),
                            "hit_count_max": float(hit_count_rows.max().item()),
                        })

                row_width = max(1, int(param[0].numel()))
                row_bytes = row_width * torch.empty((), dtype=torch.float32, device="cpu").element_size()
                chunk_rows = max(1, (256 * 1024 * 1024) // row_bytes)
                for start in range(0, idx.numel(), chunk_rows):
                    end = min(idx.numel(), start + chunk_rows)
                    idx_chunk = idx[start:end]
                    update_chunk = update[start:end]
                    param_rows = param.index_select(0, idx_chunk).float()
                    debug_raise_nonfinite("sparse_scalar_adam.param_rows_before", param_rows, idx_chunk)
                    if engram_sparse_sanitize:
                        param_rows.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                    if weight_decay:
                        mask = (update_chunk * param_rows) > 0
                        update_chunk.addcmul_(param_rows, mask, value=lr * lr * weight_decay)
                    param_rows.add_(update_chunk, alpha=-1.0)
                    debug_raise_nonfinite("sparse_scalar_adam.param_rows_after", param_rows, idx_chunk)
                    if engram_sparse_sanitize:
                        param_rows.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                    index_copy_cast_chunked_(param, 0, idx_chunk, param_rows)
                state["exp_avg_row"].index_copy_(0, idx, exp_avg_row)
                state["exp_avg_sq_row"].index_copy_(0, idx, exp_avg_sq_row)
                if hit_count_rows is not None:
                    state["hit_count"].index_copy_(0, idx, hit_count_rows)

        return loss


class SparseAdamWithTail(torch.optim.Optimizer):
    """Sparse Adam that keeps recently touched rows active for zero-grad momentum tails."""
    def __init__(self, params, lr: float, betas: tuple[float, float] = (0.75, 0.95),
                 eps: float = 1e-10, weight_decay: float = 0.0, tail_steps: int = 0,
                 tail_scale: float = 1.0):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        beta1, beta2 = betas
        if not 0 <= beta1 < 1:
            raise ValueError(f"Invalid beta1: {beta1}")
        if not 0 <= beta2 < 1:
            raise ValueError(f"Invalid beta2: {beta2}")
        if tail_steps < 0:
            raise ValueError(f"Invalid tail_steps: {tail_steps}")
        if tail_scale < 0:
            raise ValueError(f"Invalid tail_scale: {tail_scale}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, tail_steps=tail_steps, tail_scale=tail_scale)
        super().__init__(params, defaults)
        self.last_update_metrics: dict[str, float | int] = {}

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.last_update_metrics = {}
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            tail_steps = int(group["tail_steps"])
            tail_scale = float(group["tail_scale"])
            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue
                if not grad.is_sparse:
                    raise RuntimeError("SparseAdamWithTail requires sparse gradients")

                raw_idx = grad._indices()[0]
                grad = coalesce_row_sparse_grad(grad)
                idx = grad.indices()[0]
                grad_values = grad.values().float()
                if idx.numel() == 0:
                    continue
                debug_raise_nonfinite("sparse_adam_tail.grad_values", grad_values, idx)
                if engram_sparse_sanitize:
                    sanitize_sparse_grad_values_(grad_values)

                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param, dtype=torch.float32, device=param.device)
                    state["exp_avg_sq"] = torch.zeros_like(param, dtype=torch.float32, device=param.device)
                    state["active_rows"] = torch.empty(0, dtype=torch.long, device=param.device)
                    state["active_age"] = torch.empty(0, dtype=torch.int16, device=param.device)
                    state["hit_count"] = torch.zeros(param.shape[0], dtype=torch.float32, device=param.device)

                state["step"] += 1
                step = int(state["step"])

                if tail_steps > 0 and state["active_rows"].numel() > 0:
                    keep = state["active_age"] < tail_steps
                    old_rows = state["active_rows"][keep]
                    old_age = state["active_age"][keep]
                    if old_rows.numel() > 0:
                        all_rows = torch.cat((old_rows, idx))
                        active_idx = torch.unique(all_rows, sorted=True)
                        old_pos = torch.searchsorted(active_idx, old_rows)
                        cur_pos = torch.searchsorted(active_idx, idx)
                        active_age = torch.full(
                            (active_idx.numel(),), tail_steps + 1,
                            dtype=torch.int16, device=param.device,
                        )
                        active_age[old_pos] = old_age + 1
                        active_age[cur_pos] = 0
                    else:
                        active_idx = idx
                        cur_pos = torch.arange(idx.numel(), device=param.device)
                        active_age = torch.zeros(idx.numel(), dtype=torch.int16, device=param.device)
                else:
                    active_idx = idx
                    cur_pos = torch.arange(idx.numel(), device=param.device)
                    active_age = torch.zeros(idx.numel(), dtype=torch.int16, device=param.device)

                grad_active = torch.zeros((active_idx.numel(), param.shape[1]), dtype=torch.float32, device=param.device)
                grad_active.index_copy_(0, cur_pos, grad_values)

                if raw_idx.numel() == idx.numel():
                    hit_delta = torch.ones(idx.numel(), dtype=torch.float32, device=param.device)
                else:
                    raw_unique, raw_counts = torch.unique(raw_idx, sorted=True, return_counts=True)
                    hit_delta = raw_counts.to(dtype=torch.float32).index_select(0, torch.searchsorted(raw_unique, idx))
                hit_count_rows = state["hit_count"].index_select(0, idx)
                hit_count_rows.add_(hit_delta)

                exp_avg_rows = state["exp_avg"].index_select(0, active_idx)
                exp_avg_sq_rows = state["exp_avg_sq"].index_select(0, active_idx)
                param_rows = param.index_select(0, active_idx).float()
                debug_raise_nonfinite("sparse_adam_tail.exp_avg_before", exp_avg_rows, active_idx)
                debug_raise_nonfinite("sparse_adam_tail.exp_avg_sq_before", exp_avg_sq_rows, active_idx)
                debug_raise_nonfinite("sparse_adam_tail.param_rows_before", param_rows, active_idx)
                if engram_sparse_sanitize:
                    exp_avg_rows.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                    exp_avg_sq_rows.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                    exp_avg_sq_rows.clamp_min_(0.0)
                    param_rows.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

                exp_avg_rows.mul_(beta1).add_(grad_active, alpha=1 - beta1)
                exp_avg_sq_rows.mul_(beta2).addcmul_(grad_active, grad_active, value=1 - beta2)
                debug_raise_nonfinite("sparse_adam_tail.exp_avg_after_moment", exp_avg_rows, active_idx)
                debug_raise_nonfinite("sparse_adam_tail.exp_avg_sq_after_moment", exp_avg_sq_rows, active_idx)
                if engram_sparse_sanitize:
                    exp_avg_rows.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                    exp_avg_sq_rows.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                    exp_avg_sq_rows.clamp_min_(0.0)

                state["exp_avg"].index_copy_(0, active_idx, exp_avg_rows)
                state["exp_avg_sq"].index_copy_(0, active_idx, exp_avg_sq_rows)

                bias1 = 1 - beta1 ** step
                bias2 = 1 - beta2 ** step
                step_size = lr * (bias2 ** 0.5 / bias1)
                update = exp_avg_rows
                exp_avg_sq_rows.sqrt_().add_(eps)
                update.div_(exp_avg_sq_rows)
                update.mul_(step_size)
                debug_raise_nonfinite("sparse_adam_tail.update", update, active_idx)
                if engram_sparse_sanitize:
                    update.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                if tail_scale != 1.0:
                    update[active_age > 0].mul_(tail_scale)
                if weight_decay:
                    mask = (update * param_rows) > 0
                    update.addcmul_(param_rows, mask, value=lr * lr * weight_decay)

                if engram_update_metrics and engram_update_metrics_every > 0 and step % engram_update_metrics_every == 0:
                    touched_update = update.index_select(0, cur_pos)
                    touched_param = param_rows.index_select(0, cur_pos)
                    update_rms = rms_no_alloc(touched_update)
                    param_rms = rms_no_alloc(touched_param)
                    grad_rms = rms_no_alloc(grad_values)
                    table_numel = max(1, param.numel())
                    self.last_update_metrics = {
                        "adam_step": step,
                        "rows": int(active_idx.numel()),
                        "active_rows": int(active_idx.numel()),
                        "touched_rows": int(idx.numel()),
                        "table_rows": int(param.shape[0]),
                        "table_numel": int(param.numel()),
                        "grad_rms": float(grad_rms.item()),
                        "update_rms": float(update_rms.item()),
                        "param_rms": float(param_rms.item()),
                        "touched_grad_rms": float(grad_rms.item()),
                        "touched_update_rms": float(update_rms.item()),
                        "touched_param_rms": float(param_rms.item()),
                        "table_grad_rms": float(table_rms_no_alloc(grad_values, table_numel).item()),
                        "table_update_rms": float(table_rms_no_alloc(update, table_numel).item()),
                        "lr": float(lr),
                        "step_size": float(step_size),
                        "hit_count_mean": float(hit_count_rows.mean().item()),
                        "hit_count_max": float(hit_count_rows.max().item()),
                    }

                param_rows.add_(update, alpha=-1.0)
                debug_raise_nonfinite("sparse_adam_tail.param_rows_after", param_rows, active_idx)
                if engram_sparse_sanitize:
                    param_rows.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                index_copy_cast_chunked_(param, 0, active_idx, param_rows)
                state["hit_count"].index_copy_(0, idx, hit_count_rows)
                state["active_rows"] = active_idx[active_age < tail_steps] if tail_steps > 0 else active_idx[:0]
                state["active_age"] = active_age[active_age < tail_steps] if tail_steps > 0 else active_age[:0]

        return loss


class SparseRowWiseAdagrad(torch.optim.Optimizer):
    """Sparse AdaGrad with one accumulated second-moment scalar per row."""
    def __init__(self, params, lr: float, eps: float = 1e-10, weight_decay: float = 0.0):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        defaults = dict(lr=lr, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.last_update_metrics: dict[str, float | int] = {}

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.last_update_metrics = {}
        for group in self.param_groups:
            lr = group["lr"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue
                if not grad.is_sparse:
                    raise RuntimeError("SparseRowWiseAdagrad requires sparse gradients")

                grad = coalesce_row_sparse_grad(grad)
                idx = grad.indices()[0]
                grad_values = grad.values().float()
                if idx.numel() == 0:
                    continue
                debug_raise_nonfinite("sparse_row_adagrad.grad_values", grad_values, idx)
                if engram_sparse_sanitize:
                    sanitize_sparse_grad_values_(grad_values)

                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["sum_sq_row"] = torch.zeros(param.shape[0], dtype=torch.float32, device=param.device)

                state["step"] += 1
                step = int(state["step"])

                param_rows = param.index_select(0, idx).float()
                if weight_decay:
                    grad_values = grad_values + weight_decay * param_rows

                sum_sq_row = state["sum_sq_row"].index_select(0, idx)
                grad_row_rms = row_rms_stable(grad_values)
                sum_sq_row.add_(grad_row_rms.square())
                debug_raise_nonfinite("sparse_row_adagrad.sum_sq_row", sum_sq_row)
                if engram_sparse_sanitize:
                    sum_sq_row.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                    sum_sq_row.clamp_min_(0.0)

                update = grad_values / sum_sq_row.sqrt().add_(eps).unsqueeze(1)
                update.mul_(lr)
                debug_raise_nonfinite("sparse_row_adagrad.update", update, idx)
                if engram_sparse_sanitize:
                    update.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

                if engram_update_metrics and engram_update_metrics_every > 0 and step % engram_update_metrics_every == 0:
                    update_rms = rms_no_alloc(update)
                    param_rms = rms_no_alloc(param_rows)
                    grad_rms = rms_no_alloc(grad_values)
                    table_numel = max(1, param.numel())
                    self.last_update_metrics = {
                        "adam_step": step,
                        "rows": int(idx.numel()),
                        "touched_rows": int(idx.numel()),
                        "table_rows": int(param.shape[0]),
                        "table_numel": int(param.numel()),
                        "grad_rms": float(grad_rms.item()),
                        "update_rms": float(update_rms.item()),
                        "param_rms": float(param_rms.item()),
                        "touched_grad_rms": float(grad_rms.item()),
                        "touched_update_rms": float(update_rms.item()),
                        "touched_param_rms": float(param_rms.item()),
                        "table_grad_rms": float(table_rms_no_alloc(grad_values, table_numel).item()),
                        "table_update_rms": float(table_rms_no_alloc(update, table_numel).item()),
                        "lr": float(lr),
                        "step_size": float(lr),
                    }

                param_rows.add_(update, alpha=-1.0)
                debug_raise_nonfinite("sparse_row_adagrad.param_rows_after", param_rows, idx)
                if engram_sparse_sanitize:
                    param_rows.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                index_copy_cast_chunked_(param, 0, idx, param_rows)
                state["sum_sq_row"].index_copy_(0, idx, sum_sq_row)

        return loss


class SparseAdamWithHitLR(torch.optim.Optimizer):
    """Sparse Adam with per-row LR scaled by cumulative raw embedding hits."""
    def __init__(self, params, lr: float, betas: tuple[float, float] = (0.75, 0.95),
                 eps: float = 1e-10, weight_decay: float = 0.0, exponent: float = 0.5,
                 min_scale: float = 0.0, max_scale: float = float("inf"), fal: bool = False,
                 ifal: bool = False):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        beta1, beta2 = betas
        if not 0 <= beta1 < 1:
            raise ValueError(f"Invalid beta1: {beta1}")
        if not 0 <= beta2 < 1:
            raise ValueError(f"Invalid beta2: {beta2}")
        if min_scale < 0 or max_scale <= 0 or min_scale > max_scale:
            raise ValueError(f"Invalid hit LR scale bounds: min={min_scale} max={max_scale}")
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            exponent=exponent,
            min_scale=min_scale,
            max_scale=max_scale,
            fal=fal,
            ifal=ifal,
        )
        super().__init__(params, defaults)
        self.last_update_metrics: dict[str, float | int] = {}

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.last_update_metrics = {}
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            exponent = group["exponent"]
            min_scale = group["min_scale"]
            max_scale = group["max_scale"]
            fal = group["fal"]
            ifal = group["ifal"]
            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue
                if not grad.is_sparse:
                    raise RuntimeError("SparseAdamWithHitLR requires sparse gradients")

                raw_idx = grad._indices()[0]
                grad = coalesce_row_sparse_grad(grad)
                idx = grad.indices()[0]
                grad_values = grad.values().float()
                if idx.numel() == 0:
                    continue
                if engram_sparse_sanitize:
                    sanitize_sparse_grad_values_(grad_values)

                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param, dtype=torch.float32, device=param.device)
                    state["exp_avg_sq"] = torch.zeros_like(param, dtype=torch.float32, device=param.device)
                    state["hit_count"] = torch.zeros(param.shape[0], dtype=torch.float32, device=param.device)

                state["step"] += 1
                step = int(state["step"])

                if raw_idx.numel() == idx.numel():
                    hit_delta = torch.ones(idx.numel(), dtype=torch.float32, device=param.device)
                else:
                    raw_unique, raw_counts = torch.unique(raw_idx, sorted=True, return_counts=True)
                    hit_delta = raw_counts.to(dtype=torch.float32).index_select(0, torch.searchsorted(raw_unique, idx))

                hit_count_rows = state["hit_count"].index_select(0, idx)
                hit_count_rows.add_(hit_delta)
                if fal:
                    max_hit = state["hit_count"].max().clamp_min(1.0)
                    denom = torch.log1p(max_hit)
                    lr_scale = torch.log1p(hit_count_rows) / denom
                    lr_scale.clamp_(min=min_scale, max=max_scale)
                elif ifal:
                    lr_scale = torch.log1p(hit_count_rows).reciprocal()
                    lr_scale.div_(lr_scale.mean().clamp_min(1e-12))
                    lr_scale.clamp_(min=min_scale, max=max_scale)
                elif exponent == 0:
                    lr_scale = torch.ones_like(hit_count_rows)
                else:
                    lr_scale = hit_count_rows.pow(exponent)
                    lr_scale.clamp_(min=min_scale, max=max_scale)
                hit_lr_blend = scheduled_scalar(
                    engram_sparse_hit_lr_blend,
                    engram_sparse_hit_lr_blend_final,
                    engram_sparse_hit_lr_blend_schedule_steps,
                    engram_sparse_hit_lr_blend_schedule_start,
                    step,
                )
                if hit_lr_blend != 1.0:
                    lr_scale.sub_(1.0).mul_(hit_lr_blend).add_(1.0)

                exp_avg_rows = state["exp_avg"].index_select(0, idx)
                exp_avg_sq_rows = state["exp_avg_sq"].index_select(0, idx)
                param_rows = param.index_select(0, idx).float()

                exp_avg_rows.mul_(beta1).add_(grad_values, alpha=1 - beta1)
                exp_avg_sq_rows.mul_(beta2).addcmul_(grad_values, grad_values, value=1 - beta2)

                bias1 = 1 - beta1 ** step
                bias2 = 1 - beta2 ** step
                step_size = lr * (bias2 ** 0.5 / bias1)
                update = exp_avg_rows / exp_avg_sq_rows.sqrt().add_(eps)
                update.mul_(step_size)
                update.mul_(lr_scale.unsqueeze(1))
                if weight_decay:
                    mask = (update * param_rows) > 0
                    update.addcmul_(param_rows, mask, value=lr * lr * weight_decay)

                if engram_update_metrics and engram_update_metrics_every > 0 and step % engram_update_metrics_every == 0:
                    update_rms = rms_no_alloc(update)
                    param_rms = rms_no_alloc(param_rows)
                    grad_rms = rms_no_alloc(grad_values)
                    table_numel = max(1, param.numel())
                    self.last_update_metrics = {
                        "adam_step": step,
                        "rows": int(idx.numel()),
                        "touched_rows": int(idx.numel()),
                        "table_rows": int(param.shape[0]),
                        "table_numel": int(param.numel()),
                        "grad_rms": float(grad_rms.item()),
                        "update_rms": float(update_rms.item()),
                        "param_rms": float(param_rms.item()),
                        "touched_grad_rms": float(grad_rms.item()),
                        "touched_update_rms": float(update_rms.item()),
                        "touched_param_rms": float(param_rms.item()),
                        "table_grad_rms": float(table_rms_no_alloc(grad_values, table_numel).item()),
                        "table_update_rms": float(table_rms_no_alloc(update, table_numel).item()),
                        "lr": float(lr),
                        "step_size": float(step_size),
                        "hit_lr_scale_mean": float(lr_scale.mean().item()),
                        "hit_lr_scale_min": float(lr_scale.min().item()),
                        "hit_lr_scale_max": float(lr_scale.max().item()),
                        "hit_lr_blend": float(hit_lr_blend),
                        "hit_count_mean": float(hit_count_rows.mean().item()),
                        "hit_count_max": float(hit_count_rows.max().item()),
                    }

                param_rows.add_(update, alpha=-1.0)
                index_copy_cast_chunked_(param, 0, idx, param_rows)
                state["exp_avg"].index_copy_(0, idx, exp_avg_rows)
                state["exp_avg_sq"].index_copy_(0, idx, exp_avg_sq_rows)
                state["hit_count"].index_copy_(0, idx, hit_count_rows)

        return loss


@dataclass(slots=True)
class ParamConfig:
    """Per-parameter configuration for NorMuonAndAdam optimizer."""
    label: str
    optim: str  # "adam" or "normuon"
    comms: str  # "none", "replicated", "sharded" or "sharded_sparse"
    adam_betas: tuple[float, float] | None
    lr_mul: float
    wd_mul: float
    lr: float
    initial_lr: float
    weight_decay: float
    # Adam-specific
    eps: float | None = None
    adam_variant: str = "standard"
    # NorMuon-specific
    reshape: tuple | None = None
    chunk_size: int | None = None
    momentum: float | None = None
    beta2: float | None = None
    per_matrix_lr_mul: list[float] | None = None


class NorMuonAndAdam:
    """
    Combined optimizer that handles both NorMuon (for projection matrices) and
    Adam (for embeddings/scalars/gate weights).

    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, Muon uses a Newton-Schulz iteration (replaced
    here with Polar Express), which has the advantage that it can be stably run in bfloat16 on the GPU.

    Muon is applied only to the projection matrices in the attention and MLP layers, and is not recommended
    for embeddings, scalars, or individual weight vectors (e.g., bias terms or gate weights).

    Differences from standard Muon:
    - Newton-Shulz is replaced with Polar Express for the orthogonalization step
    - NorMuon adds a low-rank variance estimator similar to Adafactor. https://arxiv.org/pdf/2510.05491
    - Cautious weight decay, a gated version of decoupled weight decay
    - Mantissa tracking for precision

    Adam (for embeddings/scalars/gates):
    - Standard Adam with bias correction
    - Cautious weight decay

    Configuration:
    Unlike torch.optim.Optimizer, this class uses per-parameter configs from a `param_table` dict
    and does not include parameter "groups". All parameters require a .label attribute, and a
    corresponding entry in the param_table to specify their hyperparameters (lr_mul, wd_mul, adam_betas, etc.).

    Communication and ordering:
    Gradient communication is explicitly scheduled rather than hook-driven.
    Reductions are launched in `scatter_order`, while update math and final
    gathers are executed in `work_order`. These orders are independent and
    must each contain every parameter label exactly once.

    Two communication modes are supported per parameter:
    - 'replicated': Gradients are all-reduced and each rank computes the full update.
    - 'sharded': Gradients are reduce-scattered, each rank updates its shard,
      and results are all-gathered.

    Adam parameters may be freely sharded. NorMuon operates on full matrices; sharding is
    supported by grouping matrices into parameter banks. NorMuon parameters must have a
    `.reshape` attribute that reshapes the bank so that the leading dimension is divisible
    by world_size.

    # Contributors include @YouJiacheng, @KonstantinWilleke, @alexrgilbert, @adricarda,
    # @tuttyfrutyee, @vdlad, @ryanyang0, @vagrawal, @varunneal, @chrisjmccormick
    """
    def __init__(self, named_params, param_table: dict, scatter_order: list, work_order: list,
                 adam_defaults: dict, normuon_defaults: dict):
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

        # Store defaults for each optimizer type
        self.adam_defaults = adam_defaults
        self.normuon_defaults = normuon_defaults
        self.param_table = param_table
        self.scatter_order = scatter_order
        self.work_order = work_order

        # Collect params by label and build config
        self.param_cfgs: dict[nn.Parameter, ParamConfig] = {}
        self.param_states: dict[nn.Parameter, dict] = {}
        self._param_by_label: dict[str, nn.Parameter] = {}
        for name, param in named_params:
            label = getattr(param, "label", None)
            assert label is not None and label in param_table  # all params must have valid label
            assert label not in self._param_by_label  # exactly one param per label
            self._param_by_label[label] = param
            self._build_param_cfg(param, label)

        # Assert scatter_order and work_order match present labels exactly
        present = self._param_by_label.keys()
        assert set(scatter_order) == present and set(work_order) == present

        # Handle world_size=1: overwrite comms to "none"
        if self.world_size == 1:
            for p_cfg in self.param_cfgs.values():
                p_cfg.comms = "none"

        # Initialize state for all params
        self._init_state()

        # 0-D CPU tensors to avoid recompilation
        self._step_size_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._eff_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._eff_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

        # Track async operations
        self._reduce_futures: dict[nn.Parameter, tuple] = {}
        self._sparse_async_data: dict[nn.Parameter, list] = {}
        self.last_update_metrics: dict[str, dict[str, float | int]] = {}

        # Embed/lm_head tying state
        self.split_embed = False
        self._lm_head_param = self._param_by_label.get("lm_head")
        self._embed_param = self._param_by_label.get("embed")

    def _build_param_cfg(self, param: nn.Parameter, label: str):
        """Build config for a single parameter from param_table."""
        table_entry = self.param_table[label]
        optim = table_entry["optim"]
        comms = table_entry["comms"]
        if comms == "sharded_sparse" and not _sparse_comms_active():
            comms = "sharded"
        adam_betas = table_entry.get("adam_betas")
        lr_mul = table_entry.get("lr_mul", 1.0)
        wd_mul = table_entry.get("wd_mul", 1.0)

        if optim == "adam":
            chunk_size = param.shape[0] // self.world_size if comms.startswith("sharded") else None
            adam_variant = "standard"
            if param.ndim >= 2 and label in adam_embed_vecadam_labels:
                if adam_embed_scalar_adam:
                    adam_variant = "row_scalar_both"
                elif adam_embed_vector_adam:
                    adam_variant = "row_scalar_v"
            p_cfg = ParamConfig(
                label=label,
                optim=optim,
                comms=comms,
                adam_betas=tuple(adam_betas) if adam_betas else None,
                lr_mul=lr_mul,
                wd_mul=wd_mul,
                lr=self.adam_defaults["lr"],
                initial_lr=self.adam_defaults["lr"],
                weight_decay=self.adam_defaults["weight_decay"],
                eps=self.adam_defaults["eps"],
                adam_variant=adam_variant,
                chunk_size=chunk_size,
            )
        elif optim == "normuon":
            reshape = getattr(param, "reshape", None)
            if reshape is None:
                raise ValueError(f"NorMuon param {label} must have .reshape attribute")
            if reshape[0] % self.world_size != 0:
                raise ValueError(f"reshape[0]={reshape[0]} must be divisible by world_size")

            chunk_size = reshape[0] // self.world_size
            chunk_shape = (chunk_size, *reshape[1:])
            # Shape-based LR multiplier for NorMuon
            shape_mult = max(1.0, chunk_shape[-2] / chunk_shape[-1]) ** 0.5 if len(chunk_shape) >= 2 else 1.0
            lr_mul = shape_mult * lr_mul

            # Per-matrix LR multipliers for MLP c_proj (2x LR on odd indices)
            per_matrix_lr_mul = None
            if label == "mlp_bank":
                rank = dist.get_rank() if dist.is_initialized() else 0
                start_idx = rank * chunk_size
                per_matrix_lr_mul = []
                for i in range(chunk_size):
                    global_idx = start_idx + i
                    is_c_proj = (global_idx % 2 == 1)
                    per_matrix_lr_mul.append(2.0 if is_c_proj else 1.0)

            p_cfg = ParamConfig(
                label=label,
                optim=optim,
                comms=comms,
                adam_betas=tuple(adam_betas) if adam_betas else None,
                lr_mul=lr_mul,
                wd_mul=wd_mul,
                lr=self.normuon_defaults["lr"],
                initial_lr=self.normuon_defaults["lr"],
                weight_decay=self.normuon_defaults["weight_decay"],
                reshape=reshape,
                chunk_size=chunk_size,
                momentum=self.normuon_defaults["momentum"],
                beta2=self.normuon_defaults["beta2"],
                per_matrix_lr_mul=per_matrix_lr_mul,
            )
        else:
            raise ValueError(f"Unknown optim type: {optim}")

        self.param_cfgs[param] = p_cfg

    def _init_state(self):
        """Initialize optimizer state for all parameters."""
        for param, p_cfg in self.param_cfgs.items():
            if p_cfg.optim == "adam":
                # Sharded params use chunk state, replicated use full state
                if p_cfg.comms.startswith("sharded"):
                    chunk = param[:p_cfg.chunk_size]
                else:
                    chunk = param
                if p_cfg.adam_variant == "row_scalar_both":
                    self.param_states[param] = dict(
                        step=0,
                        exp_avg_row=torch.zeros(chunk.shape[0], dtype=torch.float32, device=param.device),
                        exp_avg_sq_row=torch.zeros(chunk.shape[0], dtype=torch.float32, device=param.device),
                    )
                elif p_cfg.adam_variant == "row_scalar_v":
                    exp_avg = torch.zeros_like(chunk, dtype=torch.float32, device=param.device)
                    self.param_states[param] = dict(
                        step=0,
                        exp_avg=exp_avg,
                        exp_avg_sq_row=torch.zeros(chunk.shape[0], dtype=torch.float32, device=param.device),
                    )
                else:
                    exp_avg = torch.zeros_like(chunk, dtype=torch.float32, device=param.device)
                    self.param_states[param] = dict(step=0, exp_avg=exp_avg, exp_avg_sq=torch.zeros_like(exp_avg))

            elif p_cfg.optim == "normuon":
                chunk_shape = (p_cfg.chunk_size, *p_cfg.reshape[1:])

                # Momentum buffer (FP32 for precision)
                momentum_buffer = torch.zeros(
                    chunk_shape, dtype=torch.float32, device=param.device
                )

                # Second momentum buffer - reduced along one dimension
                if chunk_shape[-2] >= chunk_shape[-1]:
                    second_mom_shape = (*chunk_shape[:-1], 1)
                else:
                    second_mom_shape = (*chunk_shape[:-2], 1, chunk_shape[-1])
                second_momentum_buffer = torch.zeros(
                    second_mom_shape, dtype=torch.float32, device=param.device
                )

                # Mantissa buffer for precision tracking
                mantissa = torch.zeros(
                    chunk_shape, dtype=torch.uint16, device=param.device
                )

                self.param_states[param] = dict(
                    momentum_buffer=momentum_buffer,
                    second_momentum_buffer=second_momentum_buffer,
                    mantissa=mantissa,
                    update_smoothing_buffer=torch.zeros(
                        chunk_shape, dtype=torch.bfloat16, device=param.device
                    ) if normuon_update_smoothing > 0 else None,
                )

    # -----------------------------------
    # Reduce/Gather operations

    def _launch_reduce(self, param: nn.Parameter, grad: Tensor):
        """Launch async reduce for a parameter based on its comms policy."""
        p_cfg = self.param_cfgs[param]

        if p_cfg.comms == "none":
            if p_cfg.optim == "normuon":
                # NorMuon needs reshaped gradient even without communication
                grad = grad.view(p_cfg.reshape)
            self._reduce_futures[param] = (None, grad)
        elif p_cfg.comms == "replicated":
            future = dist.all_reduce(grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
            self._reduce_futures[param] = (future, grad)
        elif p_cfg.comms == "sharded":
            if p_cfg.optim == "normuon":
                # NorMuon: reshape before reduce_scatter
                grad_reshaped = grad.view(p_cfg.reshape)
                grad_chunk = torch.empty(
                    (p_cfg.chunk_size, *grad_reshaped.shape[1:]),
                    dtype=grad.dtype,
                    device=grad.device
                )
                future = dist.reduce_scatter_tensor(
                    grad_chunk, grad_reshaped.contiguous(), op=dist.ReduceOp.AVG, async_op=True
                ).get_future()
                self._reduce_futures[param] = (future, grad_chunk)
            else:
                # Adam: simple reduce_scatter
                grad_chunk = torch.empty_like(grad[:p_cfg.chunk_size])
                future = dist.reduce_scatter_tensor(
                    grad_chunk, grad, op=dist.ReduceOp.AVG, async_op=True
                ).get_future()
                self._reduce_futures[param] = (future, grad_chunk)
        elif p_cfg.comms == "sharded_sparse":
            sparse_state = self._sparse_async_data[param]
            send_idxes = sparse_state["send_idxes"]
            send_counts = sparse_state["send_counts"]
            recv_counts = sparse_state["recv_counts"]
            recv_vals, val_fut = sparse_comms_share_gradients(
                grad, send_idxes, send_counts, recv_counts
            )
            self._reduce_futures[param].extend((val_fut, recv_vals))

    def _launch_gather(self, param: nn.Parameter, p_slice: Tensor) -> "torch.futures.Future":
        """Launch async all_gather for a sharded parameter."""
        p_cfg = self.param_cfgs[param]
        if p_cfg.optim == "normuon":
            full_param = param.data.view(p_cfg.reshape)
            assert full_param.is_contiguous()
            return dist.all_gather_into_tensor(
                full_param, p_slice.contiguous(), async_op=True
            ).get_future()
        else:
            return dist.all_gather_into_tensor(
                param, p_slice.contiguous(), async_op=True
            ).get_future()

    # -----------------------------------
    # State management

    def reset(self):
        """Reset NorMuon momentum buffers and split_embed state (called on training reset)."""
        self.split_embed = False
        self.last_update_metrics.clear()
        for param, p_cfg in self.param_cfgs.items():
            if p_cfg.optim == "normuon":
                p_state = self.param_states[param]
                p_state["momentum_buffer"].zero_()
                p_state["mantissa"].zero_()
                p_state["second_momentum_buffer"].zero_()
                if p_state.get("update_smoothing_buffer") is not None:
                    p_state["update_smoothing_buffer"].zero_()

    def copy_lm_state_to_embed(self):
        """
        Copy the optimizer state from the lm_head to the embed at the untie point.
        This requires an all-gather + reshard because of different sharding:
        - lm_head (768, 50304) is sharded to (96, 50304) per rank (along model_dim)
        - embed (50304, 768) is sharded to (6288, 768) per rank (along vocab_size)

        We all-gather the lm_head momentum, transpose it, then each rank takes their
        embed shard to get the correct momentum state.
        """
        lm_head = self._lm_head_param
        embed = self._embed_param
        lm_state = self.param_states[lm_head]
        embed_state = self.param_states[embed]
        lm_cfg = self.param_cfgs[lm_head]
        embed_cfg = self.param_cfgs[embed]

        embed_state['step'] = lm_state['step'] # Preserve step count for bias correction

        # Copy optimizer state with all-gather + transpose + reshard
        if self.world_size > 1:
            rank = dist.get_rank()
            lm_chunk_size = lm_cfg.chunk_size  # 96
            embed_chunk_size = embed_cfg.chunk_size  # 6288

            # All-gather lm_head momentum to get full (768, 50304) tensor.
            lm_exp_avg = torch.empty(lm_head.shape[0], lm_head.shape[1], dtype=lm_state["exp_avg"].dtype, device=lm_state["exp_avg"].device)
            dist.all_gather_into_tensor(lm_exp_avg, lm_state["exp_avg"].contiguous())
            embed_exp_avg = lm_exp_avg.T[rank * embed_chunk_size:(rank + 1) * embed_chunk_size]
            if "exp_avg" in embed_state:
                embed_state["exp_avg"].copy_(embed_exp_avg)
            else:
                embed_state["exp_avg_row"].copy_(row_rms_stable(embed_exp_avg.float()))
            lm_exp_avg_sq = torch.empty(lm_head.shape[0], lm_head.shape[1], dtype=lm_state["exp_avg_sq"].dtype, device=lm_state["exp_avg_sq"].device)
            dist.all_gather_into_tensor(lm_exp_avg_sq, lm_state["exp_avg_sq"].contiguous())
            embed_exp_avg_sq = lm_exp_avg_sq.T[rank * embed_chunk_size:(rank + 1) * embed_chunk_size]
            if "exp_avg_sq" in embed_state:
                embed_state["exp_avg_sq"].copy_(embed_exp_avg_sq)
            else:
                embed_state["exp_avg_sq_row"].copy_(embed_exp_avg_sq.float().mean(dim=1))
        else:
            # Single GPU: simple transpose
            embed_exp_avg = lm_state["exp_avg"].T
            if "exp_avg" in embed_state:
                embed_state["exp_avg"].copy_(embed_exp_avg)
            else:
                embed_state["exp_avg_row"].copy_(row_rms_stable(embed_exp_avg.float()))
            embed_exp_avg_sq = lm_state["exp_avg_sq"].T
            if "exp_avg_sq" in embed_state:
                embed_state["exp_avg_sq"].copy_(embed_exp_avg_sq)
            else:
                embed_state["exp_avg_sq_row"].copy_(embed_exp_avg_sq.float().mean(dim=1))

        # Mark as split
        self.split_embed = True

    def state_dict(self):
        """Return the optimizer state as a dict."""
        return {
            "param_states": {id(p): s for p, s in self.param_states.items()},
            "param_cfgs": {id(p): s for p, s in self.param_cfgs.items()},
        }

    def load_state_dict(self, state_dict):
        """Load optimizer state from a dict."""
        # Build id->param mapping
        id_to_param = {id(p): p for p in self.param_cfgs}

        # Load state, preserving dtypes
        for param_id, saved_p_state in state_dict["param_states"].items():
            if param_id in id_to_param:
                param = id_to_param[param_id]
                p_state = self.param_states[param]
                for k, v in saved_p_state.items():
                    if isinstance(v, torch.Tensor) and k in p_state:
                        target_dtype = p_state[k].dtype
                        p_state[k] = v.to(dtype=target_dtype, device=p_state[k].device)
                    else:
                        p_state[k] = v

    # -----------------------------------
    # Unified optimizer step with explicit ordering

    @torch.no_grad()
    def step(self, do_adam: bool = True, adam_every_step_labels: set[str] | None = None):
        """
        Combined optimizer step with explicit ordering.

        Args:
            do_adam: If True, update Adam params. NorMuon params always updated.

        Flow:
        1. Scatter phase: Launch reduces in scatter_order
        2. Work phase: Process updates in work_order
           - Wait for reduce, compute update, launch gather
        3. Finalize phase: Wait for gathers

        While the embeddings are tied:
        - Comms and update math are only done on lm_head.
        - We add embed.grad.T into lm_head.grad before comms.
        - After lm_head gather, we copy lm_head.data.T --> embed.data
        """
        rank = dist.get_rank() if dist.is_initialized() else 0
        lm_param, embed_param = self._lm_head_param, self._embed_param
        adam_every_step_labels = adam_every_step_labels or set()

        # ===== Phase 1: Launch reduces in scatter_order =====
        for label in self.scatter_order:
            param = self._param_by_label[label]
            p_cfg = self.param_cfgs[param]

            update_adam = do_adam or label in adam_every_step_labels
            if p_cfg.optim == "adam" and not update_adam:
                continue
            if param.grad is None:
                continue

            # lm_head when tied: aggregate embed.grad.T (tiled Triton transpose-add)
            if label == "lm_head" and update_adam and not self.split_embed:
                if embed_param is not None and embed_param.grad is not None:
                    transpose_add(embed_param.grad, param.grad)

            # Skip embed when tied (copied from lm_head after gather)
            if label == "embed" and not self.split_embed:
                continue

            self._launch_reduce(param, param.grad)

        # ===== Phase 2: Process updates in work_order =====
        gather_futures = []
        lm_head_gather_future = None

        for label in self.work_order:
            param = self._param_by_label[label]
            if param not in self._reduce_futures:
                continue

            p_cfg = self.param_cfgs[param]
            update_adam = do_adam or label in adam_every_step_labels
            if p_cfg.optim == "adam" and not update_adam:
                continue
            # Wait for reduce
            if p_cfg.comms != "sharded_sparse":
                future, grad_chunk = self._reduce_futures[param]
                if future is not None:
                    future.wait()
            else:
                idxes_fut, recv_idxes, recv_fut, recv_vals = self._reduce_futures[param]
                idxes_fut.wait()
                recv_fut.wait()

                grad_chunk = sparse_comms_merge_gradients(param.grad, recv_idxes, recv_vals, rank, world_size)

            # Apply update based on optim type
            if p_cfg.optim == "adam":
                p_slice = self._adam_update(param, grad_chunk, p_cfg, rank)
            else:
                p_slice = self._normuon_update(param, grad_chunk, p_cfg, rank)
            # Launch gather for sharded params
            if p_cfg.comms.startswith("sharded") and self.world_size > 1:
                gather_fut = self._launch_gather(param, p_slice)
                if label == "lm_head":
                    lm_head_gather_future = gather_fut
                else:
                    gather_futures.append(gather_fut)

        # ===== Phase 3: Wait for gathers, sync embed if tied =====
        # Wait for lm_head gather first so we can copy to embed while other gathers complete
        if lm_head_gather_future is not None:
            lm_head_gather_future.wait()

        # When tied: copy lm_head.T to embed (tiled Triton transpose for coalesced writes)
        if do_adam and not self.split_embed and embed_param is not None and lm_param is not None:
            transpose_copy(lm_param.data, embed_param.data)

        # Wait for remaining gathers
        for fut in gather_futures:
            fut.wait()

        self._reduce_futures.clear()
        self._sparse_async_data.clear()

        # Clear grads for updated params
        for label, param in self._param_by_label.items():
            p_cfg = self.param_cfgs[param]
            update_adam = do_adam or label in adam_every_step_labels
            if p_cfg.optim == "adam" and not update_adam:
                continue  # Don't clear Adam grads on even steps
            param.grad = None

    # -----------------------------------
    # Adam update

    def _adam_update(self, param: nn.Parameter, grad_chunk: Tensor, p_cfg: ParamConfig, rank: int) -> Tensor:
        """Apply Adam update to a parameter. Returns the updated p_slice."""
        beta1, beta2 = p_cfg.adam_betas
        lr = p_cfg.lr * p_cfg.lr_mul

        # Get parameter slice
        if p_cfg.comms.startswith("sharded"):
            p_slice = param[rank * p_cfg.chunk_size:(rank + 1) * p_cfg.chunk_size]
        else:
            p_slice = param

        p_state = self.param_states[param]
        p_state["step"] += 1
        t = p_state["step"]

        bias1, bias2 = 1 - beta1 ** t, 1 - beta2 ** t
        self._step_size_t.fill_(lr * (bias2 ** 0.5 / bias1))
        self._eff_wd_t.fill_(lr * lr * p_cfg.weight_decay * p_cfg.wd_mul)

        if p_cfg.adam_variant == "row_scalar_v":
            NorMuonAndAdam._adam_update_row_scalar_v(
                p_slice, grad_chunk, p_state["exp_avg"], p_state["exp_avg_sq_row"],
                beta1, beta2, p_cfg.eps, self._step_size_t, self._eff_wd_t
            )
            return p_slice
        if p_cfg.adam_variant == "row_scalar_both":
            NorMuonAndAdam._adam_update_row_scalar_both(
                p_slice, grad_chunk, p_state["exp_avg_row"], p_state["exp_avg_sq_row"],
                beta1, beta2, p_cfg.eps, self._step_size_t, self._eff_wd_t
            )
            return p_slice

        metrics_enabled = (
            engram_update_metrics
            and p_cfg.label == "bigram_embed.embedding"
            and engram_update_metrics_every > 0
            and t % engram_update_metrics_every == 0
        )
        if metrics_enabled:
            touched_rows = None
            touched_before = None
            touched_grad_rms = None
            if engram_touched_update_metrics and grad_chunk.ndim >= 2:
                touched_parts = []
                chunk_rows = max(1, engram_touched_update_metrics_chunk)
                for row_start in range(0, grad_chunk.shape[0], chunk_rows):
                    grad_part = grad_chunk[row_start:row_start + chunk_rows]
                    touched_part = grad_part.detach().abs().sum(dim=1) > 0
                    if bool(touched_part.any().item()):
                        touched_parts.append(touched_part.nonzero(as_tuple=False).flatten() + row_start)
                if touched_parts:
                    touched_rows = torch.cat(touched_parts)
                    touched_before = p_slice.index_select(0, touched_rows).float()
                    touched_grad = grad_chunk.index_select(0, touched_rows).float()
                    touched_grad_rms = rms_no_alloc(touched_grad)
            grad_rms, update_rms, param_rms = NorMuonAndAdam._adam_update_step_with_metrics(
                p_slice, grad_chunk, p_state["exp_avg"], p_state["exp_avg_sq"],
                beta1, beta2, p_cfg.eps, self._step_size_t, self._eff_wd_t
            )
            metrics = {
                "adam_step": t,
                "rows": int(grad_chunk.shape[0]),
                "table_rows": int(grad_chunk.shape[0]),
                "table_numel": int(grad_chunk.numel()),
                "grad_rms": float(grad_rms.item()),
                "update_rms": float(update_rms.item()),
                "param_rms": float(param_rms.item()),
                "table_grad_rms": float(grad_rms.item()),
                "table_update_rms": float(update_rms.item()),
                "lr": float(lr),
                "step_size": float(self._step_size_t.item()),
            }
            if touched_rows is not None and touched_before is not None and touched_grad_rms is not None:
                touched_after = p_slice.index_select(0, touched_rows).float()
                touched_update_rms = rms_no_alloc(touched_after - touched_before)
                touched_param_rms = rms_no_alloc(touched_before)
                metrics.update({
                    "touched_rows": int(touched_rows.numel()),
                    "touched_grad_rms": float(touched_grad_rms.item()),
                    "touched_update_rms": float(touched_update_rms.item()),
                    "touched_param_rms": float(touched_param_rms.item()),
                })
            self.last_update_metrics[p_cfg.label] = metrics
        else:
            NorMuonAndAdam._adam_update_step(
                p_slice, grad_chunk, p_state["exp_avg"], p_state["exp_avg_sq"],
                beta1, beta2, p_cfg.eps, self._step_size_t, self._eff_wd_t
            )

        return p_slice

    @staticmethod
    @torch.compile(dynamic=False, fullgraph=True)
    def _adam_update_row_scalar_v(p_slice, g_slice, exp_avg, exp_avg_sq_row, beta1, beta2, eps, step_size_t, eff_wd_t):
        """Adam update with vector first moment and one second-moment scalar per row."""
        g_float = g_slice.float()
        exp_avg.mul_(beta1).add_(g_slice, alpha=1 - beta1)
        exp_avg_sq_row.mul_(beta2).add_(g_float.square().flatten(start_dim=1).mean(dim=1), alpha=1 - beta2)
        update = exp_avg.div(exp_avg_sq_row.sqrt().add_(eps).view(-1, *([1] * (exp_avg.ndim - 1)))).mul_(step_size_t)
        mask = (update * p_slice) > 0
        update.addcmul_(p_slice, mask, value=eff_wd_t)
        p_slice.add_(other=update, alpha=-1.0)

    @staticmethod
    @torch.compile(dynamic=False, fullgraph=True)
    def _adam_update_row_scalar_both(p_slice, g_slice, exp_avg_row, exp_avg_sq_row, beta1, beta2, eps, step_size_t, eff_wd_t):
        """Adam update with row-scalar first-moment magnitude and row-scalar second moment."""
        g_float = g_slice.float()
        grad_row_rms = g_float.square().flatten(start_dim=1).mean(dim=1).sqrt()
        exp_avg_row.mul_(beta1).add_(grad_row_rms, alpha=1 - beta1)
        exp_avg_sq_row.mul_(beta2).add_(grad_row_rms.square(), alpha=1 - beta2)
        view_shape = (-1, *([1] * (g_slice.ndim - 1)))
        direction = g_float / grad_row_rms.clamp_min(eps).view(view_shape)
        update = direction * (exp_avg_row / exp_avg_sq_row.sqrt().add_(eps)).view(view_shape)
        update.mul_(step_size_t)
        mask = (update * p_slice) > 0
        update.addcmul_(p_slice, mask, value=eff_wd_t)
        p_slice.add_(other=update.to(dtype=p_slice.dtype), alpha=-1.0)

    @staticmethod
    @torch.compile(dynamic=False, fullgraph=True)
    def _adam_update_step(p_slice, g_slice, exp_avg, exp_avg_sq, beta1, beta2, eps, step_size_t, eff_wd_t):
        """Compiled Adam update step."""
        exp_avg.mul_(beta1).add_(g_slice, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(g_slice, g_slice, value=1 - beta2)
        update = exp_avg.div(exp_avg_sq.sqrt().add_(eps)).mul_(step_size_t)
        # Cautious weight decay
        mask = (update * p_slice) > 0
        update.addcmul_(p_slice, mask, value=eff_wd_t)
        p_slice.add_(other=update, alpha=-1.0)

    @staticmethod
    @torch.compile(dynamic=False, fullgraph=True)
    def _adam_update_step_with_metrics(p_slice, g_slice, exp_avg, exp_avg_sq, beta1, beta2, eps, step_size_t, eff_wd_t):
        """Compiled Adam update step with RMS metrics for the Engram table."""
        grad_float = g_slice.float()
        param_float = p_slice.float()
        grad_rms = grad_float.norm() / (grad_float.numel() ** 0.5)
        param_rms = param_float.norm() / (param_float.numel() ** 0.5)
        exp_avg.mul_(beta1).add_(g_slice, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(g_slice, g_slice, value=1 - beta2)
        update = exp_avg.div(exp_avg_sq.sqrt().add_(eps)).mul_(step_size_t)
        # Cautious weight decay
        mask = (update * p_slice) > 0
        update.addcmul_(p_slice, mask, value=eff_wd_t)
        update_float = update.float()
        update_rms = update_float.norm() / (update_float.numel() ** 0.5)
        p_slice.add_(other=update, alpha=-1.0)
        return grad_rms, update_rms, param_rms

    # -----------------------------------
    # NorMuon update

    def _normuon_update(self, param: nn.Parameter, grad_chunk: Tensor, p_cfg: ParamConfig, rank: int) -> Tensor:
        """Apply NorMuon update to a parameter. Returns the updated p_slice."""
        chunk_shape = grad_chunk.shape

        p_state = self.param_states[param]
        grad_chunk = grad_chunk.float()  # FP32 for momentum
        debug_check_finite(f"normuon.{p_cfg.label}.grad_chunk", grad_chunk)

        self._momentum_t.fill_(p_cfg.momentum)
        self._eff_lr_t.fill_(p_cfg.lr_mul * p_cfg.lr)
        self._eff_wd_t.fill_(p_cfg.wd_mul * p_cfg.weight_decay * p_cfg.lr)

        # Fused Nesterov momentum + Polar Express orthogonalization
        is_large_matrix = chunk_shape[-2] > 1024
        v_chunk = polar_express(
            grad_chunk, p_state["momentum_buffer"], self._momentum_t,
            split_baddbmm=is_large_matrix,
        )
        debug_check_finite(f"normuon.{p_cfg.label}.polar_v", v_chunk)

        # Variance reduction
        red_dim = -1 if chunk_shape[-2] >= chunk_shape[-1] else -2
        v_chunk = NorMuonAndAdam._apply_normuon_variance_reduction(
            v_chunk, p_state["second_momentum_buffer"], p_cfg.beta2, red_dim
        )
        debug_check_finite(f"normuon.{p_cfg.label}.variance_reduced_v", v_chunk)
        if normuon_update_smoothing > 0:
            smoothing_buffer = p_state["update_smoothing_buffer"]
            smoothing_buffer.copy_(
                normuon_update_smoothing * smoothing_buffer
                + (1.0 - normuon_update_smoothing) * v_chunk.to(dtype=smoothing_buffer.dtype)
            )
            v_chunk = smoothing_buffer.to(dtype=v_chunk.dtype)
            debug_check_finite(f"normuon.{p_cfg.label}.update_smoothed_v", v_chunk)

        # Update parameter, in place, with cautious weight decay
        param_view = param.data.view(p_cfg.reshape)
        p_slice = param_view[rank * p_cfg.chunk_size:(rank + 1) * p_cfg.chunk_size]

        # MLP has per-matrix LR multipliers (c_proj gets 2x LR)
        if p_cfg.per_matrix_lr_mul is not None:
            self._eff_wd_t.fill_(p_cfg.wd_mul * p_cfg.weight_decay * p_cfg.lr)
            for mat_idx in range(p_cfg.chunk_size):
                self._eff_lr_t.fill_(p_cfg.lr_mul * p_cfg.per_matrix_lr_mul[mat_idx] * p_cfg.lr)
                NorMuonAndAdam._cautious_wd_and_update_inplace(
                    p_slice[mat_idx].view(torch.uint16), p_state["mantissa"][mat_idx], v_chunk[mat_idx],
                    self._eff_wd_t, self._eff_lr_t
                )
        else:
            NorMuonAndAdam._cautious_wd_and_update_inplace(
                p_slice.view(torch.uint16), p_state["mantissa"], v_chunk,
                self._eff_wd_t, self._eff_lr_t
            )

        return p_slice

    @staticmethod
    @torch.compile(dynamic=False, fullgraph=True)
    def _cautious_wd_and_update_inplace(p, mantissa, grad, wd_tensor, lr_tensor):
        """
        Cautious weight decay + parameter update. wd_tensor and lr_tensor are 0-D CPU tensors.
        Mantissa is tracked to enable higher precision updates on bfloat16 parameters.
        bfloat16 format: 1 sign bit + 8 exponent bits + 7 mantissa bits = 16 bits total
        float32 format: 1 sign bit + 8 exponent bits + 23 mantissa bits = 32 bits total
        """
        assert p.dtype == mantissa.dtype == torch.uint16
        grad = grad.float()
        wd_factor = wd_tensor.to(torch.float32)
        lr_factor = lr_tensor.to(torch.float32)
        p_precise_raw = (p.to(torch.uint32) << 16) | mantissa.to(torch.uint32)
        p_precise = p_precise_raw.view(torch.float32)
        mask = (grad * p_precise) >= 0
        p_precise.copy_(p_precise - (p_precise * mask * wd_factor * lr_factor) - (grad * lr_factor))
        p.copy_((p_precise_raw >> 16).to(torch.uint16))
        mantissa.copy_(p_precise_raw.to(torch.uint16))

    @staticmethod
    @torch.compile(dynamic=False, fullgraph=True)
    def _apply_normuon_variance_reduction(v_chunk, second_momentum_buffer, beta2, red_dim):
        """NorMuon variance reduction. Algebraically fuses the normalization steps to minimize memory ops."""
        v_mean = v_chunk.float().square().mean(dim=red_dim, keepdim=True)
        red_dim_size = v_chunk.size(red_dim)
        v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True).mul_(red_dim_size)
        v_norm = v_norm_sq.sqrt_()
        second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
        step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt_()
        scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
        v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt_()
        final_scale = step_size * (v_norm / v_norm_new.clamp_min_(1e-10))
        return v_chunk.mul_(final_scale.type_as(v_chunk))

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the model

def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))


def debug_check_finite(name: str, x: Tensor):
    if not rom_debug_nan or rom_debug_nan_current_step < rom_debug_nan_min_step:
        return
    if torch.isfinite(x).all():
        return
    x_detached = x.detach().float()
    absmax = torch.nan_to_num(x_detached, nan=0.0, posinf=float("inf"), neginf=float("inf")).abs().amax().item()
    finite_frac = torch.isfinite(x_detached).float().mean().item()
    raise RuntimeError(f"ROM_DEBUG_NAN nonfinite {name}: shape={tuple(x.shape)} dtype={x.dtype} finite_frac={finite_frac:.6f} absmax={absmax:.6g}")


def debug_report_engram_backward(name: str, grad: Tensor, addresses: Tensor | None = None) -> Tensor:
    if not engram_debug_backward or not rom_debug_nan or rom_debug_nan_current_step < rom_debug_nan_min_step:
        return grad
    with torch.no_grad():
        grad_f = grad.detach().float()
        finite = torch.isfinite(grad_f)
        clean_abs = torch.nan_to_num(grad_f, nan=0.0, posinf=0.0, neginf=0.0).abs()
        huge = clean_abs > 1.0e18
        if bool(finite.all().item()) and not bool(huge.any().item()):
            return grad
        nonfinite = ~finite
        nonfinite_count = int(nonfinite.sum().item())
        huge_count = int(huge.sum().item())
        finite_count = grad.numel() - nonfinite_count
        row_info = ""
        if addresses is not None and grad.ndim >= 3:
            bad_site = (nonfinite | huge).flatten(start_dim=2).any(dim=-1)
            bad_pos = torch.nonzero(bad_site, as_tuple=False)
            if bad_pos.numel() > 0:
                first = bad_pos[0]
                token_pos = int(first[0].item())
                head_pos = int(first[1].item())
                row_info = (
                    f" bad_sites={int(bad_pos.size(0))}"
                    f" first_token={token_pos} first_head={head_pos}"
                    f" first_row={int(addresses[token_pos, head_pos].item())}"
                )
        print(
            f"ROM_DEBUG_ENGRAM_BACKWARD step={rom_debug_nan_current_step} {name}:"
            f" shape={tuple(grad.shape)} dtype={grad.dtype}"
            f" finite_count={finite_count} nonfinite_count={nonfinite_count}"
            f" huge_count={huge_count} finite_frac={finite_count / max(1, grad.numel()):.9f}"
            f" absmax_finite={float(clean_abs.amax().item()):.6g}{row_info}",
            flush=True,
        )
    return grad


def is_torch_compiling() -> bool:
    compiler = getattr(torch, "compiler", None)
    if compiler is not None and hasattr(compiler, "is_compiling"):
        return bool(compiler.is_compiling())
    dynamo = getattr(torch, "_dynamo", None)
    return bool(dynamo is not None and hasattr(dynamo, "is_compiling") and dynamo.is_compiling())


def apply_rom_short_conv(value: Tensor, short_conv_norm: nn.Module | None, short_conv_module: nn.Module | None) -> Tensor:
    if short_conv_module is None:
        return value
    debug_check_finite("rom_short_conv.value", value)
    conv_in = short_conv_norm(value).T.unsqueeze(0)
    debug_check_finite("rom_short_conv.input", conv_in)
    conv_out = short_conv_module(conv_in)[..., :value.size(0)].squeeze(0).T
    debug_check_finite("rom_short_conv.output", conv_out)
    return value + F.silu(conv_out)


def apply_rom_ema_smooth(value: Tensor) -> Tensor:
    if not rom_ema_smooth:
        return value
    debug_check_finite("rom_ema_smooth.value", value)
    kernel = min(rom_ema_kernel, value.size(0))
    ages = torch.arange(kernel, device=value.device, dtype=torch.float32)
    weights = torch.pow(torch.full_like(ages, rom_ema_alpha), ages)
    weights = weights / weights.sum().clamp_min(1e-12)
    weights = weights.flip(0).to(dtype=value.dtype).view(1, 1, kernel).expand(value.size(-1), 1, kernel)
    smooth = F.conv1d(F.pad(value.T.unsqueeze(0), (kernel - 1, 0)), weights, groups=value.size(-1)).squeeze(0).T
    debug_check_finite("rom_ema_smooth.output", smooth)
    return smooth


def apply_rom_read_mlp(read: Tensor, norm_module: nn.Module | None, fc1: nn.Module | None, fc2: nn.Module | None) -> Tensor:
    if fc1 is None or fc2 is None or norm_module is None:
        return read
    debug_check_finite("rom_read_mlp.input", read)
    delta = fc2(F.silu(fc1(norm_module(read))))
    debug_check_finite("rom_read_mlp.delta", delta)
    return read + delta


def engram_gate_value(memory_features: Tensor, hidden_states: Tensor, value_proj: nn.Module, key_proj: nn.Module, short_conv_norm: nn.Module | None = None, short_conv_module: nn.Module | None = None) -> Tensor:
    debug_check_finite("engram_gate.memory_features", memory_features)
    debug_check_finite("engram_gate.hidden_states", hidden_states)
    value = value_proj(memory_features)
    debug_check_finite("engram_gate.value", value)
    key_raw = key_proj(memory_features)
    debug_check_finite("engram_gate.key_raw", key_raw)
    key = norm(key_raw)
    debug_check_finite("engram_gate.key", key)
    query = norm(hidden_states)
    debug_check_finite("engram_gate.query", query)
    gate = (key * query).sum(dim=-1) / math.sqrt(hidden_states.size(-1))
    debug_check_finite("engram_gate.dot_gate", gate)
    gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
    debug_check_finite("engram_gate.signed_sqrt_gate", gate)
    output = value * torch.sigmoid(gate).unsqueeze(-1)
    debug_check_finite("engram_gate.output", output)
    output = apply_rom_short_conv(output, short_conv_norm, short_conv_module)
    output = apply_rom_ema_smooth(output)
    if engram_normalize_readout:
        output = norm(output)
        debug_check_finite("engram_gate.normalized_output", output)
        return output
    return rom_output_scale * output


def init_rom_state_weight(weight: Tensor, num_heads: int, key_dim: int, value_dim: int) -> None:
    with torch.no_grad():
        weight.zero_()
        view = weight.view(weight.shape[0], num_heads, key_dim, value_dim)
        if rom_state_diag_init:
            diag_dim = min(key_dim, value_dim)
            diag = torch.empty((weight.shape[0], num_heads, diag_dim), device=weight.device, dtype=torch.float32)
            diag.normal_(std=rom_state_init_std)
            for d in range(diag_dim):
                view[:, :, d, d] = diag[:, :, d].to(dtype=view.dtype)
        elif rom_state_init_std > 0:
            view.normal_(std=rom_state_init_std)
        if rom_state_frob_norm > 0:
            flat = view.float().flatten(start_dim=1)
            row_norm = flat.square().sum(dim=1, keepdim=True).sqrt()
            scale = rom_state_frob_norm / row_norm.clamp_min(1e-12)
            view.mul_(scale.view(-1, 1, 1, 1).to(dtype=view.dtype))


class RomBigramMemory(nn.Module):
    """Read a compact gated-delta-style state matrix from each bigram hash row."""

    def __init__(self, num_rows: int, model_dim: int, num_heads: int, key_dim: int, value_dim: int, *, enable_write: bool = False):
        super().__init__()
        if min(num_rows, model_dim, num_heads, key_dim, value_dim) <= 0:
            raise ValueError("ROM dimensions must be positive")
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        if rom_mqa and model_dim % key_dim != 0:
            raise ValueError("ROM_MQA requires model_dim to be divisible by key_dim")
        self.num_read_queries = model_dim // key_dim if rom_mqa else 1
        self.num_read_heads = num_heads * self.num_read_queries
        self.enable_write = enable_write
        if rom_state_sparse_embedding:
            self.state = nn.Embedding(num_rows, num_heads * key_dim * value_dim, sparse=True, dtype=torch.bfloat16)
            init_rom_state_weight(self.state.weight, num_heads, key_dim, value_dim)
        else:
            self.state = nn.Parameter(torch.zeros(num_rows, num_heads, key_dim, value_dim, dtype=torch.bfloat16))
            init_rom_state_weight(self.state, num_heads, key_dim, value_dim)
        if rom_state_hit_rms_low > 0 and rom_state_hit_rms_high > 0:
            self.register_buffer("hit_hist", torch.zeros(num_rows, dtype=torch.int32), persistent=False)
        else:
            self.hit_hist = None
        self._recovered_write_rows: list[Tensor] = []
        self._recovered_write_updates: list[Tensor] = []
        self.q_proj = nn.Linear(model_dim, self.num_read_heads * key_dim, bias=False)
        if rom_engram_gate:
            self.engram_key_proj = nn.Linear(self.num_read_heads * value_dim, model_dim, bias=False)
        else:
            self.gate_proj = nn.Linear(model_dim, self.num_read_heads, bias=False)
        self.read_dim = self.num_read_heads * value_dim
        if rom_read_mlp:
            read_hidden_dim = max(1, int(round(self.read_dim * rom_read_mlp_hidden_mult)))
            self.read_mlp_norm = nn.RMSNorm(self.read_dim)
            self.read_mlp_fc1 = nn.Linear(self.read_dim, read_hidden_dim, bias=False)
            self.read_mlp_fc2 = nn.Linear(read_hidden_dim, self.read_dim, bias=False)
            nn.init.zeros_(self.read_mlp_fc2.weight)
        self.out_proj = nn.Linear(self.num_read_heads * value_dim, model_dim, bias=False)
        nn.init.zeros_(self.out_proj.weight)
        if rom_short_conv:
            self.short_conv_norm = nn.RMSNorm(model_dim)
            self.short_conv = nn.Conv1d(model_dim, model_dim, rom_short_conv_kernel, groups=model_dim, bias=False, padding=rom_short_conv_kernel - 1)
            nn.init.zeros_(self.short_conv.weight)
        if enable_write:
            self.k_proj = nn.Linear(model_dim, num_heads * key_dim, bias=False)
            self.v_proj = nn.Linear(model_dim, num_heads * value_dim, bias=False)
            self.beta_proj = nn.Linear(model_dim, num_heads, bias=False)
            self.decay_proj = nn.Linear(model_dim, num_heads, bias=False)
            self.write_gate_proj = nn.Linear(model_dim, num_heads, bias=False)
            self.write_alpha = nn.Parameter(torch.zeros(1))

    def forward(self, addresses: Tensor, hidden_states: Tensor) -> Tensor:
        assert addresses.ndim == 1
        assert hidden_states.ndim == 2
        assert addresses.size(0) == hidden_states.size(0)
        q = self.q_proj(hidden_states).view(-1, self.num_heads, self.num_read_queries, self.key_dim)
        q = F.normalize(q, p=2, dim=-1)
        if rom_state_sparse_embedding:
            state = self.state(addresses.long()).view(-1, self.num_heads, self.key_dim, self.value_dim)
        else:
            state = self.state[addresses.long()]
        debug_check_finite("rom_bigram.state_gather", state)
        if rom_state_recovered_normwrite and self.training:
            rows_for_write = addresses.detach().long()
            q_for_write = q.detach().squeeze(2)
            if self.hit_hist is not None and torch.is_grad_enabled():
                hits_before = self.hit_hist.index_select(0, rows_for_write).float()
                self.hit_hist.index_add_(0, rows_for_write, torch.ones_like(rows_for_write, dtype=self.hit_hist.dtype))
            else:
                hits_before = None

            def recovered_write_hook(grad: Tensor) -> Tensor:
                with torch.no_grad():
                    grad_f = torch.nan_to_num(grad.float(), nan=0.0, posinf=0.0, neginf=0.0)
                    mem_grad = torch.einsum("thk,thkv->thv", q_for_write.float(), grad_f)
                    mem_rms = mem_grad.square().mean(dim=-1, keepdim=True).sqrt().clamp_min_(1e-12)
                    if hits_before is not None:
                        hit_t = (hits_before / rom_state_hit_rms_knee).clamp(0.0, 1.0).view(-1, 1, 1)
                        write_rms = rom_state_hit_rms_low + (rom_state_hit_rms_high - rom_state_hit_rms_low) * hit_t
                    else:
                        write_rms = rom_state_write_rms
                    delta_mem = -mem_grad * (write_rms / mem_rms)
                    update = torch.einsum("thk,thv->thkv", q_for_write.float(), delta_mem)
                    update = update.reshape(update.size(0), -1)
                    unique, inverse = torch.unique(rows_for_write, sorted=False, return_inverse=True)
                    summed = torch.zeros((unique.numel(), update.size(1)), device=update.device, dtype=torch.float32)
                    counts = torch.zeros((unique.numel(), 1), device=update.device, dtype=torch.float32)
                    summed.index_add_(0, inverse, update.float())
                    counts.index_add_(0, inverse, torch.ones((rows_for_write.numel(), 1), device=update.device, dtype=torch.float32))
                    self._recovered_write_rows.append(unique)
                    self._recovered_write_updates.append((summed / counts.clamp_min(1.0)).to(dtype=state.dtype))
                return torch.zeros_like(grad)

            state.register_hook(recovered_write_hook)
        if self.enable_write:
            k = self.k_proj(hidden_states).view(-1, self.num_heads, self.key_dim)
            k = F.normalize(k, p=2, dim=-1)
            v = self.v_proj(hidden_states).view(-1, self.num_heads, self.value_dim)
            beta = torch.sigmoid(self.beta_proj(hidden_states)).unsqueeze(-1)
            decay = torch.exp(-F.softplus(self.decay_proj(hidden_states))).unsqueeze(-1).unsqueeze(-1)
            write_gate = torch.sigmoid(self.write_gate_proj(hidden_states)).unsqueeze(-1).unsqueeze(-1)
            cell = state * decay
            prediction = torch.einsum("thkv,thk->thv", cell, k)
            delta_v = beta * (v - prediction)
            write = k.unsqueeze(-1) * delta_v.unsqueeze(-2)
            state = cell + torch.tanh(self.write_alpha)[0].to(dtype=write.dtype) * write_gate * write
            debug_check_finite("rom_bigram.write_state", state)
        read = torch.einsum("thqk,thkv->thqv", q, state)
        if rom_normalize_readout:
            read_f = read.float()
            read_rms = read_f.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
            read = (read_f * (rom_readout_rms / read_rms)).to(dtype=read.dtype)
        debug_check_finite("rom_bigram.read", read)
        read = read.reshape(hidden_states.size(0), self.num_read_heads * self.value_dim)
        read = apply_rom_read_mlp(read, getattr(self, "read_mlp_norm", None), getattr(self, "read_mlp_fc1", None), getattr(self, "read_mlp_fc2", None))
        if rom_engram_gate:
            return engram_gate_value(read, hidden_states, self.out_proj, self.engram_key_proj, getattr(self, "short_conv_norm", None), getattr(self, "short_conv", None))
        gate = torch.sigmoid(self.gate_proj(hidden_states)).unsqueeze(-1)
        output = self.out_proj((read.view(hidden_states.size(0), self.num_read_heads, self.value_dim) * gate).reshape(hidden_states.size(0), self.num_read_heads * self.value_dim))
        output = apply_rom_short_conv(output, getattr(self, "short_conv_norm", None), getattr(self, "short_conv", None))
        output = apply_rom_ema_smooth(output)
        return rom_output_scale * output

    @torch.no_grad()
    def apply_recovered_normwrite(self, row_cap: float) -> None:
        if not self._recovered_write_rows:
            return
        rows = torch.cat(self._recovered_write_rows, dim=0)
        updates = torch.cat(self._recovered_write_updates, dim=0).float()
        self._recovered_write_rows.clear()
        self._recovered_write_updates.clear()
        if rows.numel() == 0:
            return
        unique, inverse = torch.unique(rows, sorted=False, return_inverse=True)
        summed = torch.zeros((unique.numel(), updates.size(1)), device=updates.device, dtype=torch.float32)
        counts = torch.zeros((unique.numel(), 1), device=updates.device, dtype=torch.float32)
        summed.index_add_(0, inverse, updates)
        counts.index_add_(0, inverse, torch.ones((rows.numel(), 1), device=updates.device, dtype=torch.float32))
        update = summed / counts.clamp_min(1.0)
        param = self.state.weight if isinstance(self.state, nn.Embedding) else self.state
        if isinstance(self.state, nn.Embedding):
            param.index_add_(0, unique, update.to(dtype=param.dtype))
            if row_cap > 0:
                cur = param.index_select(0, unique).float()
                row_rms = row_rms_stable(cur).unsqueeze(1)
                cur.mul_((row_cap / row_rms.clamp_min(1e-12)).clamp_max(1.0))
                param.index_copy_(0, unique, cur.to(dtype=param.dtype))
        else:
            flat = param.view(param.size(0), -1)
            flat.index_add_(0, unique, update.to(dtype=flat.dtype))
            if row_cap > 0:
                cur = flat.index_select(0, unique).float()
                row_rms = row_rms_stable(cur).unsqueeze(1)
                cur.mul_((row_cap / row_rms.clamp_min(1e-12)).clamp_max(1.0))
                flat.index_copy_(0, unique, cur.to(dtype=flat.dtype))


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
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


ENGRAM_ANALYSIS_CHUNKS = []
ENGRAM_ATTNRES_ANALYSIS_CHUNKS = []
ENGRAM_ANALYSIS_PROMPT_INDEX = -1
ENGRAM_ANALYSIS_PROMPT_TEXT = ""
ENGRAM_OUTPUT_GRAD_METRIC_ACCUM: dict[int, dict[str, float]] = {}

def record_engram_output_grad_metrics(layer_id: int | None, output: Tensor) -> None:
    if not (engram_output_grad_metrics and output.requires_grad):
        return
    key = int(layer_id) if layer_id is not None else -1
    stats = ENGRAM_OUTPUT_GRAD_METRIC_ACCUM.setdefault(
        key,
        {"out_sum_sq": 0.0, "out_numel": 0.0, "grad_sum_sq": 0.0, "grad_numel": 0.0, "calls": 0.0},
    )
    out = output.detach().float()
    stats["out_sum_sq"] += float(out.square().sum().item())
    stats["out_numel"] += float(out.numel())
    stats["calls"] += 1.0

    def hook(grad: Tensor) -> Tensor:
        g = grad.detach().float()
        hook_stats = ENGRAM_OUTPUT_GRAD_METRIC_ACCUM.setdefault(
            key,
            {"out_sum_sq": 0.0, "out_numel": 0.0, "grad_sum_sq": 0.0, "grad_numel": 0.0, "calls": 0.0},
        )
        hook_stats["grad_sum_sq"] += float(g.square().sum().item())
        hook_stats["grad_numel"] += float(g.numel())
        return grad

    output.register_hook(hook)

def consume_engram_output_grad_metrics() -> tuple[dict[str, float], str]:
    if not ENGRAM_OUTPUT_GRAD_METRIC_ACCUM:
        return {}, ""
    logs: dict[str, float] = {}
    suffixes = []
    total_grad_sum = 0.0
    total_grad_numel = 0.0
    total_out_sum = 0.0
    total_out_numel = 0.0
    for layer_id in sorted(ENGRAM_OUTPUT_GRAD_METRIC_ACCUM):
        stats = ENGRAM_OUTPUT_GRAD_METRIC_ACCUM[layer_id]
        label = f"l{layer_id}" if layer_id >= 0 else "unknown"
        grad_rms = math.sqrt(stats["grad_sum_sq"] / max(stats["grad_numel"], 1.0))
        out_rms = math.sqrt(stats["out_sum_sq"] / max(stats["out_numel"], 1.0))
        logs[f"engram_output_grad/{label}_grad_rms"] = grad_rms
        logs[f"engram_output_grad/{label}_out_rms"] = out_rms
        logs[f"engram_output_grad/{label}_calls"] = stats["calls"]
        suffixes.append(f" engram_out_grad_rms_{label}:{grad_rms:.3e} engram_out_rms_{label}:{out_rms:.3e}")
        total_grad_sum += stats["grad_sum_sq"]
        total_grad_numel += stats["grad_numel"]
        total_out_sum += stats["out_sum_sq"]
        total_out_numel += stats["out_numel"]
    mean_grad_rms = math.sqrt(total_grad_sum / max(total_grad_numel, 1.0))
    mean_out_rms = math.sqrt(total_out_sum / max(total_out_numel, 1.0))
    logs["engram_output_grad/mean_grad_rms"] = mean_grad_rms
    logs["engram_output_grad/mean_out_rms"] = mean_out_rms
    suffix = f" engram_out_grad_rms_mean:{mean_grad_rms:.3e} engram_out_rms_mean:{mean_out_rms:.3e}" + "".join(suffixes)
    ENGRAM_OUTPUT_GRAD_METRIC_ACCUM.clear()
    return logs, suffix


def record_engram_analysis(input_ids: Tensor, addresses: Tensor, gate: Tensor, value: Tensor, output: Tensor, head_output: Tensor | None = None, *, layer_id: int | None = None, readout_kind: str = "lm"):
    if not engram_analyze:
        return
    with torch.no_grad():
        value_norm = value.detach().float().norm(dim=-1)
        head_output_norm = None
        if head_output is not None:
            head_output_norm = head_output.detach().float().norm(dim=-1).cpu()
        ENGRAM_ANALYSIS_CHUNKS.append({
            "input_ids": input_ids.detach().to(torch.int32).cpu(),
            "addresses": addresses.detach().to(torch.int64).cpu(),
            "gate": torch.sigmoid(gate.detach().float()).cpu(),
            "gate_pre_sigmoid": gate.detach().float().cpu(),
            "value_norm": value_norm.cpu(),
            "output_norm": output.detach().float().norm(dim=-1).cpu(),
            "head_output_norm": head_output_norm,
            "layer_id": -1 if layer_id is None else int(layer_id),
            "readout_kind": str(readout_kind),
            "prompt_index": int(ENGRAM_ANALYSIS_PROMPT_INDEX),
            "prompt_text": ENGRAM_ANALYSIS_PROMPT_TEXT,
        })


def record_engram_attnres_analysis(layer_id: int, memory_weight: Tensor, memory_cos: Tensor, memory_rms_ratio: Tensor):
    if not engram_analyze:
        return
    with torch.no_grad():
        ENGRAM_ATTNRES_ANALYSIS_CHUNKS.append({
            "memory_weight": memory_weight.detach().float().squeeze(-1).cpu(),
            "memory_cos": memory_cos.detach().float().cpu(),
            "memory_rms_ratio": memory_rms_ratio.detach().float().cpu(),
            "layer_id": int(layer_id),
            "prompt_index": int(ENGRAM_ANALYSIS_PROMPT_INDEX),
            "prompt_text": ENGRAM_ANALYSIS_PROMPT_TEXT,
        })


def _tensor_quantiles(x: Tensor, qs: tuple[float, ...]) -> dict[str, float]:
    if x.numel() == 0:
        return {str(q): float("nan") for q in qs}
    x = x.float()
    values = torch.quantile(x, torch.tensor(qs, dtype=torch.float32))
    return {str(q): float(v) for q, v in zip(qs, values)}


def _decode_token_window(token_ids: list[int]) -> str:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        return enc.decode(token_ids)
    except Exception:
        return " ".join(str(t) for t in token_ids)


def _token_category(text: str) -> str:
    if text == "":
        return "empty"
    if "\n" in text:
        return "newline"
    stripped = text.strip()
    if stripped == "":
        return "whitespace"
    if "|" in stripped:
        return "pipe"
    if any(c.isdigit() for c in stripped):
        return "digit"
    if stripped.isalpha():
        return "alpha"
    if all(not c.isalnum() for c in stripped):
        return "punct"
    return "mixed"


def _token_examples(tokens: Tensor, gate: Tensor, output_norm: Tensor, *, metric: Tensor, largest: bool, k: int) -> list[dict]:
    count = min(k, metric.numel())
    values, positions = torch.topk(metric, k=count, largest=largest)
    examples = []
    for value, pos_t in zip(values.tolist(), positions.tolist()):
        window_start = max(0, pos_t - 16)
        window_end = min(tokens.numel(), pos_t + 17)
        token_id = int(tokens[pos_t].item())
        token_text = _decode_token_window([token_id])
        examples.append({
            "position": int(pos_t),
            "token_id": token_id,
            "token_text": token_text,
            "category": _token_category(token_text),
            "metric": float(value),
            "gate": float(gate[pos_t].item()),
            "output_norm": float(output_norm[pos_t].item()),
            "window_text": _decode_token_window([int(x) for x in tokens[window_start:window_end].tolist()]),
        })
    return examples


def write_engram_analysis(module: "EngramBigramMemory", output_path: str, *, run_id: str, checkpoint_path: str, loss: float, step: int):
    if not master_process:
        return
    if not ENGRAM_ANALYSIS_CHUNKS:
        raise RuntimeError("No Engram analysis chunks were recorded")

    tokens = torch.cat([chunk["input_ids"] for chunk in ENGRAM_ANALYSIS_CHUNKS])
    addresses = torch.cat([chunk["addresses"] for chunk in ENGRAM_ANALYSIS_CHUNKS])
    gate = torch.cat([chunk["gate"] for chunk in ENGRAM_ANALYSIS_CHUNKS])
    gate_pre = torch.cat([chunk["gate_pre_sigmoid"] for chunk in ENGRAM_ANALYSIS_CHUNKS])
    value_norm = torch.cat([chunk["value_norm"] for chunk in ENGRAM_ANALYSIS_CHUNKS])
    output_norm = torch.cat([chunk["output_norm"] for chunk in ENGRAM_ANALYSIS_CHUNKS])
    head_output_chunks = [chunk["head_output_norm"] for chunk in ENGRAM_ANALYSIS_CHUNKS if chunk["head_output_norm"] is not None]
    head_output_norm = torch.cat(head_output_chunks) if head_output_chunks else None
    gate_token = gate.mean(dim=-1) if gate.ndim == 2 else gate
    gate_pre_token = gate_pre.mean(dim=-1) if gate_pre.ndim == 2 else gate_pre
    value_norm_token = value_norm.mean(dim=-1) if value_norm.ndim == 2 else value_norm

    category_stats = {}
    token_text_cache: dict[int, str] = {}
    for pos in range(tokens.numel()):
        token_id = int(tokens[pos].item())
        text = token_text_cache.get(token_id)
        if text is None:
            text = _decode_token_window([token_id])
            token_text_cache[token_id] = text
        category = _token_category(text)
        stats = category_stats.setdefault(category, {"count": 0, "gate_sum": 0.0, "output_norm_sum": 0.0})
        stats["count"] += 1
        stats["gate_sum"] += float(gate_token[pos].item())
        stats["output_norm_sum"] += float(output_norm[pos].item())
    for stats in category_stats.values():
        denom = max(1, stats["count"])
        stats["avg_gate"] = stats["gate_sum"] / denom
        stats["avg_output_norm"] = stats["output_norm_sum"] / denom
        del stats["gate_sum"]
        del stats["output_norm_sum"]

    offsets = module.offsets.cpu().to(torch.int64)
    head_mods = module.head_mods.cpu().to(torch.int64)
    head_summaries = []
    top_slots = []

    for head_idx in range(module.total_hash_heads):
        num_rows = int(head_mods[head_idx].item())
        local_rows_all = addresses[:, head_idx] - offsets[head_idx]
        valid_rows = (local_rows_all >= 0) & (local_rows_all < num_rows)
        local_rows = local_rows_all[valid_rows]
        source_positions = torch.nonzero(valid_rows, as_tuple=False).flatten()
        counts = torch.bincount(local_rows, minlength=num_rows)
        touched = counts > 0
        gate_sum = torch.zeros(num_rows, dtype=torch.float32)
        output_sum = torch.zeros(num_rows, dtype=torch.float32)
        gate_for_all = gate[:, head_idx].float() if gate.ndim == 2 else gate.float()
        output_for_all = head_output_norm[:, head_idx].float() if head_output_norm is not None else output_norm.float()
        gate_for_head = gate_for_all[valid_rows]
        output_for_head = output_for_all[valid_rows]
        gate_sum.scatter_add_(0, local_rows, gate_for_head)
        output_sum.scatter_add_(0, local_rows, output_for_head)
        top_count = min(engram_analyze_topk, int(touched.sum().item()))
        top_counts, top_rows = torch.topk(counts, k=max(top_count, 1))
        ngram = 2 + head_idx // module.num_heads
        hash_head = head_idx % module.num_heads
        head_summaries.append({
            "head_idx": head_idx,
            "ngram": ngram,
            "hash_head": hash_head,
            "rows": num_rows,
            "unique_rows": int(touched.sum().item()),
            "unique_fraction": float(touched.float().mean().item()),
            "max_count": int(counts.max().item()),
            "mean_count_per_touched_row": float(counts[touched].float().mean().item()) if touched.any() else 0.0,
        })
        for row, count in zip(top_rows.tolist(), top_counts.tolist()):
            if count <= 0:
                continue
            mask_positions = source_positions[torch.nonzero(local_rows == row, as_tuple=False).flatten()]
            examples = []
            for pos in mask_positions[:3].tolist():
                ngram_start = max(0, pos - ngram + 1)
                window_start = max(0, pos - 16)
                window_end = min(tokens.numel(), pos + 17)
                ngram_ids = [int(x) for x in tokens[ngram_start:pos + 1].tolist()]
                window_ids = [int(x) for x in tokens[window_start:window_end].tolist()]
                examples.append({
                    "position": pos,
                    "ngram_token_ids": ngram_ids,
                    "ngram_text": _decode_token_window(ngram_ids),
                    "window_text": _decode_token_window(window_ids),
                    "gate": float(gate_for_all[pos].item()),
                    "output_norm": float(output_for_all[pos].item()),
                })
            top_slots.append({
                "head_idx": head_idx,
                "ngram": ngram,
                "hash_head": hash_head,
                "row": int(row),
                "absolute_row": int((row + offsets[head_idx]).item()),
                "count": int(count),
                "avg_gate": float((gate_sum[row] / counts[row].clamp_min(1)).item()),
                "avg_output_norm": float((output_sum[row] / counts[row].clamp_min(1)).item()),
                "examples": examples,
            })

    top_slots.sort(key=lambda item: (item["avg_output_norm"], item["count"]), reverse=True)
    top_slots = top_slots[:engram_analyze_topk]
    result = {
        "run_id": run_id,
        "checkpoint_path": checkpoint_path,
        "checkpoint_step": step,
        "analysis_tokens": int(tokens.numel()),
        "loss": loss,
        "engram_dim": module.memory_dim,
        "engram_heads": module.num_heads,
        "engram_max_ngram": module.max_ngram,
        "total_hash_heads": module.total_hash_heads,
        "head_dim": module.head_dim,
        "store_dim": getattr(module, "store_dim", module.head_dim),
        "untied_proj": bool(engram_untied_proj),
        "gate": {
            "mean": float(gate.mean().item()),
            "std": float(gate.std(unbiased=False).item()),
            "quantiles": _tensor_quantiles(gate, (0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99)),
            "frac_gt_0_5": float((gate > 0.5).float().mean().item()),
            "frac_gt_0_75": float((gate > 0.75).float().mean().item()),
            "frac_gt_0_9": float((gate > 0.9).float().mean().item()),
            "token_mean_quantiles": _tensor_quantiles(gate_token, (0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99)),
        },
        "gate_pre_sigmoid": {
            "mean": float(gate_pre_token.mean().item()),
            "std": float(gate_pre_token.std(unbiased=False).item()),
            "quantiles": _tensor_quantiles(gate_pre_token, (0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99)),
        },
        "value_norm": {
            "mean": float(value_norm_token.mean().item()),
            "quantiles": _tensor_quantiles(value_norm_token, (0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99)),
        },
        "output_norm": {
            "mean": float(output_norm.mean().item()),
            "quantiles": _tensor_quantiles(output_norm, (0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99)),
        },
        "head_summaries": head_summaries,
        "top_slots": top_slots,
        "token_category_stats": dict(sorted(category_stats.items())),
        "top_tokens_by_output_norm": _token_examples(tokens, gate_token, output_norm, metric=output_norm, largest=True, k=engram_analyze_topk),
        "top_tokens_by_gate": _token_examples(tokens, gate_token, output_norm, metric=gate_token, largest=True, k=engram_analyze_topk),
        "bottom_tokens_by_gate": _token_examples(tokens, gate_token, output_norm, metric=gate_token, largest=False, k=engram_analyze_topk),
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    csv_path = output_path.replace(".json", "_top_slots.csv")
    with open(csv_path, "w") as f:
        f.write("head_idx,ngram,hash_head,row,absolute_row,count,avg_gate,avg_output_norm,example_ngram_text\n")
        for slot in top_slots:
            example = slot["examples"][0]["ngram_text"].replace("\n", "\\n").replace(",", " ") if slot["examples"] else ""
            f.write(f"{slot['head_idx']},{slot['ngram']},{slot['hash_head']},{slot['row']},{slot['absolute_row']},{slot['count']},{slot['avg_gate']:.8f},{slot['avg_output_norm']:.8f},{example}\n")


def load_engram_prompt_texts(path: str) -> list[str]:
    raw = Path(path).read_text()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        prompts = [part.strip() for part in re.split(r"\n---+\n", raw) if part.strip()]
        return prompts
    if isinstance(payload, list):
        prompts = []
        for item in payload:
            if isinstance(item, str):
                prompts.append(item)
            elif isinstance(item, dict) and "prompt" in item:
                prompts.append(str(item["prompt"]))
            else:
                raise ValueError("Prompt JSON list entries must be strings or objects with a 'prompt' field")
        return prompts
    if isinstance(payload, dict) and isinstance(payload.get("prompts"), list):
        return [str(item.get("prompt", item) if isinstance(item, dict) else item) for item in payload["prompts"]]
    raise ValueError("Prompt file must be a JSON list, a JSON object with 'prompts', or text blocks split by ---")


def build_prompt_batch(prompt: str) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    token_ids = [BOS_ID] + enc.encode(prompt, allowed_special={"<|endoftext|>"}) + [BOS_ID]
    if len(token_ids) < 3:
        raise ValueError("Prompt must encode to at least one non-BOS token")
    pad_count = (-(len(token_ids) - 1)) % 16
    if pad_count:
        token_ids.extend([BOS_ID] * pad_count)
    inputs_cpu = torch.tensor(token_ids[:-1], dtype=torch.int32)
    targets_cpu = torch.tensor(token_ids[1:], dtype=torch.int64)
    cum_cpu = torch.full((128,), inputs_cpu.numel(), dtype=torch.int32)
    cum_cpu[0] = 0
    cum_cpu[1] = inputs_cpu.numel()
    bigram_cpu = get_bigram_hash(inputs_cpu)
    return (
        inputs_cpu.to(device="cuda", non_blocking=True),
        targets_cpu.to(device="cuda", non_blocking=True),
        cum_cpu.to(device="cuda", non_blocking=True),
        bigram_cpu.to(device="cuda", non_blocking=True),
    )


def _token_texts_for_ids(token_ids: list[int]) -> list[str]:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        out = []
        for token_id in token_ids:
            if token_id == BOS_ID:
                out.append("<|endoftext|>")
            else:
                out.append(enc.decode([token_id]))
        return out
    except Exception:
        return [str(token_id) for token_id in token_ids]


def _color_for_score(score: float, lo: float, hi: float, *, hue: str) -> str:
    if not math.isfinite(score) or hi <= lo:
        alpha = 0.0
    else:
        alpha = max(0.0, min(1.0, (score - lo) / (hi - lo)))
    if hue == "blue":
        return f"rgba(37,99,235,{0.08 + 0.78 * alpha:.3f})"
    return f"rgba(220,38,38,{0.08 + 0.78 * alpha:.3f})"


def write_engram_prompt_gating_report(output_path: str, *, json_path: str, run_id: str, checkpoint_path: str, loss_by_prompt: list[float]):
    if not master_process:
        return
    grouped: dict[int, list[dict]] = {}
    prompt_texts: dict[int, str] = {}
    for chunk in ENGRAM_ANALYSIS_CHUNKS:
        prompt_idx = int(chunk.get("prompt_index", -1))
        grouped.setdefault(prompt_idx, []).append(chunk)
        prompt_texts[prompt_idx] = str(chunk.get("prompt_text", ""))
    if not grouped:
        raise RuntimeError("No prompt Engram analysis chunks were recorded")

    records = []
    html_sections = []
    attnres_by_prompt_layer: dict[tuple[int, int], dict] = {}
    for chunk in ENGRAM_ATTNRES_ANALYSIS_CHUNKS:
        prompt_idx = int(chunk.get("prompt_index", -1))
        layer_id = int(chunk.get("layer_id", -1))
        attnres_by_prompt_layer[(prompt_idx, layer_id)] = chunk
    for prompt_idx in sorted(grouped):
        chunks = grouped[prompt_idx]
        token_ids = [int(x) for x in chunks[0]["input_ids"].tolist()]
        token_texts = _token_texts_for_ids(token_ids)
        prompt_record = {
            "prompt_index": prompt_idx,
            "prompt": prompt_texts.get(prompt_idx, ""),
            "loss": loss_by_prompt[prompt_idx] if prompt_idx < len(loss_by_prompt) else None,
            "tokens": [{"id": tid, "text": txt} for tid, txt in zip(token_ids, token_texts)],
            "layers": [],
        }
        section_parts = [
            f"<section><h2>Prompt {prompt_idx + 1}</h2>",
            f"<p class='prompt'>{html_lib.escape(prompt_texts.get(prompt_idx, ''))}</p>",
        ]
        for chunk in chunks:
            gate = chunk["gate"].float()
            gate_token = gate.mean(dim=-1) if gate.ndim == 2 else gate
            output_norm = chunk["output_norm"].float()
            head_output = chunk.get("head_output_norm")
            head_values = head_output.float() if head_output is not None else None
            gate_values = [float(x) for x in gate_token.tolist()]
            output_values = [float(x) for x in output_norm.tolist()]
            layer_id = int(chunk.get("layer_id", -1))
            readout_kind = str(chunk.get("readout_kind", "lm"))
            attnres = attnres_by_prompt_layer.get((prompt_idx, layer_id))
            attnres_weight = attnres["memory_weight"].float() if attnres is not None else None
            attnres_cos = attnres["memory_cos"].float() if attnres is not None else None
            attnres_rms_ratio = attnres["memory_rms_ratio"].float() if attnres is not None else None
            if attnres_weight is not None and attnres_weight.ndim > 1:
                attnres_weight = attnres_weight.squeeze(0)
            if attnres_cos is not None and attnres_cos.ndim > 1:
                attnres_cos = attnres_cos.squeeze(0)
            if attnres_rms_ratio is not None and attnres_rms_ratio.ndim > 1:
                attnres_rms_ratio = attnres_rms_ratio.squeeze(0)
            if attnres_weight is not None and attnres_weight.numel() != len(token_ids):
                attnres_weight = None
            if attnres_cos is not None and attnres_cos.numel() != len(token_ids):
                attnres_cos = None
            if attnres_rms_ratio is not None and attnres_rms_ratio.numel() != len(token_ids):
                attnres_rms_ratio = None
            gq = _tensor_quantiles(gate_token, (0.05, 0.5, 0.95))
            aq = _tensor_quantiles(attnres_weight, (0.05, 0.5, 0.95)) if attnres_weight is not None else None
            oq = _tensor_quantiles(output_norm, (0.05, 0.5, 0.95))
            layer_record = {
                "layer_id": layer_id,
                "readout_kind": readout_kind,
                "gate_mean": float(gate_token.mean().item()),
                "gate_max": float(gate_token.max().item()),
                "attnres_weight_mean": float(attnres_weight.mean().item()) if attnres_weight is not None else None,
                "attnres_weight_max": float(attnres_weight.max().item()) if attnres_weight is not None else None,
                "memory_cos_mean": float(attnres_cos.mean().item()) if attnres_cos is not None else None,
                "memory_rms_ratio_mean": float(attnres_rms_ratio.mean().item()) if attnres_rms_ratio is not None else None,
                "output_norm_mean": float(output_norm.mean().item()),
                "output_norm_max": float(output_norm.max().item()),
                "tokens": [],
            }
            spans = []
            for pos, (tok_id, tok_text, gate_v, out_v) in enumerate(zip(token_ids, token_texts, gate_values, output_values)):
                attn_v = float(attnres_weight[pos].item()) if attnres_weight is not None else None
                cos_v = float(attnres_cos[pos].item()) if attnres_cos is not None else None
                rms_ratio_v = float(attnres_rms_ratio[pos].item()) if attnres_rms_ratio is not None else None
                top_head = None
                top_head_norm = None
                if head_values is not None and head_values.ndim == 2:
                    top_head_norm_t, top_head_t = torch.max(head_values[pos], dim=0)
                    top_head = int(top_head_t.item())
                    top_head_norm = float(top_head_norm_t.item())
                layer_record["tokens"].append({
                    "pos": pos,
                    "token_id": tok_id,
                    "token_text": tok_text,
                    "gate": gate_v,
                    "attnres_weight": attn_v,
                    "memory_cos": cos_v,
                    "memory_rms_ratio": rms_ratio_v,
                    "output_norm": out_v,
                    "top_head": top_head,
                    "top_head_output_norm": top_head_norm,
                })
                bg = _color_for_score(gate_v, gq["0.05"], gq["0.95"], hue="red")
                border = ""
                if attn_v is not None and aq is not None:
                    border = f"border-bottom:3px solid {_color_for_score(attn_v, aq['0.05'], aq['0.95'], hue='blue')};"
                title = f"pos {pos} id {tok_id} gate {gate_v:.3f} output_norm {out_v:.3f}"
                if attn_v is not None:
                    title += f" attnres_weight {attn_v:.3f}"
                if cos_v is not None:
                    title += f" mem_cos {cos_v:.3f}"
                if rms_ratio_v is not None:
                    title += f" mem_rms_ratio {rms_ratio_v:.3f}"
                if top_head is not None:
                    title += f" top_head {top_head} head_norm {top_head_norm:.3f}"
                label = html_lib.escape(tok_text).replace(" ", "&nbsp;").replace("\n", "↵")
                spans.append(f"<span class='tok' style='background:{bg};{border}' title='{html_lib.escape(title)}'>{label}</span>")
            prompt_record["layers"].append(layer_record)
            attn_summary = ""
            if layer_record["attnres_weight_mean"] is not None:
                attn_summary = (
                    f"; attnres mean {layer_record['attnres_weight_mean']:.3f}, "
                    f"cos {layer_record['memory_cos_mean']:.3f}, rms ratio {layer_record['memory_rms_ratio_mean']:.3f}"
                )
            section_parts.append(
                f"<h3>Layer {layer_id} {html_lib.escape(readout_kind)} "
                f"<small>gate mean {layer_record['gate_mean']:.3f}, max {layer_record['gate_max']:.3f}; "
                f"out mean {layer_record['output_norm_mean']:.3f}{attn_summary}</small></h3>"
            )
            section_parts.append("<div class='tokens'>" + "".join(spans) + "</div>")
        section_parts.append("</section>")
        records.append(prompt_record)
        html_sections.append("\n".join(section_parts))

    result = {
        "run_id": run_id,
        "checkpoint_path": checkpoint_path,
        "prompts": records,
    }
    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    css = """
body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:28px;background:#f8fafc;color:#111827}
h1{font-size:26px;margin:0 0 8px} h2{margin-top:28px} h3{font-size:15px;margin:18px 0 8px}
small{font-weight:400;color:#4b5563}.meta,.prompt{color:#374151;white-space:pre-wrap}
section{background:white;border:1px solid #e5e7eb;border-radius:8px;padding:18px;margin:18px 0}
.tokens{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;line-height:2.05;word-break:break-word}
.tok{display:inline-block;border-radius:4px;margin:1px;padding:1px 3px;white-space:pre-wrap}
.legend{height:12px;width:260px;border-radius:999px;background:linear-gradient(90deg,rgba(220,38,38,.08),rgba(220,38,38,.86));display:inline-block;vertical-align:middle}
"""
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Engram Prompt Gating</title><style>{css}</style></head>
<body>
<h1>Engram Prompt Gating</h1>
<p class="meta">run: <code>{html_lib.escape(run_id)}</code><br>checkpoint: <code>{html_lib.escape(checkpoint_path)}</code><br>color: low gate <span class="legend"></span> high gate</p>
{''.join(html_sections)}
</body></html>
"""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    Path(output_path).write_text(html)


def _canonical_token_key(token_id: int, enc) -> str:
    token_bytes = enc.decode_single_token_bytes(token_id)
    try:
        text = token_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return f"bytes:{token_bytes!r}"
    if "\ufffd" in text:
        return f"bytes:{token_bytes!r}"
    sentinel = "\ue000"
    text = unicodedata.normalize("NFKC", text)
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[ \t\r\n]+", " ", text)
    if text == " ":
        text = sentinel
    text = text.strip()
    if text == sentinel:
        text = " "
    return text if text else token_bytes.decode("utf-8", errors="replace")


def build_canonical_token_lookup(vocab_size: int) -> tuple[Tensor, int]:
    try:
        import tiktoken
    except Exception as exc:
        raise RuntimeError("ENGRAM_CANONICALIZE requires tiktoken") from exc
    enc = tiktoken.get_encoding("gpt2")
    key_to_new: dict[str, int] = {}
    old_to_new = torch.empty(vocab_size, dtype=torch.int64)
    for token_id in range(vocab_size):
        key = _canonical_token_key(token_id, enc)
        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        old_to_new[token_id] = new_id
    return old_to_new, len(key_to_new)


def maybe_disable_engram_compile(fn):
    if not engram_disable_compile_region:
        return fn
    compiler = getattr(torch, "compiler", None)
    if compiler is not None and hasattr(compiler, "disable"):
        return compiler.disable(fn)
    return torch._dynamo.disable(fn)


class PerHeadLinear(nn.Module):
    """Independent bias-free linear projection per memory read head."""

    def __init__(self, num_heads: int, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_heads, out_features, in_features))
        for head_weight in self.weight:
            nn.init.kaiming_uniform_(head_weight, a=math.sqrt(5))

    def forward(self, x: Tensor) -> Tensor:
        return torch.einsum("thd,hod->tho", x, self.weight.type_as(x))


class EngramLayerReadout(nn.Module):
    def __init__(self, total_hash_heads: int, head_dim: int, memory_dim: int, model_dim: int, max_ngram: int):
        super().__init__()
        self.total_hash_heads = total_hash_heads
        self.per_head = engram_per_head
        if self.per_head:
            if engram_untied_proj:
                self.value_proj = PerHeadLinear(total_hash_heads, head_dim, model_dim)
                self.key_proj = PerHeadLinear(total_hash_heads, head_dim, model_dim)
            else:
                self.value_proj = nn.Linear(head_dim, model_dim, bias=False)
                self.key_proj = nn.Linear(head_dim, model_dim, bias=False)
        else:
            self.value_proj = nn.Linear(memory_dim, model_dim, bias=False)
            self.key_proj = nn.Linear(memory_dim, model_dim, bias=False)
        if engram_short_conv:
            self.short_conv_norm = nn.RMSNorm(model_dim)
            self.short_conv = nn.Conv1d(
                model_dim,
                model_dim,
                engram_short_conv_kernel,
                groups=model_dim,
                bias=False,
                padding=(engram_short_conv_kernel - 1) * max_ngram,
                dilation=max_ngram,
            )
            nn.init.zeros_(self.short_conv.weight)
        else:
            self.short_conv_norm = None
            self.short_conv = None


class EngramLayerReadoutDelta(nn.Module):
    """Zero-initialized per-layer correction on top of the shared Engram readout."""

    def __init__(self, total_hash_heads: int, head_dim: int, memory_dim: int, model_dim: int):
        super().__init__()
        self.total_hash_heads = total_hash_heads
        self.per_head = engram_per_head
        if self.per_head:
            if engram_untied_proj:
                self.value_proj = PerHeadLinear(total_hash_heads, head_dim, model_dim)
                self.key_proj = PerHeadLinear(total_hash_heads, head_dim, model_dim)
            else:
                self.value_proj = nn.Linear(head_dim, model_dim, bias=False)
                self.key_proj = nn.Linear(head_dim, model_dim, bias=False)
        else:
            self.value_proj = nn.Linear(memory_dim, model_dim, bias=False)
            self.key_proj = nn.Linear(memory_dim, model_dim, bias=False)
        for param in self.parameters():
            nn.init.zeros_(param)


class EngramBigramMemory(nn.Module):
    """Engram-style static multi-head N-gram memory with context-aware gating."""

    def __init__(self, vocab_size: int, model_dim: int, memory_dim: int, num_heads: int, max_ngram: int, *, seed: int = 0, pad_id: int = 50256, token_vocab_size: int = 50257, layer_hash_ids: tuple[int, ...] = (), layer_readout_ids: tuple[int, ...] = (), layer_readout_delta_ids: tuple[int, ...] = (), layer_partition_ids: tuple[int, ...] = (), layer_partition_group_ids: tuple[int, ...] = ()):
        super().__init__()
        if min(vocab_size, model_dim, memory_dim, num_heads, max_ngram) <= 0:
            raise ValueError("Engram dimensions must be positive")
        total_hash_heads = (max_ngram - 1) * num_heads
        if memory_dim % total_hash_heads != 0:
            raise ValueError("ENGRAM_DIM must be divisible by (ENGRAM_MAX_NGRAM - 1) * ENGRAM_HEADS")
        self.model_dim = model_dim
        self.memory_dim = memory_dim
        self.num_heads = num_heads
        self.max_ngram = max_ngram
        self.total_hash_heads = total_hash_heads
        self.head_dim = memory_dim // total_hash_heads
        self.store_dim = engram_store_dim if engram_store_dim > 0 else self.head_dim
        if self.store_dim <= 0:
            raise ValueError("ENGRAM_STORE_DIM must be positive when set")
        if self.store_dim != self.head_dim and not engram_per_head:
            raise ValueError("ENGRAM_STORE_DIM currently requires ENGRAM_PER_HEAD=1")
        if engram_untied_proj and not engram_per_head:
            raise ValueError("ENGRAM_UNTIED_PROJ requires ENGRAM_PER_HEAD=1")
        self.pad_id = pad_id
        self.per_head = engram_per_head
        self.latent = engram_latent
        if self.latent:
            self.latent_quantizer = engram_latent_quantizer
            self.latent_mix_ngram = engram_latent_mix_ngram
            rows_per_head_for_bsq = max(2, engram_latent_rows_per_head if engram_latent_rows_per_head > 0 else vocab_size)
            if self.latent_quantizer == "pkm":
                self.latent_fsq_levels = ()
                self.latent_pkm_subkeys = engram_latent_pkm_subkeys
                self.latent_pkm_topk = engram_latent_pkm_topk
                self.latent_pkm_key_dim = engram_latent_pkm_key_dim
                self.latent_dim = 2 * self.latent_pkm_key_dim
                self.latent_codebook_size = self.latent_pkm_subkeys * self.latent_pkm_subkeys
                self.latent_pkm_keys = nn.Parameter(torch.empty(self.total_hash_heads, 2, self.latent_pkm_subkeys, self.latent_pkm_key_dim))
                nn.init.normal_(self.latent_pkm_keys, std=self.latent_pkm_key_dim ** -0.5)
                self.register_buffer("latent_levels", torch.empty((0,), dtype=torch.float32), persistent=False)
                latent_basis = []
            elif self.latent_quantizer == "bsq":
                self.latent_fsq_levels = ()
                self.latent_dim = engram_latent_bsq_bits if engram_latent_bsq_bits > 0 else math.ceil(math.log2(rows_per_head_for_bsq))
                if self.latent_dim >= 63:
                    raise ValueError("ENGRAM_LATENT_BSQ_BITS must be < 63 for int64 address packing")
            else:
                self.latent_fsq_levels = tuple(engram_latent_fsq_levels)
                self.latent_dim = len(self.latent_fsq_levels)
            self.latent_proj = nn.Linear(model_dim, self.total_hash_heads * self.latent_dim, bias=False)
            self.latent_input_scale = engram_latent_input_scale
            self.latent_ste_scale = engram_latent_ste_scale
            self.latent_ste_proj = nn.Linear(self.latent_dim, self.store_dim, bias=False) if self.latent_ste_scale > 0 and self.latent_quantizer != "pkm" else None
            if self.latent_ste_proj is not None:
                nn.init.zeros_(self.latent_ste_proj.weight)
            if self.latent_quantizer == "pkm":
                pass
            elif self.latent_quantizer == "bsq":
                latent_basis = [1 << i for i in range(self.latent_dim)]
                self.latent_codebook_size = 1 << self.latent_dim
                self.register_buffer("latent_levels", torch.empty((0,), dtype=torch.float32), persistent=False)
            else:
                latent_basis = []
                basis = 1
                for level in self.latent_fsq_levels:
                    latent_basis.append(basis)
                    basis *= level
                self.latent_codebook_size = basis
                self.register_buffer("latent_levels", torch.tensor(self.latent_fsq_levels, dtype=torch.float32), persistent=False)
                self.register_buffer("latent_half_width", torch.tensor([level // 2 for level in self.latent_fsq_levels], dtype=torch.float32), persistent=False)
            self.register_buffer("latent_basis", torch.tensor(latent_basis, dtype=torch.int64), persistent=False)
        if engram_canonicalize:
            canonical_lookup, canonical_vocab_size = build_canonical_token_lookup(token_vocab_size)
            self.register_buffer("canonical_lookup", canonical_lookup, persistent=False)
            self.hash_token_vocab_size = canonical_vocab_size
            self.hash_pad_id = int(canonical_lookup[min(max(pad_id, 0), token_vocab_size - 1)].item())
        else:
            self.canonical_lookup = None
            self.hash_token_vocab_size = token_vocab_size
            self.hash_pad_id = pad_id

        self.layer_partition_ids = tuple(int(i) for i in layer_partition_ids)
        if layer_partition_group_ids:
            if len(layer_partition_group_ids) != len(self.layer_partition_ids):
                raise ValueError("layer_partition_group_ids must match layer_partition_ids length")
            raw_partition_group_ids = tuple(int(i) for i in layer_partition_group_ids)
            if min(raw_partition_group_ids, default=0) < 0:
                raise ValueError("layer_partition_group_ids must be non-negative")
            unique_group_ids = tuple(sorted(set(raw_partition_group_ids)))
            group_index = {group_id: idx for idx, group_id in enumerate(unique_group_ids)}
            partition_group_ids = tuple(group_index[group_id] for group_id in raw_partition_group_ids)
        else:
            partition_group_ids = tuple(range(len(self.layer_partition_ids)))
        self.layer_partition_group_ids = partition_group_ids
        self.layer_partition_index = {layer_id: partition_group_ids[idx] for idx, layer_id in enumerate(self.layer_partition_ids)}
        seen_primes: set[int] = set()
        row_factors = engram_ngram_row_factors or (1.0,) * (max_ngram - 1)
        row_factor_norm = (max_ngram - 1) / sum(row_factors)
        if self.layer_partition_ids:
            if self.latent:
                raise ValueError("ENGRAM_LATENT currently does not support ENGRAM_LAYER_PARTITIONS")
            partition_count = max(partition_group_ids) + 1
            per_layer_vocab_size = max(2, vocab_size // partition_count)
            layer_head_mods = []
            flat_sizes = []
            for _partition_id in range(partition_count):
                head_mods = []
                for ngram_idx, _ngram in enumerate(range(2, max_ngram + 1)):
                    for _head in range(num_heads):
                        rows_for_head = max(2, int(round(per_layer_vocab_size * row_factors[ngram_idx] * row_factor_norm)))
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
                layer_offsets.append(flat_offsets[cursor: cursor + self.total_hash_heads])
                cursor += self.total_hash_heads
            self.register_buffer("head_mods", torch.tensor(layer_head_mods[0], dtype=torch.int64), persistent=False)
            self.register_buffer("offsets", torch.tensor(layer_offsets[0], dtype=torch.int64), persistent=False)
            self.register_buffer("layer_head_mods", torch.tensor(layer_head_mods, dtype=torch.int64), persistent=False)
            self.register_buffer("layer_offsets", torch.tensor(layer_offsets, dtype=torch.int64), persistent=False)
            self.num_memory_rows = sum(flat_sizes)
        else:
            head_mods = []
            for ngram_idx, _ngram in enumerate(range(2, max_ngram + 1)):
                for _head in range(num_heads):
                    rows_for_head = max(2, int(round(vocab_size * row_factors[ngram_idx] * row_factor_norm)))
                    head_mods.append(_next_prime_after(rows_for_head - 1, seen_primes))
            offsets = [0]
            for size in head_mods[:-1]:
                offsets.append(offsets[-1] + size)
            self.register_buffer("head_mods", torch.tensor(head_mods, dtype=torch.int64), persistent=False)
            self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.int64), persistent=False)
            self.layer_head_mods = None
            self.layer_offsets = None
            self.num_memory_rows = sum(head_mods)
        if self.latent:
            if self.latent_quantizer == "pkm":
                rows_per_head = self.latent_codebook_size
            else:
                rows_per_head = engram_latent_rows_per_head if engram_latent_rows_per_head > 0 else vocab_size
            rows_per_head = max(2, rows_per_head)
            head_mods = [rows_per_head] * self.total_hash_heads
            offsets = [0]
            for size in head_mods[:-1]:
                offsets.append(offsets[-1] + size)
            self.register_buffer("head_mods", torch.tensor(head_mods, dtype=torch.int64), persistent=False)
            self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.int64), persistent=False)
            self.layer_head_mods = None
            self.layer_offsets = None
            self.num_memory_rows = sum(head_mods)
        sketch_salts = torch.arange(1, engram_sketch_k + 1, dtype=torch.int64) * 0x9E3779B1
        sketch_sign_salts = torch.arange(1, engram_sketch_k + 1, dtype=torch.int64) * 0x85EBCA77
        superpose_salts = torch.arange(1, engram_superpose_k + 1, dtype=torch.int64) * 0xD1B54A35
        self.register_buffer("sketch_salts", sketch_salts, persistent=False)
        self.register_buffer("sketch_sign_salts", sketch_sign_salts, persistent=False)
        self.register_buffer("superpose_salts", superpose_salts, persistent=False)
        if engram_sketch_dim_signs and engram_sketch_k > 1:
            if engram_sketch_dim_sign_mode == "hadamard":
                slot_ids = torch.arange(0, engram_sketch_k, dtype=torch.int64).view(-1, 1)
                dim_ids = torch.arange(0, self.store_dim, dtype=torch.int64).view(1, -1)
                parity = torch.zeros((engram_sketch_k, self.store_dim), dtype=torch.int64)
                for bit in range(max(1, (engram_sketch_k - 1).bit_length())):
                    parity = torch.bitwise_xor(
                        parity,
                        torch.bitwise_and((slot_ids >> bit) & 1, (dim_ids >> bit) & 1),
                    )
                sketch_dim_signs = torch.where(parity == 0, 1.0, -1.0).unsqueeze(0)
                sketch_dim_signs = sketch_dim_signs.expand(self.total_hash_heads, -1, -1).clone()
            elif engram_sketch_dim_sign_mode == "balanced":
                head_ids = torch.arange(1, self.total_hash_heads + 1, dtype=torch.int64).view(-1, 1, 1)
                slot_ids = torch.arange(1, engram_sketch_k + 1, dtype=torch.int64).view(1, -1, 1)
                dim_ids = torch.arange(1, self.store_dim + 1, dtype=torch.int64).view(1, 1, -1)
                sign_mix = self._avalanche_hash(
                    (head_ids * 0x9E3779B1)
                    ^ (slot_ids * 0x85EBCA77)
                    ^ (dim_ids * 0xC2B2AE3D)
                    ^ int(engram_hash_seed)
                )
                order = torch.argsort(sign_mix, dim=1)
                ranks = torch.empty_like(order)
                slot_ranks = torch.arange(engram_sketch_k, dtype=torch.int64).view(1, -1, 1).expand_as(order)
                ranks.scatter_(1, order, slot_ranks)
                positives = (engram_sketch_k + 1) // 2
                sketch_dim_signs = torch.where(ranks < positives, 1.0, -1.0)
            else:
                head_ids = torch.arange(1, self.total_hash_heads + 1, dtype=torch.int64).view(-1, 1, 1)
                slot_ids = torch.arange(1, engram_sketch_k + 1, dtype=torch.int64).view(1, -1, 1)
                dim_ids = torch.arange(1, self.store_dim + 1, dtype=torch.int64).view(1, 1, -1)
                sign_mix = self._avalanche_hash(
                    (head_ids * 0x9E3779B1)
                    ^ (slot_ids * 0x85EBCA77)
                    ^ (dim_ids * 0xC2B2AE3D)
                    ^ int(engram_hash_seed)
                )
                sketch_dim_signs = torch.where(torch.bitwise_and(sign_mix, 1) == 0, 1.0, -1.0)
            if engram_sketch_include_base:
                sketch_dim_signs[:, 0, :] = 1.0
        else:
            sketch_dim_signs = torch.empty(0)
        self.register_buffer("sketch_dim_signs", sketch_dim_signs, persistent=False)

        generator = torch.Generator()
        generator.manual_seed(seed)
        max_multiplier = max(1, (2**31 - 1) // max(1, self.hash_token_vocab_size))
        multipliers = torch.randint(1, max_multiplier, (max_ngram,), generator=generator, dtype=torch.int64) * 2 + 1
        self.register_buffer("multipliers", multipliers, persistent=False)
        self.layer_hash_ids = tuple(int(i) for i in layer_hash_ids)
        self.layer_hash_index = {layer_id: idx for idx, layer_id in enumerate(self.layer_hash_ids)}
        if self.layer_hash_ids:
            layer_multipliers = []
            for layer_id in self.layer_hash_ids:
                layer_generator = torch.Generator()
                layer_generator.manual_seed(engram_hash_seed + 10007 * layer_id)
                layer_multipliers.append(
                    torch.randint(1, max_multiplier, (max_ngram,), generator=layer_generator, dtype=torch.int64) * 2 + 1
                )
            self.register_buffer("layer_multipliers", torch.stack(layer_multipliers), persistent=False)
        else:
            self.layer_multipliers = None

        self.offload = engram_offload
        self._offload_pending: list[tuple[Tensor, Tensor, Tensor | None, Tensor | None]] = []
        self._offload_step = 0
        self._offload_lazy_moments = self.offload and engram_offload_lazy_moments
        self._offload_moment_rows: Tensor | None = None
        self._offload_moment_index: Tensor | None = None
        self._offload_prefetch: tuple[int, int, Tensor, Tensor, Future] | None = None
        self._offload_prefetch_moments = False
        self._offload_adam_future: Future | None = None
        self._offload_executor: ThreadPoolExecutor | None = None
        self._offload_stream: torch.cuda.Stream | None = None
        self._offload_stream_device: torch.device | None = None
        self.last_update_metrics: dict[str, float | int] = {}
        if self.offload:
            self.embedding = None
            self._offload_executor = ThreadPoolExecutor(max_workers=1)
            self._init_offloaded_embedding()
        else:
            self.embedding = nn.Embedding(self.num_memory_rows, self.store_dim, sparse=(engram_sparse_adam or engram_sparse_vector_adam or engram_sparse_scalar_adam or engram_sparse_row_adagrad))
            if engram_sparse_grad_coalesce_hook and self.embedding.weight.requires_grad:
                self.embedding.weight.register_hook(
                    lambda grad: (
                        coalesce_row_sparse_grad(grad)
                        if grad.is_sparse and rom_debug_nan_current_step >= engram_sparse_grad_coalesce_hook_start
                        else grad
                    )
                )
            if engram_init_zero:
                nn.init.zeros_(self.embedding.weight)
            else:
                self.embedding.weight.data.normal_(std=engram_init_std)
            if engram_freeze_memory:
                self.embedding.weight.requires_grad_(False)
        if engram_shadow_grad:
            if self.offload:
                raise ValueError("ENGRAM_SHADOW_GRAD currently requires GPU-resident Engram")
            self.shadow_embedding = nn.Embedding(self.num_memory_rows, model_dim)
            nn.init.zeros_(self.shadow_embedding.weight)
            self.shadow_embedding.weight.requires_grad_(False)
            self.last_shadow_metrics: dict[str, float | int] = {}
        else:
            self.shadow_embedding = None
            self.last_shadow_metrics = {}
        self.memory_proj = nn.Linear(self.store_dim, self.head_dim, bias=False) if self.store_dim != self.head_dim else None
        self.layer_readout_ids = tuple(int(i) for i in layer_readout_ids)
        if self.layer_readout_ids:
            self.layer_readouts = nn.ModuleDict({
                str(layer_id): EngramLayerReadout(self.total_hash_heads, self.head_dim, memory_dim, model_dim, max_ngram)
                for layer_id in self.layer_readout_ids
            })
        else:
            self.layer_readouts = nn.ModuleDict()
            self.value_proj = nn.Linear(self.head_dim if self.per_head else memory_dim, model_dim, bias=False)
            self.key_proj = nn.Linear(self.head_dim if self.per_head else memory_dim, model_dim, bias=False)
            if self.per_head and engram_untied_proj:
                self.value_proj = PerHeadLinear(self.total_hash_heads, self.head_dim, model_dim)
                self.key_proj = PerHeadLinear(self.total_hash_heads, self.head_dim, model_dim)
            if engram_short_conv:
                self.short_conv_norm = nn.RMSNorm(model_dim)
                self.short_conv = nn.Conv1d(
                    model_dim,
                    model_dim,
                    engram_short_conv_kernel,
                    groups=model_dim,
                    bias=False,
                    padding=(engram_short_conv_kernel - 1) * max_ngram,
                    dilation=max_ngram,
                )
                nn.init.zeros_(self.short_conv.weight)
            else:
                self.short_conv_norm = None
                self.short_conv = None
        self.layer_readout_delta_ids = tuple(int(i) for i in layer_readout_delta_ids)
        self.layer_readout_deltas = nn.ModuleDict({
            str(layer_id): EngramLayerReadoutDelta(self.total_hash_heads, self.head_dim, memory_dim, model_dim)
            for layer_id in self.layer_readout_delta_ids
        })
        self.layer_readout_delta_index = {layer_id: idx for idx, layer_id in enumerate(self.layer_readout_delta_ids)}
        if engram_layer_readout_delta_learned_scale and self.layer_readout_delta_ids:
            p = engram_layer_readout_delta_learned_scale_init / engram_layer_readout_delta_learned_scale_max
            p = min(max(p, 1e-6), 1.0 - 1e-6)
            init_logit = math.log(p / (1.0 - p))
            self.layer_readout_delta_scale_logits = nn.Parameter(torch.full((len(self.layer_readout_delta_ids), 2), init_logit))
        else:
            self.layer_readout_delta_scale_logits = None
        if engram_cache_readout:
            self.cache_readouts = nn.ModuleDict({
                str(engram_cache_recon_source_layer): EngramLayerReadout(self.total_hash_heads, self.head_dim, memory_dim, model_dim, max_ngram)
            })
        else:
            self.cache_readouts = nn.ModuleDict()
        if engram_hit_hist:
            self.register_buffer("hit_hist", torch.zeros(self.num_memory_rows, dtype=torch.int32), persistent=False)
        else:
            self.hit_hist = None
        self._eval_hit_hist: Tensor | None = None
        self.last_read_hit_scale: Tensor | None = None
        self.last_hot_split_metrics: dict[str, float | int] = {}
        if engram_layer_head_mix:
            layer_head_mix_ids = self.layer_hash_ids or self.layer_partition_ids
            if not layer_head_mix_ids:
                raise ValueError("ENGRAM_LAYER_HEAD_MIX requires layer-aware Engram ids")
            if engram_head_mix_init and len(engram_head_mix_init) != self.total_hash_heads:
                raise ValueError("ENGRAM_HEAD_MIX_INIT must have one value per hash head")
            self.layer_head_mix_ids = tuple(int(i) for i in layer_head_mix_ids)
            self.layer_head_mix_index = {layer_id: idx for idx, layer_id in enumerate(self.layer_head_mix_ids)}
            self.layer_head_mix_logits = nn.Parameter(torch.zeros(len(self.layer_head_mix_ids), self.total_hash_heads))
            if engram_head_mix_init:
                init = torch.tensor(engram_head_mix_init, dtype=self.layer_head_mix_logits.dtype)
                self.layer_head_mix_logits.data.copy_(init.view(1, -1).expand_as(self.layer_head_mix_logits))
            if engram_head_mix_freeze:
                self.layer_head_mix_logits.requires_grad_(False)
            self.head_mix_logits = None
            self.layer_head_mix_delta_logits = None
        elif engram_head_mix:
            if engram_layer_head_mix_delta:
                layer_head_mix_ids = self.layer_hash_ids or self.layer_partition_ids
                if not layer_head_mix_ids:
                    raise ValueError("ENGRAM_LAYER_HEAD_MIX_DELTA requires layer-aware Engram ids")
                self.layer_head_mix_ids = tuple(int(i) for i in layer_head_mix_ids)
                self.layer_head_mix_index = {layer_id: idx for idx, layer_id in enumerate(self.layer_head_mix_ids)}
                self.layer_head_mix_delta_logits = nn.Parameter(torch.zeros(len(self.layer_head_mix_ids), self.total_hash_heads))
            else:
                self.layer_head_mix_ids = ()
                self.layer_head_mix_index = {}
                self.layer_head_mix_delta_logits = None
            if engram_head_mix_init and len(engram_head_mix_init) != self.total_hash_heads:
                raise ValueError("ENGRAM_HEAD_MIX_INIT must have one value per hash head")
            self.head_mix_logits = nn.Parameter(torch.zeros(self.total_hash_heads))
            if engram_head_mix_init:
                init = torch.tensor(engram_head_mix_init, dtype=self.head_mix_logits.dtype)
                self.head_mix_logits.data.copy_(init)
            if engram_head_mix_freeze:
                self.head_mix_logits.requires_grad_(False)
            self.layer_head_mix_logits = None
        else:
            self.layer_head_mix_ids = ()
            self.layer_head_mix_index = {}
            self.head_mix_logits = None
            self.layer_head_mix_logits = None
            self.layer_head_mix_delta_logits = None
        if engram_sketch_slot_mix or engram_sketch_combine_mix:
            self.sketch_slot_mix_logits = nn.Parameter(torch.zeros(self.total_hash_heads, engram_sketch_k))
        else:
            self.sketch_slot_mix_logits = None
        if engram_sketch_aux_learned_scale and engram_sketch_include_base and engram_sketch_k > 1:
            p = engram_sketch_aux_learned_scale_init / engram_sketch_aux_learned_scale_max
            p = min(max(p, 1e-6), 1.0 - 1e-6)
            init_logit = math.log(p / (1.0 - p))
            self.sketch_aux_scale_logit = nn.Parameter(torch.tensor(init_logit))
        else:
            self.sketch_aux_scale_logit = None
        if engram_static_gate:
            num_gate_layers = len(self.layer_hash_ids) if self.layer_hash_ids else 1
            num_gate_heads = self.total_hash_heads if self.per_head else 1
            self.static_gate_logits = nn.Parameter(torch.full((num_gate_layers, num_gate_heads), engram_static_gate_init))
        else:
            self.static_gate_logits = None

    def _register_debug_backward(self, label: str, tensor: Tensor, addresses: Tensor | None = None) -> Tensor:
        if engram_debug_backward and self.training and tensor.requires_grad and not is_torch_compiling():
            address_snapshot = addresses.detach() if addresses is not None else None
            tensor.register_hook(lambda grad: debug_report_engram_backward(label, grad, address_snapshot))
        return tensor

    def _shadow_grad_write_hook(self, addresses: Tensor, grad: Tensor) -> None:
        if self.shadow_embedding is None or grad.ndim != 2:
            return
        with torch.no_grad():
            grad_f = grad.detach().float()
            grad_rms = grad_f.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-12)
            target = -grad_f * (engram_shadow_write_rms / grad_rms)
            per_row = target.unsqueeze(1).expand(-1, addresses.size(1), -1).reshape(-1, target.size(-1))
            per_row = per_row / (max(engram_shadow_scale, 1e-6) * math.sqrt(addresses.size(1)))
            row_idx = addresses.detach().reshape(-1).to(dtype=torch.long)
            unique, inverse = torch.unique(row_idx, sorted=False, return_inverse=True)
            summed = torch.zeros((unique.numel(), per_row.size(-1)), device=per_row.device, dtype=torch.float32)
            counts = torch.zeros((unique.numel(), 1), device=per_row.device, dtype=torch.float32)
            summed.index_add_(0, inverse, per_row)
            counts.index_add_(0, inverse, torch.ones((row_idx.numel(), 1), device=per_row.device, dtype=torch.float32))
            update = summed / counts.clamp_min(1.0)
            kept_rows = unique.numel()
            hit_mean = 0.0
            if engram_shadow_hit_max > 0 and self.hit_hist is not None:
                hit_counts = self.hit_hist.index_select(0, unique).to(device=update.device, dtype=torch.float32)
                hit_mean = float(hit_counts.mean().item()) if hit_counts.numel() else 0.0
                keep = hit_counts <= float(engram_shadow_hit_max)
                kept_rows = int(keep.sum().item())
                if kept_rows == 0:
                    self.last_shadow_metrics = {
                        "rows": int(unique.numel()),
                        "kept_rows": 0,
                        "hit_mean": hit_mean,
                        "grad_rms": float(rms_no_alloc(grad_f).item()),
                        "target_rms": float(rms_no_alloc(target).item()),
                        "update_rms": 0.0,
                        "row_rms": 0.0,
                    }
                    return
                unique = unique[keep]
                update = update[keep]
            current = self.shadow_embedding.weight.index_select(0, unique).float()
            current.mul_(engram_shadow_decay).add_(update, alpha=engram_shadow_write_alpha)
            if engram_shadow_row_rms_cap > 0:
                rms = row_rms_stable(current).unsqueeze(-1)
                current.mul_((engram_shadow_row_rms_cap / rms.clamp_min(1e-12)).clamp_max(1.0))
            self.shadow_embedding.weight.index_copy_(0, unique, current.to(dtype=self.shadow_embedding.weight.dtype))
            self.last_shadow_metrics = {
                "rows": int(unique.numel()),
                "kept_rows": kept_rows,
                "hit_mean": hit_mean,
                "grad_rms": float(rms_no_alloc(grad_f).item()),
                "target_rms": float(rms_no_alloc(target).item()),
                "update_rms": float(rms_no_alloc(update).item()),
                "row_rms": float(rms_no_alloc(current).item()),
            }

    def _apply_shadow_grad(self, addresses: Tensor, output: Tensor) -> Tensor:
        if self.shadow_embedding is None:
            return output
        shadow_heads = self.shadow_embedding(addresses.to(dtype=torch.long)).to(dtype=output.dtype)
        shadow_out = shadow_heads.sum(dim=1) / math.sqrt(addresses.size(1))
        debug_check_finite("engram.shadow_out", shadow_out)
        combined = output + engram_shadow_scale * shadow_out
        debug_check_finite("engram.shadow_combined_output", combined)
        if self.training and combined.requires_grad:
            combined.register_hook(lambda grad: (self._shadow_grad_write_hook(addresses, grad), grad)[1])
        return combined

    def _empty_cpu_table(self, shape, dtype, *, pin: bool) -> Tensor:
        if not pin:
            return torch.empty(shape, dtype=dtype, device="cpu")
        try:
            return torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)
        except RuntimeError:
            return torch.empty(shape, dtype=dtype, device="cpu")

    def _init_offloaded_embedding(self):
        self.wait_offloaded_adam()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(engram_offload_seed)
        self.offload_weight = self._empty_cpu_table(
            (self.num_memory_rows, self.store_dim),
            torch.bfloat16,
            pin=engram_offload_pin,
        )
        if engram_init_zero:
            self.offload_weight.zero_()
        else:
            self.offload_weight.normal_(std=engram_init_std, generator=generator)
        if self._offload_lazy_moments:
            self.offload_exp_avg = torch.empty((0, self.store_dim), dtype=engram_offload_moment_dtype, device="cpu")
            self.offload_exp_avg_sq = torch.empty_like(self.offload_exp_avg) if engram_offload_second_moment else torch.empty((0, 0), dtype=engram_offload_moment_dtype, device="cpu")
            self._offload_moment_rows = torch.empty((0,), dtype=torch.int64, device="cpu")
            self._offload_moment_index = torch.full((self.num_memory_rows,), -1, dtype=torch.int32, device="cpu")
        else:
            self.offload_exp_avg = torch.zeros((self.num_memory_rows, self.store_dim), dtype=engram_offload_moment_dtype, device="cpu")
            self.offload_exp_avg_sq = torch.zeros_like(self.offload_exp_avg) if engram_offload_second_moment else torch.empty((0, 0), dtype=engram_offload_moment_dtype, device="cpu")
            self._offload_moment_rows = None
            self._offload_moment_index = None
        self._offload_pending.clear()
        self._offload_step = 0
        self.last_update_metrics.clear()

    def reset_offloaded_embedding(self):
        if not self.offload:
            return
        self._init_offloaded_embedding()

    def reset_hit_hist(self):
        if self.hit_hist is not None:
            self.hit_hist.zero_()
        self._eval_hit_hist = None

    def _active_hit_hist(self) -> Tensor | None:
        return self._eval_hit_hist if self._eval_hit_hist is not None else self.hit_hist

    def global_hit_hist(self) -> Tensor | None:
        if self.hit_hist is None:
            return None
        hist = self.hit_hist.detach().clone()
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(hist, op=dist.ReduceOp.SUM)
        return hist

    def begin_global_hit_hist_eval(self):
        self._eval_hit_hist = self.global_hit_hist()

    def end_global_hit_hist_eval(self):
        self._eval_hit_hist = None

    def record_hit_hist(self, addresses: Tensor, *, readout_kind: str = "lm"):
        if self.hit_hist is None or not self.training or not torch.is_grad_enabled():
            return
        if readout_kind not in engram_hit_hist_kinds:
            return
        with torch.no_grad():
            rows = addresses.detach().reshape(-1)
            if rows.numel() == 0:
                return
            ones = torch.ones(rows.shape, dtype=self.hit_hist.dtype, device=rows.device)
            self.hit_hist.index_add_(0, rows, ones)

    def read_hit_scale(self, addresses: Tensor) -> Tensor | None:
        hist = self._active_hit_hist()
        if engram_read_hit_scale_exponent == 0 or hist is None:
            self.last_read_hit_scale = None
            return None
        with torch.no_grad():
            counts = hist.index_select(0, addresses.detach().reshape(-1)).view_as(addresses).float()
            scale = (counts + engram_read_hit_scale_offset).pow(engram_read_hit_scale_exponent)
            if engram_read_hit_scale_norm_mean:
                scale = scale / scale.mean().clamp_min(1e-6)
            scale.clamp_(min=engram_read_hit_scale_min, max=engram_read_hit_scale_max)
            self.last_read_hit_scale = scale.detach()
        return scale

    def mask_unhit_eval_memory(self, addresses: Tensor, memory_heads: Tensor) -> Tensor:
        hist = self._active_hit_hist()
        mask_by_hits = engram_mask_unhit_eval or engram_mask_hit_min_eval > 0 or engram_mask_hit_max_eval > 0
        if not mask_by_hits or hist is None or self.training:
            return memory_heads
        with torch.no_grad():
            counts = hist.index_select(0, addresses.reshape(-1)).view_as(addresses)
            hit = counts > 0
            if engram_mask_hit_min_eval > 0:
                hit = hit & (counts >= engram_mask_hit_min_eval)
            if engram_mask_hit_max_eval > 0:
                hit = hit & (counts <= engram_mask_hit_max_eval)
            if engram_mask_hit_invert_eval:
                hit = ~hit
            if addresses.ndim == memory_heads.ndim:
                hit = hit.any(dim=-1)
            hit_mask = hit.unsqueeze(-1)
            if engram_mask_unhit_eval_mode == "zero":
                replacement = torch.zeros((), device=memory_heads.device, dtype=memory_heads.dtype)
            else:
                seed = addresses.detach().to(torch.int64)
                while seed.ndim > memory_heads.ndim - 1:
                    seed = torch.bitwise_xor(seed[..., 0], seed[..., -1])
                dim_ids = torch.arange(1, memory_heads.size(-1) + 1, dtype=torch.int64, device=memory_heads.device)
                random_mix = self._avalanche_hash(
                    seed.unsqueeze(-1)
                    ^ (dim_ids.view(*([1] * seed.ndim), -1) * 0x9E3779B1)
                    ^ int(engram_hash_seed)
                )
                replacement = torch.where(torch.bitwise_and(random_mix, 1) == 0, 1.0, -1.0).to(dtype=memory_heads.dtype)
        return torch.where(hit_mask, memory_heads, replacement)

    def scale_eval_memory_by_hits(self, addresses: Tensor, memory_heads: Tensor) -> Tensor:
        hist = self._active_hit_hist()
        scale_by_hits = (
            engram_eval_hit_scale != 1.0
            and (engram_eval_hit_scale_min > 0 or engram_eval_hit_scale_max > 0)
        )
        if not scale_by_hits or hist is None or self.training:
            return memory_heads
        with torch.no_grad():
            counts = hist.index_select(0, addresses.reshape(-1)).view_as(addresses)
            selected = torch.ones_like(counts, dtype=torch.bool)
            if engram_eval_hit_scale_min > 0:
                selected = selected & (counts >= engram_eval_hit_scale_min)
            if engram_eval_hit_scale_max > 0:
                selected = selected & (counts <= engram_eval_hit_scale_max)
            if engram_eval_hit_scale_invert:
                selected = ~selected
            if addresses.ndim == memory_heads.ndim:
                selected = selected.any(dim=-1)
            scale = torch.where(
                selected,
                torch.full((), engram_eval_hit_scale, device=memory_heads.device, dtype=memory_heads.dtype),
                torch.ones((), device=memory_heads.device, dtype=memory_heads.dtype),
            )
        return memory_heads * scale.unsqueeze(-1)

    def apply_hot_split(self, addresses: Tensor, memory_heads: Tensor) -> Tensor:
        self.last_hot_split_metrics = {}
        aux_scale = engram_hot_split_aux_scale
        if engram_hot_split_aux_scale_schedule_steps > 0:
            schedule_step = float(rom_debug_nan_current_step - engram_hot_split_aux_scale_schedule_start)
            progress = min(max(schedule_step / float(engram_hot_split_aux_scale_schedule_steps), 0.0), 1.0)
            aux_scale = engram_hot_split_aux_scale + progress * (engram_hot_split_aux_scale_final - engram_hot_split_aux_scale)
        if engram_hot_split_ramp_steps > 0:
            aux_scale *= min(1.0, max(0.0, float(rom_debug_nan_current_step)) / float(engram_hot_split_ramp_steps))
        hist = self._active_hit_hist()
        if (
            not engram_hot_split
            or hist is None
            or aux_scale == 0
            or (engram_hot_split_train_only and not self.training)
            or memory_heads.ndim < 2
            or addresses.ndim < 1
        ):
            return memory_heads
        with torch.no_grad():
            counts = hist.index_select(0, addresses.detach().reshape(-1)).view_as(addresses)
            hot = counts >= engram_hot_split_min_hits
            if addresses.ndim == memory_heads.ndim:
                hot = hot.any(dim=-1)
                read_counts = counts.amax(dim=-1)
            else:
                read_counts = counts
            if hot.shape != memory_heads.shape[:-1]:
                self.last_hot_split_metrics = {"shape_mismatch": 1}
                return memory_heads
            hot_count = int(hot.sum().item())
            total_count = max(1, hot.numel())
            self.last_hot_split_metrics = {
                "frac": hot_count / total_count,
                "count": hot_count,
                "aux_scale": aux_scale,
                "mean_hits": float(counts.float().mean().item()),
                "hot_mean_hits": float(read_counts[hot].float().mean().item()) if hot_count > 0 else 0.0,
            }
        if not bool(hot.any().item()):
            return memory_heads

        view_shape = [1] * addresses.ndim
        view_shape[1 if addresses.ndim >= 2 else 0] = -1
        head_mods = self.head_mods.to(device=addresses.device).view(*view_shape)
        offsets = self.offsets.to(device=addresses.device).view(*view_shape)
        relative = (addresses.detach().to(torch.int64) - offsets) % head_mods
        head_salt = torch.arange(1, head_mods.numel() + 1, dtype=torch.int64, device=addresses.device).view(*view_shape)
        slot_shape = [1] * (addresses.ndim + 1)
        slot_shape[-1] = -1
        slot_salt = torch.arange(0, engram_hot_split_aux_slots, dtype=torch.int64, device=addresses.device).view(*slot_shape)
        aux_mix = self._avalanche_hash(
            relative.unsqueeze(-1)
            ^ ((head_salt * 0xC2B2AE3D).unsqueeze(-1))
            ^ (slot_salt * 0x9E3779B1)
            ^ (int(engram_hash_seed) + 0xD1B54A35)
        )
        aux_addresses = (aux_mix % head_mods.unsqueeze(-1)) + offsets.unsqueeze(-1)
        with torch.no_grad():
            self.last_hot_split_metrics["aux_slots"] = engram_hot_split_aux_slots
            self.last_hot_split_metrics["aux_same_frac"] = float((aux_addresses == addresses.unsqueeze(-1)).float().mean().item())
        if engram_hot_split_dedup_aux and not self.offload:
            flat_aux_addresses = aux_addresses.reshape(-1)
            unique_aux_addresses, inverse_aux_addresses = torch.unique(flat_aux_addresses, sorted=False, return_inverse=True)
            with torch.no_grad():
                self.last_hot_split_metrics["aux_unique_rows"] = int(unique_aux_addresses.numel())
                self.last_hot_split_metrics["aux_raw_rows"] = int(flat_aux_addresses.numel())
                self.last_hot_split_metrics["aux_unique_frac"] = float(unique_aux_addresses.numel() / max(1, flat_aux_addresses.numel()))
            aux_memory = self.embedding(unique_aux_addresses).index_select(0, inverse_aux_addresses).view(*aux_addresses.shape, self.store_dim)
        else:
            aux_memory = self._lookup_memory_heads(aux_addresses)
        aux_memory = aux_memory.to(dtype=memory_heads.dtype).sum(dim=-2)
        while aux_memory.ndim > memory_heads.ndim:
            aux_memory = aux_memory.sum(dim=-2)
        if engram_hot_split_detach_aux:
            aux_memory = aux_memory.detach()
        mixed = (memory_heads + aux_memory * aux_scale) / math.sqrt(1.0 + engram_hot_split_aux_slots * aux_scale * aux_scale)
        return torch.where(hot.unsqueeze(-1), mixed, memory_heads)

    def apply_hit_dropout(self, addresses: Tensor, memory_heads: Tensor) -> Tensor:
        hist = self._active_hit_hist()
        hit_dropout = engram_hit_dropout
        if engram_hit_dropout_schedule_steps > 0:
            schedule_step = float(rom_debug_nan_current_step - engram_hit_dropout_schedule_start)
            progress = min(max(schedule_step / float(engram_hit_dropout_schedule_steps), 0.0), 1.0)
            hit_dropout = engram_hit_dropout + progress * (engram_hit_dropout_final - engram_hit_dropout)
        if engram_hit_dropout_decay_steps > 0:
            decay_step = float(rom_debug_nan_current_step - engram_hit_dropout_decay_start)
            decay_progress = min(max(decay_step / float(engram_hit_dropout_decay_steps), 0.0), 1.0)
            hit_dropout = hit_dropout + decay_progress * (engram_hit_dropout_decay_final - hit_dropout)
        if hit_dropout <= 0 or hist is None or not self.training or not torch.is_grad_enabled():
            return memory_heads
        with torch.no_grad():
            counts = hist.index_select(0, addresses.detach().reshape(-1)).view_as(addresses)
            hot = counts >= engram_hit_dropout_min_hits
            if addresses.ndim == memory_heads.ndim:
                hot = hot.any(dim=-1)
            keep = torch.rand(hot.shape, device=memory_heads.device) >= hit_dropout
            keep_scale = keep.to(dtype=memory_heads.dtype)
            if engram_hit_dropout_invert_scale:
                keep_scale = keep_scale / max(1e-6, 1.0 - hit_dropout)
            scale = torch.where(
                hot,
                keep_scale,
                torch.ones((), device=memory_heads.device, dtype=memory_heads.dtype),
            )
        return memory_heads * scale.unsqueeze(-1)

    def hit_hist_summary(self, hist: Tensor | None = None) -> dict[str, float | int | list[int] | list[str]]:
        hist = self._active_hit_hist() if hist is None else hist
        if hist is None:
            return {}
        rows = max(1, hist.numel())
        ever_hit = int((hist > 0).sum().item())
        hit_gt1 = int((hist > 1).sum().item())
        total_hits = int(hist.sum(dtype=torch.int64).item())
        max_hits = int(hist.max().item()) if hist.numel() else 0
        nonzero = hist[hist > 0]
        labels = ["1", "2-3", "4-7", "8-15", "16-31", "32-63", "64-127", "128-255", "256-511", "512-1023", "1024-2047", "2048-4095", "4096-8191", "8192-16383", "16384-32767", "32768+"]
        if nonzero.numel():
            buckets = torch.floor(torch.log2(nonzero.float())).to(torch.int64).clamp_(0, len(labels) - 1)
            bucket_counts = torch.bincount(buckets, minlength=len(labels)).to("cpu", non_blocking=False).tolist()
            mean_hits_touched = float(nonzero.float().mean().item())
        else:
            bucket_counts = [0] * len(labels)
            mean_hits_touched = 0.0
        return {
            "rows": int(hist.numel()),
            "total_hits": total_hits,
            "ever_hit_rows": ever_hit,
            "hit_gt1_rows": hit_gt1,
            "frac_ever_hit": ever_hit / rows,
            "frac_hit_gt1": hit_gt1 / rows,
            "max_hits": max_hits,
            "mean_hits_per_row": total_hits / rows,
            "mean_hits_per_touched_row": mean_hits_touched,
            "log2_bucket_labels": labels,
            "log2_bucket_counts": bucket_counts,
        }

    def save_hit_hist(self, path: str, *, step: int, hit_hist: Tensor | None = None):
        hit_hist = self._active_hit_hist() if hit_hist is None else hit_hist
        if hit_hist is None:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "step": int(step),
            "hit_hist": hit_hist.detach().to("cpu", non_blocking=False),
            "summary": self.hit_hist_summary(hit_hist),
            "num_memory_rows": int(self.num_memory_rows),
            "total_hash_heads": int(self.total_hash_heads),
            "head_mods": self.head_mods.detach().cpu(),
            "offsets": self.offsets.detach().cpu(),
        }
        torch.save(payload, path)

    def offload_state_dict(self) -> dict:
        if not self.offload:
            return {}
        self.wait_offloaded_adam()
        state = {
            "weight": self.offload_weight,
            "exp_avg": self.offload_exp_avg,
            "exp_avg_sq": self.offload_exp_avg_sq,
            "step": self._offload_step,
        }
        if self._offload_lazy_moments:
            state["moment_rows"] = self._offload_moment_rows
            state["lazy_moments"] = True
        return state

    def load_offload_state_dict(self, state: dict):
        if not self.offload or not state:
            return
        self.offload_weight.copy_(state["weight"])
        if self._offload_lazy_moments:
            self.offload_exp_avg = state["exp_avg"].clone()
            self.offload_exp_avg_sq = state["exp_avg_sq"].clone()
            self._offload_moment_rows = state.get("moment_rows", torch.empty((0,), dtype=torch.int64, device="cpu")).clone()
            self._offload_moment_index = torch.full((self.num_memory_rows,), -1, dtype=torch.int32, device="cpu")
            if self._offload_moment_rows.numel():
                self._offload_moment_index[self._offload_moment_rows] = torch.arange(self._offload_moment_rows.numel(), dtype=torch.int32, device="cpu")
        else:
            self.offload_exp_avg.copy_(state["exp_avg"])
            if engram_offload_second_moment:
                self.offload_exp_avg_sq.copy_(state["exp_avg_sq"])
        self._offload_step = int(state.get("step", 0))
        self._offload_pending.clear()
        self._offload_prefetch = None
        self.wait_offloaded_adam()

    def _pin_for_staging(self, rows: Tensor) -> Tensor:
        if not engram_offload_pin_staging:
            return rows
        try:
            return rows.pin_memory()
        except RuntimeError:
            return rows

    def _lookup_offload_moments(self, rows: Tensor) -> tuple[Tensor, Tensor]:
        if not self._offload_lazy_moments:
            m = self.offload_exp_avg.index_select(0, rows)
            if engram_offload_second_moment:
                v = self.offload_exp_avg_sq.index_select(0, rows)
            else:
                v = torch.empty((0, 0), dtype=engram_offload_moment_dtype, device="cpu")
            return m, v
        assert self._offload_moment_rows is not None
        assert self._offload_moment_index is not None
        m = torch.zeros((rows.numel(), self.store_dim), dtype=engram_offload_moment_dtype, device="cpu")
        v = torch.zeros_like(m) if engram_offload_second_moment else torch.empty((0, 0), dtype=engram_offload_moment_dtype, device="cpu")
        if self._offload_moment_rows.numel() == 0 or rows.numel() == 0:
            return m, v
        pos = self._offload_moment_index.index_select(0, rows).to(dtype=torch.int64)
        found = pos >= 0
        if found.any():
            found_pos = pos[found]
            m[found] = self.offload_exp_avg.index_select(0, found_pos)
            if engram_offload_second_moment:
                v[found] = self.offload_exp_avg_sq.index_select(0, found_pos)
        return m, v

    def _store_offload_moments(self, rows: Tensor, m: Tensor, v: Tensor):
        m = m.to(dtype=engram_offload_moment_dtype, device="cpu")
        v = v.to(dtype=engram_offload_moment_dtype, device="cpu") if engram_offload_second_moment else None
        if not self._offload_lazy_moments:
            self.offload_exp_avg[rows] = m
            if engram_offload_second_moment:
                assert v is not None
                self.offload_exp_avg_sq[rows] = v
            return
        assert self._offload_moment_rows is not None
        assert self._offload_moment_index is not None
        if rows.numel() == 0:
            return
        pos = self._offload_moment_index.index_select(0, rows).to(dtype=torch.int64)
        known = pos >= 0
        if known.any():
            known_pos = pos[known]
            self.offload_exp_avg[known_pos] = m[known]
            if engram_offload_second_moment:
                assert v is not None
                self.offload_exp_avg_sq[known_pos] = v[known]
        new = ~known
        if not new.any():
            return
        new_rows = rows[new].clone()
        start = self._offload_moment_rows.numel()
        new_count = new_rows.numel()
        self._offload_moment_index[new_rows] = torch.arange(start, start + new_count, dtype=torch.int32, device="cpu")
        self._offload_moment_rows = torch.cat([self._offload_moment_rows, new_rows])
        self.offload_exp_avg = torch.cat([self.offload_exp_avg, m[new].clone()])
        if engram_offload_second_moment:
            assert v is not None
            self.offload_exp_avg_sq = torch.cat([self.offload_exp_avg_sq, v[new].clone()])

    def _copy_stream_for(self, device: torch.device) -> torch.cuda.Stream:
        if self._offload_stream is None or self._offload_stream_device != device:
            self._offload_stream = torch.cuda.Stream(device=device)
            self._offload_stream_device = device
        return self._offload_stream

    def set_prefetch_moments(self, enabled: bool):
        self._offload_prefetch_moments = enabled

    def _stage_offloaded_rows(self, unique_cpu: Tensor, device: torch.device, need_grad: bool, need_moments: bool, stream: torch.cuda.Stream | None):
        cpu_rows = self.offload_weight.index_select(0, unique_cpu)
        cpu_m, cpu_v = self._lookup_offload_moments(unique_cpu) if need_moments else (None, None)
        cpu_rows = self._pin_for_staging(cpu_rows)
        if cpu_m is not None:
            cpu_m = self._pin_for_staging(cpu_m)
        if cpu_v is not None:
            cpu_v = self._pin_for_staging(cpu_v)
        event = None
        if device.type == "cuda":
            with torch.cuda.device(device):
                assert stream is not None
                with torch.cuda.stream(stream):
                    staged = cpu_rows.to(device=device, non_blocking=cpu_rows.is_pinned())
                    staged_m = cpu_m.to(device=device, non_blocking=cpu_m.is_pinned()) if cpu_m is not None else None
                    staged_v = cpu_v.to(device=device, non_blocking=cpu_v.is_pinned()) if cpu_v is not None else None
                    event = torch.cuda.Event()
                    event.record(stream)
        else:
            staged = cpu_rows.to(device=device)
            staged_m = cpu_m.to(device=device) if cpu_m is not None else None
            staged_v = cpu_v.to(device=device) if cpu_v is not None else None
        if need_grad:
            staged = staged.detach().requires_grad_(True)
        return unique_cpu, staged, staged_m, staged_v, event

    def prefetch(self, input_ids: Tensor):
        if not self.offload or not engram_offload_prefetch:
            return
        self.wait_offloaded_adam()
        if self._offload_prefetch is not None:
            # Drain stale work before replacing it; this should only happen on unusual control flow.
            self._offload_prefetch[-1].result()
            self._offload_prefetch = None

        addresses = self._hash(input_ids)
        flat = addresses.reshape(-1)
        unique_rows, inverse = torch.unique(flat, sorted=False, return_inverse=True)
        unique_cpu = unique_rows.to(device="cpu", non_blocking=False)
        device = input_ids.device
        stream = self._copy_stream_for(device) if device.type == "cuda" else None
        need_grad = self.training and torch.is_grad_enabled()
        need_moments = need_grad and engram_offload_gpu_adam and self._offload_prefetch_moments
        assert self._offload_executor is not None
        future = self._offload_executor.submit(self._stage_offloaded_rows, unique_cpu, device, need_grad, need_moments, stream)
        self._offload_prefetch = (input_ids.data_ptr(), input_ids.numel(), addresses, inverse, future)

    def _consume_prefetched_memory_heads(self, input_ids: Tensor):
        if not self.offload or self._offload_prefetch is None:
            return None
        input_ptr, input_numel, addresses, inverse, future = self._offload_prefetch
        self._offload_prefetch = None
        unique_cpu, staged, staged_m, staged_v, event = future.result()
        if input_ptr != input_ids.data_ptr() or input_numel != input_ids.numel():
            return None
        if event is not None:
            torch.cuda.current_stream(input_ids.device).wait_event(event)
        if self.training and torch.is_grad_enabled():
            self._offload_pending.append((unique_cpu, staged, staged_m, staged_v))
        gathered = staged.index_select(0, inverse)
        return addresses, gathered.view(*addresses.shape, self.store_dim)

    @maybe_disable_engram_compile
    def _lookup_memory_heads(self, addresses: Tensor) -> Tensor:
        if not self.offload:
            flat_addresses = addresses.reshape(-1)
            return self.embedding(flat_addresses).view(*addresses.shape, self.store_dim)

        flat = addresses.reshape(-1)
        unique_rows, inverse = torch.unique(flat, sorted=False, return_inverse=True)
        unique_cpu = unique_rows.to(device="cpu", non_blocking=False)
        cpu_rows = self._pin_for_staging(self.offload_weight.index_select(0, unique_cpu))
        need_moments = self.training and torch.is_grad_enabled() and engram_offload_gpu_adam and self._offload_prefetch_moments
        cpu_m, cpu_v = self._lookup_offload_moments(unique_cpu) if need_moments else (None, None)
        cpu_m = self._pin_for_staging(cpu_m) if cpu_m is not None else None
        cpu_v = self._pin_for_staging(cpu_v) if cpu_v is not None else None
        staged = cpu_rows.to(device=addresses.device, non_blocking=cpu_rows.is_pinned())
        staged_m = cpu_m.to(device=addresses.device, non_blocking=cpu_m.is_pinned()) if cpu_m is not None else None
        staged_v = cpu_v.to(device=addresses.device, non_blocking=cpu_v.is_pinned()) if cpu_v is not None else None
        if self.training and torch.is_grad_enabled():
            staged = staged.detach().requires_grad_(True)
            self._offload_pending.append((unique_cpu, staged, staged_m, staged_v))
        gathered = staged.index_select(0, inverse)
        return gathered.view(*addresses.shape, self.store_dim)

    def _lookup_combined_memory_heads(
        self,
        addresses: Tensor,
        signs: Tensor | None = None,
        dim_signs: Tensor | None = None,
        layer_id: int | None = None,
    ) -> Tensor:
        accum = None
        superpose_base_mix = engram_superpose_include_base and signs is None and dim_signs is None and addresses.size(-1) > 1
        sketch_base_mix = engram_sketch_include_base and signs is not None and addresses.size(-1) > 1
        superpose_aux_scale = engram_superpose_aux_scale
        if superpose_base_mix and engram_superpose_aux_scale_schedule_steps > 0:
            schedule_step = float(rom_debug_nan_current_step - engram_superpose_aux_scale_schedule_start)
            progress = min(max(schedule_step / float(engram_superpose_aux_scale_schedule_steps), 0.0), 1.0)
            superpose_aux_scale = engram_superpose_aux_scale + progress * (engram_superpose_aux_scale_final - engram_superpose_aux_scale)
        sketch_aux_scale = engram_sketch_aux_scale
        if sketch_base_mix and engram_sketch_aux_scale_schedule_steps > 0:
            schedule_step = float(rom_debug_nan_current_step - engram_sketch_aux_scale_schedule_start)
            progress = min(max(schedule_step / float(engram_sketch_aux_scale_schedule_steps), 0.0), 1.0)
            sketch_aux_scale = engram_sketch_aux_scale + progress * (engram_sketch_aux_scale_final - engram_sketch_aux_scale)
        sketch_aux_scale = self._sketch_aux_scale(sketch_aux_scale, dtype=torch.float32, device=addresses.device)
        slot_mix_weights = None
        if engram_sketch_combine_mix and signs is not None and self.sketch_slot_mix_logits is not None:
            slots = addresses.size(-1)
            logits = self.sketch_slot_mix_logits.float()
            if engram_sketch_combine_mix_mode == "bounded":
                slot_mix_weights = 1.0 + engram_sketch_combine_mix_max_dev * torch.tanh(logits)
                slot_mix_weights = slot_mix_weights / slot_mix_weights.mean(dim=-1, keepdim=True).clamp_min(1e-6)
            else:
                slot_mix_weights = F.softmax(logits, dim=-1) * float(slots)
            slot_mix_weights = slot_mix_weights.to(device=addresses.device, dtype=torch.float32)
        for slot in range(addresses.size(-1)):
            memory = self._lookup_memory_heads(addresses[..., slot])
            if signs is not None:
                memory = memory * signs[..., slot].to(dtype=memory.dtype).unsqueeze(-1)
            if dim_signs is not None:
                memory = memory * dim_signs[:, slot, :].to(device=memory.device, dtype=memory.dtype).unsqueeze(0)
            if engram_layer_row_signs and engram_layer_row_signs_aux_only and layer_id is not None and slot > 0:
                memory = memory * self._layer_row_signs(addresses[..., slot], layer_id, memory)
                debug_check_finite("engram.combined_slot_layer_row_signed_memory", memory)
            if slot_mix_weights is not None:
                memory = memory * slot_mix_weights[:, slot].to(dtype=memory.dtype).view(1, self.total_hash_heads, 1)
            if superpose_base_mix and slot > 0:
                memory = memory * superpose_aux_scale
            if sketch_base_mix and slot > 0:
                memory = memory * sketch_aux_scale
            accum = memory if accum is None else accum + memory
        if accum is None:
            raise RuntimeError("Expected at least one Engram lookup slot")
        if superpose_base_mix and engram_superpose_normalize:
            denom = math.sqrt(1.0 + (addresses.size(-1) - 1) * superpose_aux_scale * superpose_aux_scale)
        elif sketch_base_mix:
            denom = torch.sqrt(torch.as_tensor(
                1.0 + (addresses.size(-1) - 1) * sketch_aux_scale * sketch_aux_scale,
                dtype=accum.dtype,
                device=accum.device,
            ))
        elif signs is None and dim_signs is None and engram_superpose_k > 1 and not engram_superpose_normalize:
            denom = 1.0
        else:
            denom = math.sqrt(float(addresses.size(-1)))
        return accum / denom

    def _forward_sketch_slot_readout(
        self,
        input_ids: Tensor,
        hidden_states: Tensor,
        layer_id: int | None,
        readout_kind: str,
        return_banks: bool,
        debug_label: str,
        addresses: Tensor,
        signs: Tensor,
        dim_signs: Tensor | None,
    ) -> Tensor:
        if return_banks:
            raise ValueError("ENGRAM_SKETCH_SLOT_READOUT is not implemented with ENGRAM_BANK_ATTNRES")
        if readout_kind != "lm":
            raise ValueError("ENGRAM_SKETCH_SLOT_READOUT currently supports lm readout only")
        if self.memory_proj is not None:
            raise ValueError("ENGRAM_SKETCH_SLOT_READOUT is not implemented with ENGRAM_STORE_DIM projection")
        readout = self
        readout_key = str(int(layer_id)) if layer_id is not None else ""
        if self.layer_readout_ids and layer_id is not None and readout_key in self.layer_readouts:
            readout = self.layer_readouts[readout_key]
        readout_delta = None
        if readout_kind == "lm" and layer_id is not None and readout_key in self.layer_readout_deltas:
            readout_delta = self.layer_readout_deltas[readout_key]
        readout_delta_value_scale, readout_delta_key_scale = self._layer_readout_delta_scales(
            layer_id, dtype=hidden_states.dtype, device=hidden_states.device
        ) if readout_delta is not None else (0.0, 0.0)
        read_hit_scale = self.read_hit_scale(addresses)
        hit_hist_addresses = addresses[..., :1] if engram_sketch_hit_hist_base_only and addresses.ndim > 0 else addresses
        self.record_hit_hist(hit_hist_addresses, readout_kind=readout_kind)
        accum = None
        last_gate = None
        last_value = None
        slots = addresses.size(-1)
        sketch_aux_scale = engram_sketch_aux_scale
        sketch_base_mix = engram_sketch_include_base and slots > 1
        if sketch_base_mix and engram_sketch_aux_scale_schedule_steps > 0:
            schedule_step = float(rom_debug_nan_current_step - engram_sketch_aux_scale_schedule_start)
            progress = min(max(schedule_step / float(engram_sketch_aux_scale_schedule_steps), 0.0), 1.0)
            sketch_aux_scale = engram_sketch_aux_scale + progress * (engram_sketch_aux_scale_final - engram_sketch_aux_scale)
        sketch_aux_scale = self._sketch_aux_scale(sketch_aux_scale, dtype=hidden_states.dtype, device=hidden_states.device)
        slot_mix_weights = None
        if self.sketch_slot_mix_logits is not None:
            slot_mix_weights = F.softmax(self.sketch_slot_mix_logits.float(), dim=-1)
            slot_mix_weights = slot_mix_weights.to(device=hidden_states.device, dtype=hidden_states.dtype)
            slot_mix_weights = slot_mix_weights * math.sqrt(float(slots))
        slot_attention_values = []
        slot_attention_logits = []
        for slot in range(slots):
            slot_addresses = addresses[..., slot]
            memory_heads = self._lookup_memory_heads(slot_addresses)
            memory_heads = memory_heads * signs[..., slot].to(dtype=memory_heads.dtype).unsqueeze(-1)
            if dim_signs is not None:
                memory_heads = memory_heads * dim_signs[:, slot, :].to(device=memory_heads.device, dtype=memory_heads.dtype).unsqueeze(0)
            memory_heads = self._register_debug_backward(f"{debug_label}.slot{slot}.memory_heads", memory_heads, slot_addresses)
            memory_heads = self.mask_unhit_eval_memory(slot_addresses, memory_heads)
            if engram_shadow_only:
                memory_heads = torch.zeros_like(memory_heads)
            memory_heads = memory_heads.to(dtype=hidden_states.dtype)
            if engram_layer_signs and layer_id is not None:
                memory_heads = memory_heads * self._layer_signs(layer_id, device=memory_heads.device, dtype=memory_heads.dtype)
                debug_check_finite("engram.slot_layer_signed_memory_heads", memory_heads)
            if engram_layer_row_signs and layer_id is not None and (not engram_layer_row_signs_aux_only or slot > 0):
                memory_heads = memory_heads * self._layer_row_signs(slot_addresses, layer_id, memory_heads)
                debug_check_finite("engram.slot_layer_row_signed_memory_heads", memory_heads)
            if engram_normalize_memory_heads:
                memory_heads = norm(memory_heads)
                debug_check_finite("engram.slot_normalized_memory_heads", memory_heads)
            memory_heads = self.apply_hit_dropout(slot_addresses, memory_heads)
            memory_heads = self.scale_eval_memory_by_hits(slot_addresses, memory_heads)
            if readout_kind == "lm" and layer_id is not None and int(layer_id) in engram_detach_memory_layers:
                memory_heads = memory_heads.detach()
                debug_check_finite("engram.slot_detached_layer_memory_heads", memory_heads)

            value_memory = memory_heads.detach() if engram_detach_value_memory else memory_heads
            value = readout.value_proj(value_memory)
            if readout_delta is not None:
                value = value + readout_delta_value_scale * readout_delta.value_proj(value_memory)
            debug_check_finite("engram.slot_value", value)
            if engram_sketch_slot_attention:
                key_memory = memory_heads.detach() if engram_detach_key_memory else memory_heads
                key_raw = readout.key_proj(key_memory)
                if readout_delta is not None:
                    key_raw = key_raw + readout_delta_key_scale * readout_delta.key_proj(key_memory)
                key = norm(key_raw)
                debug_check_finite("engram.slot_attn_key", key)
                query = norm(hidden_states)
                gate = (key * query.unsqueeze(1)).sum(dim=-1) / math.sqrt(hidden_states.size(-1))
                debug_check_finite("engram.slot_attn_logit", gate)
                if read_hit_scale is not None:
                    slot_scale = read_hit_scale[..., slot]
                    value = value * slot_scale.to(dtype=value.dtype).unsqueeze(-1)
                    debug_check_finite("engram.slot_attn_read_hit_scaled_value", value)
                if sketch_base_mix and slot > 0:
                    value = value * sketch_aux_scale
                    debug_check_finite("engram.slot_attn_aux_scaled_value", value)
                slot_attention_values.append(value)
                slot_attention_logits.append(gate)
                last_gate = gate
                last_value = value
                continue
            if engram_fixed_half_gate:
                gate = torch.zeros(value.shape[:-1], device=value.device, dtype=value.dtype)
                gated_value = value * 0.5
            elif engram_static_gate:
                gate = self._static_gate_logits(layer_id, value.size(0), dtype=value.dtype, device=value.device)
                gated_value = value * torch.sigmoid(gate).unsqueeze(-1)
            else:
                key_memory = memory_heads.detach() if engram_detach_key_memory else memory_heads
                key_raw = readout.key_proj(key_memory)
                if readout_delta is not None:
                    key_raw = key_raw + readout_delta_key_scale * readout_delta.key_proj(key_memory)
                key = norm(key_raw)
                debug_check_finite("engram.slot_key", key)
                query = norm(hidden_states)
                gate = (key * query.unsqueeze(1)).sum(dim=-1) / math.sqrt(hidden_states.size(-1))
                debug_check_finite("engram.slot_dot_gate", gate)
                gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
                debug_check_finite("engram.slot_signed_sqrt_gate", gate)
                gated_value = value * torch.sigmoid(gate).unsqueeze(-1)
            if read_hit_scale is not None:
                slot_scale = read_hit_scale[..., slot]
                gated_value = gated_value * slot_scale.to(dtype=gated_value.dtype).unsqueeze(-1)
                debug_check_finite("engram.slot_read_hit_scaled_value", gated_value)
            if sketch_base_mix and slot > 0:
                gated_value = gated_value * sketch_aux_scale
                debug_check_finite("engram.slot_aux_scaled_value", gated_value)
            if slot_mix_weights is not None:
                gated_value = gated_value * slot_mix_weights[:, slot].view(1, self.total_hash_heads, 1)
                debug_check_finite("engram.slot_mixed_value", gated_value)
            accum = gated_value if accum is None else accum + gated_value
            last_gate = gate
            last_value = value

        if accum is None or last_gate is None or last_value is None:
            if not engram_sketch_slot_attention:
                raise RuntimeError("Expected at least one Engram sketch slot")
        if engram_sketch_slot_attention:
            if not slot_attention_values or not slot_attention_logits:
                raise RuntimeError("Expected at least one Engram sketch attention slot")
            value_stack = torch.stack(slot_attention_values, dim=-2)
            logit_stack = torch.stack(slot_attention_logits, dim=-1)
            debug_check_finite("engram.slot_attn_logit_stack", logit_stack)
            slot_attention_weights = F.softmax(logit_stack.float(), dim=-1).to(dtype=value_stack.dtype)
            debug_check_finite("engram.slot_attn_weights", slot_attention_weights)
            gated_value = (value_stack * slot_attention_weights.unsqueeze(-1)).sum(dim=-2)
        elif slot_mix_weights is not None:
            gated_value = accum
        elif sketch_base_mix:
            denom = torch.sqrt(torch.as_tensor(
                1.0 + (slots - 1) * sketch_aux_scale * sketch_aux_scale,
                dtype=accum.dtype,
                device=accum.device,
            ))
            gated_value = accum / denom
        else:
            gated_value = accum / math.sqrt(float(slots))
        debug_check_finite("engram.slot_merged_gated_value", gated_value)
        if self.training and engram_head_dropout_current > 0:
            keep_prob = 1.0 - engram_head_dropout_current
            head_mask = torch.empty(
                gated_value.shape[:2],
                device=gated_value.device,
                dtype=torch.float32,
            ).bernoulli_(keep_prob).to(dtype=gated_value.dtype).unsqueeze(-1)
            gated_value = gated_value * (head_mask / keep_prob)
            debug_check_finite("engram.slot_head_dropout_value", gated_value)
        head_mix_weights = self._head_mix_weights(layer_id, dtype=gated_value.dtype, device=gated_value.device)
        if head_mix_weights is not None:
            gated_value = gated_value * head_mix_weights
            debug_check_finite("engram.slot_head_mixed_value", gated_value)
        ngram_read_scale_weights = self._ngram_read_scale_weights(dtype=gated_value.dtype, device=gated_value.device)
        if ngram_read_scale_weights is not None:
            gated_value = gated_value * ngram_read_scale_weights
            debug_check_finite("engram.slot_ngram_read_scaled_value", gated_value)
        merged_value = gated_value.sum(dim=1)
        if head_mix_weights is None:
            merged_value = merged_value / math.sqrt(self.total_hash_heads)
        debug_check_finite("engram.slot_merged_head_value", merged_value)
        output = apply_rom_short_conv(merged_value, readout.short_conv_norm, readout.short_conv)
        record_engram_analysis(input_ids, addresses, last_gate, last_value, output, gated_value, layer_id=layer_id, readout_kind=readout_kind)
        if self.training and engram_output_dropout_current > 0:
            keep_prob = 1.0 - engram_output_dropout_current
            output_mask = torch.empty(
                output.shape[:-1],
                device=output.device,
                dtype=torch.float32,
            ).bernoulli_(keep_prob).to(dtype=output.dtype).unsqueeze(-1)
            output = output * (output_mask / keep_prob)
            debug_check_finite("engram.slot_output_dropout", output)
        if engram_normalize_readout:
            output = norm(output)
            debug_check_finite("engram.slot_normalized_output", output)
            output = self._apply_shadow_grad(addresses, output)
            output = self._register_debug_backward(f"{debug_label}.output", output)
            record_engram_output_grad_metrics(layer_id, output)
            return output
        output = rom_output_scale * output
        debug_check_finite("engram.slot_scaled_output", output)
        output = self._apply_shadow_grad(addresses, output)
        output = self._register_debug_backward(f"{debug_label}.output", output)
        record_engram_output_grad_metrics(layer_id, output)
        return output

    def first_readout_param(self) -> nn.Parameter:
        if self.layer_readout_ids:
            return next(iter(self.layer_readouts.values())).value_proj.weight
        return self.value_proj.weight

    def wait_offloaded_adam(self):
        if self._offload_adam_future is None:
            return
        self._offload_adam_future.result()
        self._offload_adam_future = None

    def start_offloaded_adam(self, *, do_adam: bool, lr: float, betas: tuple[float, float], eps: float, weight_decay: float = 0.0):
        if not self.offload:
            return
        if not do_adam:
            return
        self.wait_offloaded_adam()
        if not self._offload_pending:
            return
        pending = self._offload_pending
        self._offload_pending = []
        assert self._offload_executor is not None
        self._offload_adam_future = self._offload_executor.submit(
            self._step_offloaded_adam_worker,
            pending,
            lr,
            betas,
            eps,
            weight_decay,
        )

    def step_offloaded_adam(self, *, do_adam: bool, lr: float, betas: tuple[float, float], eps: float, weight_decay: float = 0.0):
        if not self.offload:
            return
        if not do_adam:
            return
        self.wait_offloaded_adam()
        if not self._offload_pending:
            return
        pending = self._offload_pending
        self._offload_pending = []
        self._step_offloaded_adam_worker(pending, lr, betas, eps, weight_decay)

    @torch.no_grad()
    def _step_offloaded_adam_worker(self, pending: list[tuple[Tensor, Tensor, Tensor | None, Tensor | None]], lr: float, betas: tuple[float, float], eps: float, weight_decay: float = 0.0):
        if engram_offload_gpu_adam:
            self._step_offloaded_adam_gpu(pending, lr, betas, eps, weight_decay)
            return
        if not engram_offload_merge_pending:
            self._step_offloaded_adam_unmerged(pending, lr, betas, eps, weight_decay)
            return

        rows = []
        grads = []
        for row_idx, staged, _staged_m, _staged_v in pending:
            grad = staged.grad
            if grad is None:
                continue
            rows.append(row_idx)
            grads.append(grad.detach().to(device="cpu", dtype=torch.float32, non_blocking=False))
            staged.grad = None
        if not rows:
            return

        row_idx = torch.cat(rows)
        grad = torch.cat(grads)
        unique_rows, inverse = torch.unique(row_idx, sorted=False, return_inverse=True)
        grad_sum = torch.zeros((unique_rows.numel(), self.store_dim), dtype=torch.float32, device="cpu")
        grad_sum.index_add_(0, inverse, grad)

        beta1, beta2 = betas
        if not engram_offload_second_moment:
            beta2 = 0.0
        self._offload_step += 1
        bias1 = 1 - beta1 ** self._offload_step
        bias2 = 1 - beta2 ** self._offload_step
        step_size = lr * (bias2 ** 0.5 / bias1)

        m, v = self._lookup_offload_moments(unique_rows)
        m = m.float()
        v = v.float()
        w = self.offload_weight.index_select(0, unique_rows).float()
        m.mul_(beta1).add_(grad_sum, alpha=1 - beta1)
        if engram_offload_second_moment:
            v.mul_(beta2).addcmul_(grad_sum, grad_sum, value=1 - beta2)
        else:
            v = grad_sum.square()
        update = m / v.sqrt().add_(eps)
        if weight_decay:
            mask = (update * w) > 0
            update.addcmul_(w, mask, value=lr * lr * weight_decay)
        if engram_update_metrics and engram_update_metrics_every > 0 and self._offload_step % engram_update_metrics_every == 0:
            table_numel = max(1, self.offload_weight.numel())
            scaled_update = update * abs(step_size)
            grad_rms = grad_sum.norm() / (max(1, grad_sum.numel()) ** 0.5)
            update_rms = scaled_update.norm() / (max(1, scaled_update.numel()) ** 0.5)
            param_rms = w.norm() / (max(1, w.numel()) ** 0.5)
            table_grad_rms = grad_sum.norm() / (table_numel ** 0.5)
            table_update_rms = scaled_update.norm() / (table_numel ** 0.5)
            self.last_update_metrics = {
                "adam_step": self._offload_step,
                "rows": int(unique_rows.numel()),
                "touched_rows": int(unique_rows.numel()),
                "table_rows": int(self.offload_weight.shape[0]),
                "table_numel": int(self.offload_weight.numel()),
                "grad_rms": float(grad_rms.item()),
                "update_rms": float(update_rms.item()),
                "param_rms": float(param_rms.item()),
                "touched_grad_rms": float(grad_rms.item()),
                "touched_update_rms": float(update_rms.item()),
                "touched_param_rms": float(param_rms.item()),
                "table_grad_rms": float(table_grad_rms.item()),
                "table_update_rms": float(table_update_rms.item()),
                "lr": float(lr),
                "step_size": float(step_size),
            }
        w.add_(update, alpha=-step_size)
        self.offload_weight[unique_rows] = w.to(dtype=self.offload_weight.dtype)
        self._store_offload_moments(unique_rows, m, v)

    def _step_offloaded_adam_gpu(self, pending: list[tuple[Tensor, Tensor, Tensor | None, Tensor | None]], lr: float, betas: tuple[float, float], eps: float, weight_decay: float = 0.0):
        rows = []
        weights = []
        moments = []
        variances = []
        grads = []
        for row_idx, staged, m, v in pending:
            grad = staged.grad
            if grad is None:
                continue
            if m is None:
                m_cpu, _ = self._lookup_offload_moments(row_idx)
                m = m_cpu.to(device=staged.device, dtype=torch.float32, non_blocking=False)
            if v is None:
                _, v_cpu = self._lookup_offload_moments(row_idx)
                v = v_cpu.to(device=staged.device, dtype=torch.float32, non_blocking=False)
            rows.append(row_idx)
            weights.append(staged.detach())
            moments.append(m.float())
            variances.append(v.float())
            grads.append(grad.detach())
            staged.grad = None
        if not rows:
            return

        beta1, beta2 = betas
        if not engram_offload_second_moment:
            beta2 = 0.0
        self._offload_step += 1
        bias1 = 1 - beta1 ** self._offload_step
        bias2 = 1 - beta2 ** self._offload_step
        step_size = lr * (bias2 ** 0.5 / bias1)

        row_idx = torch.cat(rows)
        w = torch.cat(weights).float()
        m = torch.cat(moments)
        v = torch.cat(variances)
        grad = torch.cat(grads).float()

        m.mul_(beta1).add_(grad, alpha=1 - beta1)
        if engram_offload_second_moment:
            v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        else:
            v = grad.square()
        update = m / v.sqrt().add_(eps)
        if weight_decay:
            mask = (update * w) > 0
            update.addcmul_(w, mask, value=lr * lr * weight_decay)
        if engram_update_metrics and engram_update_metrics_every > 0 and self._offload_step % engram_update_metrics_every == 0:
            table_numel = max(1, self.offload_weight.numel())
            scaled_update = update * abs(step_size)
            grad_rms = grad.norm() / (max(1, grad.numel()) ** 0.5)
            update_rms = scaled_update.norm() / (max(1, scaled_update.numel()) ** 0.5)
            param_rms = w.norm() / (max(1, w.numel()) ** 0.5)
            table_grad_rms = grad.norm() / (table_numel ** 0.5)
            table_update_rms = scaled_update.norm() / (table_numel ** 0.5)
            self.last_update_metrics = {
                "adam_step": self._offload_step,
                "rows": int(row_idx.numel()),
                "touched_rows": int(row_idx.numel()),
                "table_rows": int(self.offload_weight.shape[0]),
                "table_numel": int(self.offload_weight.numel()),
                "grad_rms": float(grad_rms.item()),
                "update_rms": float(update_rms.item()),
                "param_rms": float(param_rms.item()),
                "touched_grad_rms": float(grad_rms.item()),
                "touched_update_rms": float(update_rms.item()),
                "touched_param_rms": float(param_rms.item()),
                "table_grad_rms": float(table_grad_rms.item()),
                "table_update_rms": float(table_update_rms.item()),
                "lr": float(lr),
                "step_size": float(step_size),
            }
        w.add_(update, alpha=-step_size)

        self.offload_weight[row_idx] = w.to(dtype=self.offload_weight.dtype, device="cpu")
        self._store_offload_moments(row_idx, m, v)

    def _step_offloaded_adam_unmerged(self, pending: list[tuple[Tensor, Tensor, Tensor | None, Tensor | None]], lr: float, betas: tuple[float, float], eps: float, weight_decay: float = 0.0):
        beta1, beta2 = betas
        if not engram_offload_second_moment:
            beta2 = 0.0
        self._offload_step += 1
        bias1 = 1 - beta1 ** self._offload_step
        bias2 = 1 - beta2 ** self._offload_step
        step_size = lr * (bias2 ** 0.5 / bias1)

        for row_idx, staged, _staged_m, _staged_v in pending:
            grad = staged.grad
            if grad is None:
                continue
            grad = grad.detach().to(device="cpu", dtype=torch.float32, non_blocking=False)
            staged.grad = None

            m, v = self._lookup_offload_moments(row_idx)
            m = m.float()
            v = v.float()
            w = self.offload_weight.index_select(0, row_idx).float()
            m.mul_(beta1).add_(grad, alpha=1 - beta1)
            if engram_offload_second_moment:
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            else:
                v = grad.square()
            update = m / v.sqrt().add_(eps)
            if weight_decay:
                mask = (update * w) > 0
                update.addcmul_(w, mask, value=lr * lr * weight_decay)
            if engram_update_metrics and engram_update_metrics_every > 0 and self._offload_step % engram_update_metrics_every == 0:
                table_numel = max(1, self.offload_weight.numel())
                scaled_update = update * abs(step_size)
                grad_rms = grad.norm() / (max(1, grad.numel()) ** 0.5)
                update_rms = scaled_update.norm() / (max(1, scaled_update.numel()) ** 0.5)
                param_rms = w.norm() / (max(1, w.numel()) ** 0.5)
                table_grad_rms = grad.norm() / (table_numel ** 0.5)
                table_update_rms = scaled_update.norm() / (table_numel ** 0.5)
                self.last_update_metrics = {
                    "adam_step": self._offload_step,
                    "rows": int(row_idx.numel()),
                    "touched_rows": int(row_idx.numel()),
                    "table_rows": int(self.offload_weight.shape[0]),
                    "table_numel": int(self.offload_weight.numel()),
                    "grad_rms": float(grad_rms.item()),
                    "update_rms": float(update_rms.item()),
                    "param_rms": float(param_rms.item()),
                    "touched_grad_rms": float(grad_rms.item()),
                    "touched_update_rms": float(update_rms.item()),
                    "touched_param_rms": float(param_rms.item()),
                    "table_grad_rms": float(table_grad_rms.item()),
                    "table_update_rms": float(table_update_rms.item()),
                    "lr": float(lr),
                    "step_size": float(step_size),
                }
            w.add_(update, alpha=-step_size)
            self.offload_weight[row_idx] = w.to(dtype=self.offload_weight.dtype)
            self._store_offload_moments(row_idx, m, v)

    def _hash(self, input_ids: Tensor, layer_id: int | None = None) -> Tensor:
        x = input_ids.to(torch.int64)
        if self.canonical_lookup is not None:
            x = self.canonical_lookup[x.clamp_min(0).clamp_max(self.canonical_lookup.numel() - 1)]
        multipliers = self.multipliers
        if layer_id is not None and self.layer_hash_ids:
            layer_idx = self.layer_hash_index.get(int(layer_id))
            if layer_idx is not None:
                multipliers = self.layer_multipliers[layer_idx]
        head_mods = self.head_mods
        offsets = self.offsets
        if layer_id is not None and self.layer_partition_ids:
            layer_idx = self.layer_partition_index.get(int(layer_id))
            if layer_idx is not None:
                head_mods = self.layer_head_mods[layer_idx]
                offsets = self.layer_offsets[layer_idx]
        shifted = [x]
        for k in range(1, self.max_ngram):
            pad = torch.full((k,), self.hash_pad_id, dtype=torch.int64, device=x.device)
            shifted.append(torch.cat([pad, x[:-k]], dim=0))

        addresses = []
        head_idx = 0
        for ngram in range(2, self.max_ngram + 1):
            mix = shifted[0] * multipliers[0]
            for k in range(1, ngram):
                mix = torch.bitwise_xor(mix, shifted[k] * multipliers[k])
            if engram_avalanche_hash:
                mix = self._avalanche_hash(mix ^ int(engram_hash_seed))
            for _head in range(self.num_heads):
                addresses.append((mix % head_mods[head_idx]) + offsets[head_idx])
                head_idx += 1
        return torch.stack(addresses, dim=-1)

    @staticmethod
    def _avalanche_hash(x: Tensor) -> Tensor:
        x = x ^ (x >> 30)
        x = x * -4658895280553007687
        x = x ^ (x >> 27)
        x = x * -7723592293110705685
        return x ^ (x >> 31)

    @staticmethod
    def _balanced_sketch_signs(sign_mix: Tensor, positives: int) -> Tensor:
        slots = sign_mix.size(-1)
        positives = max(0, min(int(positives), int(slots)))
        order = torch.argsort(sign_mix, dim=-1)
        ranks = torch.empty_like(order)
        rank_shape = [1] * sign_mix.ndim
        rank_shape[-1] = slots
        slot_ranks = torch.arange(slots, dtype=torch.int64, device=sign_mix.device).view(*rank_shape).expand_as(order)
        ranks.scatter_(-1, order, slot_ranks)
        return torch.where(ranks < positives, 1.0, -1.0)

    def _hash_sketch(self, input_ids: Tensor, layer_id: int | None = None) -> tuple[Tensor, Tensor | None]:
        if engram_sketch_k == 1:
            return self._hash(input_ids, layer_id=layer_id), None

        x = input_ids.to(torch.int64)
        if self.canonical_lookup is not None:
            x = self.canonical_lookup[x.clamp_min(0).clamp_max(self.canonical_lookup.numel() - 1)]
        multipliers = self.multipliers
        if layer_id is not None and self.layer_hash_ids:
            layer_idx = self.layer_hash_index.get(int(layer_id))
            if layer_idx is not None:
                multipliers = self.layer_multipliers[layer_idx]
        head_mods = self.head_mods
        offsets = self.offsets
        if layer_id is not None and self.layer_partition_ids:
            layer_idx = self.layer_partition_index.get(int(layer_id))
            if layer_idx is not None:
                head_mods = self.layer_head_mods[layer_idx]
                offsets = self.layer_offsets[layer_idx]
        shifted = [x]
        for k in range(1, self.max_ngram):
            pad = torch.full((k,), self.hash_pad_id, dtype=torch.int64, device=x.device)
            shifted.append(torch.cat([pad, x[:-k]], dim=0))

        sketch_salts = self.sketch_salts.to(device=x.device)
        sign_salts = self.sketch_sign_salts.to(device=x.device)
        addresses = []
        signs = []
        head_idx = 0
        for ngram in range(2, self.max_ngram + 1):
            mix = shifted[0] * multipliers[0]
            for k in range(1, ngram):
                mix = torch.bitwise_xor(mix, shifted[k] * multipliers[k])
            if engram_avalanche_hash:
                mix = self._avalanche_hash(mix ^ int(engram_hash_seed))
            for _head in range(self.num_heads):
                head_salt = torch.tensor((head_idx + 1) * 0xC2B2AE3D, dtype=torch.int64, device=x.device)
                if engram_sketch_include_base:
                    base_address = ((mix % head_mods[head_idx]) + offsets[head_idx]).unsqueeze(-1)
                    salt_count = engram_sketch_k - 1
                    sketch_mix = self._avalanche_hash(torch.bitwise_xor(mix.unsqueeze(-1), sketch_salts[:salt_count].view(1, -1) + head_salt))
                    sign_mix = self._avalanche_hash(torch.bitwise_xor(mix.unsqueeze(-1), sign_salts[:salt_count].view(1, -1) + head_salt))
                    aux_addresses = (sketch_mix % head_mods[head_idx]) + offsets[head_idx]
                    base_sign = torch.ones_like(mix, dtype=torch.float32).unsqueeze(-1)
                    if engram_sketch_scalar_signs:
                        if engram_sketch_scalar_sign_mode == "balanced":
                            aux_signs = self._balanced_sketch_signs(sign_mix, (engram_sketch_k + 1) // 2 - 1)
                        else:
                            aux_signs = torch.where(torch.bitwise_and(sign_mix, 1) == 0, 1.0, -1.0)
                    else:
                        aux_signs = torch.ones_like(sign_mix, dtype=torch.float32)
                    addresses.append(torch.cat((base_address, aux_addresses), dim=-1))
                    signs.append(torch.cat((base_sign, aux_signs), dim=-1))
                else:
                    sketch_mix = self._avalanche_hash(torch.bitwise_xor(mix.unsqueeze(-1), sketch_salts.view(1, -1) + head_salt))
                    sign_mix = self._avalanche_hash(torch.bitwise_xor(mix.unsqueeze(-1), sign_salts.view(1, -1) + head_salt))
                    addresses.append((sketch_mix % head_mods[head_idx]) + offsets[head_idx])
                    if engram_sketch_scalar_signs:
                        if engram_sketch_scalar_sign_mode == "balanced":
                            signs.append(self._balanced_sketch_signs(sign_mix, (engram_sketch_k + 1) // 2))
                        else:
                            signs.append(torch.where(torch.bitwise_and(sign_mix, 1) == 0, 1.0, -1.0))
                    else:
                        signs.append(torch.ones_like(sign_mix, dtype=torch.float32))
                head_idx += 1
        return torch.stack(addresses, dim=1), torch.stack(signs, dim=1)

    def _hash_superpose(self, input_ids: Tensor, layer_id: int | None = None) -> Tensor:
        if engram_superpose_k == 1:
            return self._hash(input_ids, layer_id=layer_id)

        x = input_ids.to(torch.int64)
        if self.canonical_lookup is not None:
            x = self.canonical_lookup[x.clamp_min(0).clamp_max(self.canonical_lookup.numel() - 1)]
        multipliers = self.multipliers
        if layer_id is not None and self.layer_hash_ids:
            layer_idx = self.layer_hash_index.get(int(layer_id))
            if layer_idx is not None:
                multipliers = self.layer_multipliers[layer_idx]
        head_mods = self.head_mods
        offsets = self.offsets
        if layer_id is not None and self.layer_partition_ids:
            layer_idx = self.layer_partition_index.get(int(layer_id))
            if layer_idx is not None:
                head_mods = self.layer_head_mods[layer_idx]
                offsets = self.layer_offsets[layer_idx]
        shifted = [x]
        for k in range(1, self.max_ngram):
            pad = torch.full((k,), self.hash_pad_id, dtype=torch.int64, device=x.device)
            shifted.append(torch.cat([pad, x[:-k]], dim=0))

        superpose_salts = self.superpose_salts.to(device=x.device)
        addresses = []
        head_idx = 0
        for ngram in range(2, self.max_ngram + 1):
            mix = shifted[0] * multipliers[0]
            for k in range(1, ngram):
                mix = torch.bitwise_xor(mix, shifted[k] * multipliers[k])
            if engram_avalanche_hash:
                mix = self._avalanche_hash(mix ^ int(engram_hash_seed))
            for _head in range(self.num_heads):
                head_addresses = []
                if engram_superpose_include_base:
                    head_addresses.append((mix % head_mods[head_idx]) + offsets[head_idx])
                head_salt = torch.tensor((head_idx + 1) * 0xC2B2AE3D, dtype=torch.int64, device=x.device)
                salt_count = engram_superpose_k - 1 if engram_superpose_include_base else engram_superpose_k
                superpose_mix = self._avalanche_hash(torch.bitwise_xor(mix.unsqueeze(-1), superpose_salts[:salt_count].view(1, -1) + head_salt))
                head_addresses.append((superpose_mix % head_mods[head_idx]) + offsets[head_idx])
                addresses.append(torch.cat([addr.unsqueeze(-1) if addr.ndim == 1 else addr for addr in head_addresses], dim=-1))
                head_idx += 1
        return torch.stack(addresses, dim=1)

    @staticmethod
    def _scheduled_scale(start: float, final: float, steps: int, schedule_start: int) -> float:
        if steps <= 0:
            return start
        schedule_step = float(rom_debug_nan_current_step - schedule_start)
        progress = min(max(schedule_step / float(steps), 0.0), 1.0)
        return start + progress * (final - start)

    def _raw_layer_signs(self, layer_id: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        head_idx = torch.arange(1, self.total_hash_heads + 1, dtype=torch.int64, device=device)
        layer_salt = torch.full_like(head_idx, int(layer_id) * 0x85EBCA77 + engram_hash_seed)
        sign_mix = self._avalanche_hash(torch.bitwise_xor(head_idx * 0x9E3779B1, layer_salt))
        signs = torch.where(torch.bitwise_and(sign_mix, 1) == 0, 1.0, -1.0)
        return signs.to(dtype=dtype).view(1, self.total_hash_heads, 1)

    def _layer_signs(self, layer_id: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        signs = self._raw_layer_signs(layer_id, device=device, dtype=dtype)
        scale = engram_layer_sign_scale
        scale = self._scheduled_scale(
            engram_layer_sign_scale,
            engram_layer_sign_scale_final,
            engram_layer_sign_scale_schedule_steps,
            engram_layer_sign_scale_schedule_start,
        )
        if scale <= 0.0:
            signs = torch.ones_like(signs)
        elif scale < 1.0:
            signs = 1.0 + scale * (signs - 1.0)
        return signs

    def _layer_sign_aux_scale(self) -> float:
        return self._scheduled_scale(
            engram_layer_sign_aux_scale,
            engram_layer_sign_aux_scale_final,
            engram_layer_sign_aux_scale_schedule_steps,
            engram_layer_sign_aux_scale_schedule_start,
        )

    def _layer_readout_delta_scale(self) -> float:
        return self._scheduled_scale(
            engram_layer_readout_delta_scale,
            engram_layer_readout_delta_scale_final,
            engram_layer_readout_delta_scale_schedule_steps,
            engram_layer_readout_delta_scale_schedule_start,
        )

    def _layer_readout_delta_scales(self, layer_id: int | None, *, dtype: torch.dtype, device: torch.device) -> tuple[Tensor | float, Tensor | float]:
        base_scale = self._layer_readout_delta_scale()
        if self.layer_readout_delta_scale_logits is None or layer_id is None:
            return base_scale, base_scale
        layer_idx = self.layer_readout_delta_index.get(int(layer_id))
        if layer_idx is None:
            return base_scale, base_scale
        learned = engram_layer_readout_delta_learned_scale_max * torch.sigmoid(
            self.layer_readout_delta_scale_logits[layer_idx].float()
        )
        learned = learned.to(device=device, dtype=dtype) * base_scale
        return learned[0], learned[1]

    def layer_readout_delta_scales_snapshot(self) -> Tensor | None:
        if self.layer_readout_delta_scale_logits is None:
            return None
        return engram_layer_readout_delta_learned_scale_max * torch.sigmoid(
            self.layer_readout_delta_scale_logits.detach().float()
        )

    def _layer_row_signs(self, addresses: Tensor, layer_id: int, memory_heads: Tensor) -> Tensor:
        seed = addresses.detach().to(dtype=torch.int64)
        while seed.ndim > memory_heads.ndim - 1:
            seed = torch.bitwise_xor(seed[..., 0], seed[..., -1])
        dim_ids = torch.arange(1, memory_heads.size(-1) + 1, dtype=torch.int64, device=memory_heads.device)
        layer_salt = int(layer_id) * 0x85EBCA77 + int(engram_hash_seed)
        sign_mix = self._avalanche_hash(
            seed.unsqueeze(-1)
            ^ (dim_ids.view(*([1] * seed.ndim), -1) * 0x9E3779B1)
            ^ layer_salt
        )
        signs = torch.where(torch.bitwise_and(sign_mix, 1) == 0, 1.0, -1.0).to(dtype=memory_heads.dtype)
        scale = self._scheduled_scale(
            engram_layer_row_sign_scale,
            engram_layer_row_sign_scale_final,
            engram_layer_row_sign_scale_schedule_steps,
            engram_layer_row_sign_scale_schedule_start,
        )
        if scale <= 0.0:
            return torch.ones_like(signs)
        if scale < 1.0:
            return 1.0 + scale * (signs - 1.0)
        return signs

    def _head_mix_weights(self, layer_id: int | None, *, dtype: torch.dtype, device: torch.device) -> Tensor | None:
        logits = None
        if self.layer_head_mix_logits is not None and layer_id is not None:
            layer_idx = self.layer_head_mix_index.get(int(layer_id))
            if layer_idx is not None:
                logits = self.layer_head_mix_logits[layer_idx]
        elif self.head_mix_logits is not None:
            logits = self.head_mix_logits
            if self.layer_head_mix_delta_logits is not None and layer_id is not None:
                layer_idx = self.layer_head_mix_index.get(int(layer_id))
                if layer_idx is not None:
                    logits = logits + self.layer_head_mix_delta_logits[layer_idx]
        if logits is None:
            return None
        weights = F.softmax(logits.float(), dim=-1) * math.sqrt(float(self.total_hash_heads))
        return weights.to(device=device, dtype=dtype).view(1, self.total_hash_heads, 1)

    def _ngram_read_scale_weights(self, *, dtype: torch.dtype, device: torch.device) -> Tensor | None:
        if not engram_ngram_read_scales:
            return None
        start = torch.tensor(engram_ngram_read_scales, dtype=torch.float32, device=device)
        if engram_ngram_read_scale_schedule_steps > 0:
            final = torch.tensor(engram_ngram_read_scales_final, dtype=torch.float32, device=device)
            progress = min(max(float(rom_debug_nan_current_step) / float(engram_ngram_read_scale_schedule_steps), 0.0), 1.0)
            scales = start + progress * (final - start)
        else:
            scales = start
        if engram_ngram_read_scale_norm:
            scales = scales / scales.square().mean().sqrt().clamp_min(1e-6)
        return scales.repeat_interleave(self.num_heads).to(dtype=dtype).view(1, self.total_hash_heads, 1)

    def sketch_slot_mix_weights(self) -> Tensor | None:
        if self.sketch_slot_mix_logits is None:
            return None
        logits = self.sketch_slot_mix_logits.detach().float()
        if engram_sketch_combine_mix and engram_sketch_combine_mix_mode == "bounded":
            weights = 1.0 + engram_sketch_combine_mix_max_dev * torch.tanh(logits)
            return weights / weights.mean(dim=-1, keepdim=True).clamp_min(1e-6)
        return F.softmax(logits, dim=-1)

    def _sketch_aux_scale(self, base_scale: float, *, dtype: torch.dtype, device: torch.device) -> Tensor | float:
        if self.sketch_aux_scale_logit is None:
            return base_scale
        learned = engram_sketch_aux_learned_scale_max * torch.sigmoid(self.sketch_aux_scale_logit.float())
        return torch.as_tensor(base_scale, dtype=dtype, device=device) * learned.to(device=device, dtype=dtype)

    def sketch_aux_scale_snapshot(self) -> Tensor | None:
        if self.sketch_aux_scale_logit is None:
            return None
        return engram_sketch_aux_learned_scale_max * torch.sigmoid(self.sketch_aux_scale_logit.detach().float())

    def _static_gate_logits(self, layer_id: int | None, tokens: int, *, dtype: torch.dtype, device: torch.device) -> Tensor:
        if self.static_gate_logits is None:
            raise RuntimeError("ENGRAM_STATIC_GATE requested without static_gate_logits")
        row = 0
        if self.layer_hash_ids and layer_id is not None:
            row = self.layer_hash_index.get(int(layer_id), 0)
        logits = self.static_gate_logits[row].to(device=device, dtype=dtype)
        if self.per_head:
            return logits.view(1, self.total_hash_heads).expand(tokens, -1)
        return logits.view(1).expand(tokens)

    def _latent_addresses(self, hidden_states: Tensor, layer_id: int | None = None) -> tuple[Tensor, Tensor]:
        latent = self.latent_proj(norm(hidden_states)).view(hidden_states.size(0), self.total_hash_heads, self.latent_dim)
        latent = latent.float() * self.latent_input_scale
        if self.latent_quantizer == "bsq":
            latent = latent / latent.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
            hard = torch.where(latent >= 0, torch.ones_like(latent), -torch.ones_like(latent))
            quantized = latent + (hard - latent).detach()
            bits = hard.gt(0).to(torch.int64)
            codes = (bits * self.latent_basis.to(device=hidden_states.device).view(1, 1, -1)).sum(dim=-1)
            latent_features = quantized.to(dtype=hidden_states.dtype)
        else:
            levels = self.latent_levels.to(device=hidden_states.device)
            half_width = self.latent_half_width.to(device=hidden_states.device)
            half_l = (levels - 1.0) * (1.0 - engram_latent_fsq_eps) * 0.5
            offset = torch.where((levels.to(torch.int64) % 2) == 1, torch.zeros_like(levels), torch.full_like(levels, 0.5))
            shift = torch.tan(offset / half_l)
            bounded = torch.tanh(latent + shift) * half_l - offset
            rounded = bounded.round()
            quantized = bounded + (rounded - bounded).detach()
            indexes = (rounded + half_width).clamp_min(0)
            indexes = torch.minimum(indexes, (levels - 1.0).view(1, 1, -1))
            codes = (indexes.to(torch.int64) * self.latent_basis.to(device=hidden_states.device).view(1, 1, -1)).sum(dim=-1)
            latent_features = (quantized / half_width).to(dtype=hidden_states.dtype)
        if layer_id is not None and self.layer_hash_ids:
            layer_idx = self.layer_hash_index.get(int(layer_id))
            if layer_idx is not None:
                head_salt = torch.arange(1, self.total_hash_heads + 1, dtype=torch.int64, device=hidden_states.device)
                codes = codes + (layer_idx + 1) * 1000003 * head_salt.view(1, -1)
        addresses = (codes % self.head_mods.to(device=hidden_states.device)) + self.offsets.to(device=hidden_states.device)
        return addresses, latent_features

    def _latent_pkm_memory_heads(self, hidden_states: Tensor, layer_id: int | None = None) -> tuple[Tensor, Tensor]:
        query = self.latent_proj(norm(hidden_states)).view(hidden_states.size(0), self.total_hash_heads, 2, self.latent_pkm_key_dim).float()
        query = F.normalize(query, dim=-1) * self.latent_input_scale
        keys = F.normalize(self.latent_pkm_keys.float(), dim=-1)
        scores_a = torch.einsum("thd,hnd->thn", query[:, :, 0], keys[:, 0])
        scores_b = torch.einsum("thd,hnd->thn", query[:, :, 1], keys[:, 1])
        vals_a, idx_a = torch.topk(scores_a, self.latent_pkm_topk, dim=-1)
        vals_b, idx_b = torch.topk(scores_b, self.latent_pkm_topk, dim=-1)
        candidate_scores = vals_a.unsqueeze(-1) + vals_b.unsqueeze(-2)
        candidate_weights = torch.softmax(candidate_scores.flatten(start_dim=-2), dim=-1).to(dtype=hidden_states.dtype)
        candidate_relative = (idx_a.unsqueeze(-1) * self.latent_pkm_subkeys + idx_b.unsqueeze(-2)).flatten(start_dim=-2)
        if layer_id is not None and self.layer_hash_ids:
            layer_idx = self.layer_hash_index.get(int(layer_id))
            if layer_idx is not None:
                head_salt = torch.arange(1, self.total_hash_heads + 1, dtype=torch.int64, device=hidden_states.device)
                layer_salt = (layer_idx + 1) * 1000003 * head_salt.view(1, -1, 1)
                candidate_relative = (candidate_relative + layer_salt) % self.latent_codebook_size
        offsets = self.offsets.to(device=hidden_states.device)
        addresses = candidate_relative + offsets.view(1, -1, 1)
        candidate_memory = self._lookup_memory_heads(addresses)
        memory_heads = (candidate_memory * candidate_weights.unsqueeze(-1)).sum(dim=-2)
        return addresses[..., 0], memory_heads

    @maybe_disable_engram_compile
    def forward(self, input_ids: Tensor, hidden_states: Tensor, layer_id: int | None = None, readout_kind: str = "lm", return_banks: bool = False) -> Tensor:
        assert input_ids.ndim == 1
        assert hidden_states.ndim == 2
        if return_banks and (not self.per_head or readout_kind != "lm"):
            raise ValueError("Engram bank outputs require ENGRAM_PER_HEAD=1 and readout_kind='lm'")
        if return_banks and self.shadow_embedding is not None:
            raise ValueError("ENGRAM_SHADOW_GRAD is not implemented for ENGRAM_BANK_ATTNRES")
        hash_layer_id = layer_id if (engram_layer_hashes or self.layer_partition_ids) else None
        latent_features = None
        if self.latent and engram_latent_aux_readout:
            if engram_superpose_k > 1:
                addresses = self._hash_superpose(input_ids, layer_id=hash_layer_id)
                memory_heads = self._lookup_combined_memory_heads(addresses, layer_id=layer_id)
            else:
                addresses, sketch_signs = self._hash_sketch(input_ids, layer_id=hash_layer_id)
                if sketch_signs is None:
                    memory_heads = self._lookup_memory_heads(addresses)
                elif engram_sketch_slot_readout:
                    raise ValueError("ENGRAM_LATENT_AUX_READOUT is not implemented with ENGRAM_SKETCH_SLOT_READOUT")
                else:
                    dim_signs = self.sketch_dim_signs if engram_sketch_dim_signs else None
                    memory_heads = self._lookup_combined_memory_heads(
                        addresses, sketch_signs, dim_signs=dim_signs, layer_id=layer_id
                    )
            if self.latent_quantizer == "pkm":
                if self.latent_mix_ngram:
                    raise ValueError("ENGRAM_LATENT_MIX_NGRAM is not supported with PKM")
                latent_addresses, latent_memory_heads = self._latent_pkm_memory_heads(hidden_states, layer_id=hash_layer_id)
            else:
                latent_addresses, latent_features = self._latent_addresses(hidden_states, layer_id=hash_layer_id)
                if self.latent_mix_ngram:
                    base_addresses = self._hash(input_ids, layer_id=hash_layer_id)
                    head_mods = self.head_mods.to(device=hidden_states.device)
                    offsets = self.offsets.to(device=hidden_states.device)
                    latent_relative = (latent_addresses - offsets) % head_mods
                    base_relative = (base_addresses - offsets) % head_mods
                    latent_addresses = (torch.bitwise_xor(base_relative, latent_relative) % head_mods) + offsets
                latent_memory_heads = self._lookup_memory_heads(latent_addresses)
            debug_record_engram_addresses(
                f"engram.layer{layer_id if layer_id is not None else 'shared'}.{readout_kind}.latent_aux",
                latent_addresses,
            )
            latent_memory_heads = self._register_debug_backward(
                f"engram.layer{layer_id if layer_id is not None else 'shared'}.{readout_kind}.latent_aux_memory_heads",
                latent_memory_heads,
                latent_addresses,
            )
            latent_aux_scale = self._scheduled_scale(
                engram_latent_aux_scale,
                engram_latent_aux_scale_final,
                engram_latent_aux_scale_schedule_steps,
                engram_latent_aux_scale_schedule_start,
            )
            memory_heads = memory_heads + latent_aux_scale * latent_memory_heads
        elif self.latent:
            if self.latent_quantizer == "pkm":
                if self.latent_mix_ngram:
                    raise ValueError("ENGRAM_LATENT_MIX_NGRAM is not supported with PKM")
                addresses, memory_heads = self._latent_pkm_memory_heads(hidden_states, layer_id=hash_layer_id)
            else:
                addresses, latent_features = self._latent_addresses(hidden_states, layer_id=hash_layer_id)
                if self.latent_mix_ngram:
                    ngram_addresses = self._hash(input_ids, layer_id=hash_layer_id)
                    head_mods = self.head_mods.to(device=hidden_states.device)
                    offsets = self.offsets.to(device=hidden_states.device)
                    latent_relative = (addresses - offsets) % head_mods
                    ngram_relative = (ngram_addresses - offsets) % head_mods
                    addresses = (torch.bitwise_xor(ngram_relative, latent_relative) % head_mods) + offsets
                memory_heads = self._lookup_memory_heads(addresses)
        else:
            prefetched = None if hash_layer_id is not None or engram_sketch_k > 1 or engram_superpose_k > 1 else self._consume_prefetched_memory_heads(input_ids)
            if prefetched is None:
                if engram_superpose_k > 1:
                    addresses = self._hash_superpose(input_ids, layer_id=hash_layer_id)
                    memory_heads = self._lookup_combined_memory_heads(addresses, layer_id=layer_id)
                else:
                    addresses, sketch_signs = self._hash_sketch(input_ids, layer_id=hash_layer_id)
                    if sketch_signs is None:
                        memory_heads = self._lookup_memory_heads(addresses)
                    elif engram_sketch_slot_readout:
                        memory_heads = None
                    else:
                        dim_signs = self.sketch_dim_signs if engram_sketch_dim_signs else None
                        memory_heads = self._lookup_combined_memory_heads(
                            addresses, sketch_signs, dim_signs=dim_signs, layer_id=layer_id
                        )
            else:
                addresses, memory_heads = prefetched
        debug_label = f"engram.layer{layer_id if layer_id is not None else 'shared'}.{readout_kind}"
        if not self.latent and engram_sketch_slot_readout and engram_sketch_k > 1 and memory_heads is None:
            dim_signs = self.sketch_dim_signs if engram_sketch_dim_signs else None
            return self._forward_sketch_slot_readout(
                input_ids,
                hidden_states,
                layer_id,
                readout_kind,
                return_banks,
                debug_label,
                addresses,
                sketch_signs,
                dim_signs,
            )
        debug_record_engram_addresses(debug_label, addresses)
        memory_heads = self._register_debug_backward(f"{debug_label}.memory_heads", memory_heads, addresses)
        read_hit_scale = self.read_hit_scale(addresses)
        value_memory_heads = self.apply_hot_split(addresses, memory_heads) if engram_hot_split_value_only else None
        if not engram_hot_split_value_only:
            memory_heads = self.apply_hot_split(addresses, memory_heads)
        hit_hist_addresses = addresses
        if engram_sketch_hit_hist_base_only and engram_sketch_k > 1 and addresses.ndim > 0:
            hit_hist_addresses = addresses[..., :1]
        self.record_hit_hist(hit_hist_addresses, readout_kind=readout_kind)

        def finalize_memory_heads(raw_memory_heads: Tensor, *, debug_suffix: str = "") -> Tensor:
            final_memory_heads = self.mask_unhit_eval_memory(addresses, raw_memory_heads)
            if engram_shadow_only:
                final_memory_heads = torch.zeros_like(final_memory_heads)
            final_memory_heads = final_memory_heads.to(dtype=hidden_states.dtype)
            if engram_layer_signs and layer_id is not None:
                final_memory_heads = final_memory_heads * self._layer_signs(
                    layer_id, device=final_memory_heads.device, dtype=final_memory_heads.dtype
                )
                debug_check_finite(f"engram{debug_suffix}.layer_signed_memory_heads", final_memory_heads)
            if engram_layer_row_signs and layer_id is not None and not engram_layer_row_signs_aux_only:
                final_memory_heads = final_memory_heads * self._layer_row_signs(addresses, layer_id, final_memory_heads)
                debug_check_finite(f"engram{debug_suffix}.layer_row_signed_memory_heads", final_memory_heads)
            if latent_features is not None and self.latent_ste_proj is not None:
                final_memory_heads = final_memory_heads + self.latent_ste_scale * self.latent_ste_proj(latent_features)
            if engram_normalize_memory_heads:
                final_memory_heads = norm(final_memory_heads)
                debug_check_finite(f"engram{debug_suffix}.normalized_memory_heads", final_memory_heads)
            final_memory_heads = self.apply_hit_dropout(addresses, final_memory_heads)
            final_memory_heads = self.scale_eval_memory_by_hits(addresses, final_memory_heads)
            if readout_kind == "lm" and layer_id is not None and int(layer_id) in engram_detach_memory_layers:
                final_memory_heads = final_memory_heads.detach()
                debug_check_finite(f"engram{debug_suffix}.detached_layer_memory_heads", final_memory_heads)
            return final_memory_heads

        memory_heads = finalize_memory_heads(memory_heads)
        if value_memory_heads is not None:
            value_memory_heads = finalize_memory_heads(value_memory_heads, debug_suffix=".value")
        if self.per_head:
            memory = memory_heads
            value_memory_base = value_memory_heads if value_memory_heads is not None else memory
        else:
            memory = memory_heads.flatten(start_dim=-2)
            value_memory_base = value_memory_heads.flatten(start_dim=-2) if value_memory_heads is not None else memory
        debug_check_finite("engram.memory", memory)
        if self.memory_proj is not None:
            memory = self.memory_proj(memory)
            debug_check_finite("engram.projected_memory", memory)
            if value_memory_heads is not None:
                value_memory_base = self.memory_proj(value_memory_base)
                debug_check_finite("engram.value_projected_memory", value_memory_base)
        if readout_kind == "cache" and engram_cache_detach_memory:
            memory = memory.detach()
            value_memory_base = value_memory_base.detach()
        readout = self
        readout_key = str(int(layer_id)) if layer_id is not None else ""
        if readout_kind == "cache":
            if readout_key not in self.cache_readouts:
                raise ValueError(f"No Engram cache readout for layer_id={layer_id}")
            readout = self.cache_readouts[readout_key]
        elif self.layer_readout_ids and layer_id is not None and readout_key in self.layer_readouts:
            readout = self.layer_readouts[readout_key]
        elif readout_kind != "lm":
            raise ValueError(f"Unsupported Engram readout_kind={readout_kind!r}")
        readout_delta = None
        if readout_kind == "lm" and layer_id is not None and readout_key in self.layer_readout_deltas:
            readout_delta = self.layer_readout_deltas[readout_key]
        readout_delta_value_scale, readout_delta_key_scale = self._layer_readout_delta_scales(
            layer_id, dtype=hidden_states.dtype, device=hidden_states.device
        ) if readout_delta is not None else (0.0, 0.0)
        value_memory = value_memory_base.detach() if engram_detach_value_memory and readout_kind == "lm" else value_memory_base
        value = readout.value_proj(value_memory)
        if readout_delta is not None:
            value = value + readout_delta_value_scale * readout_delta.value_proj(value_memory)
        debug_check_finite("engram.value", value)
        if engram_fixed_half_gate and readout_kind == "lm":
            gate = torch.zeros(value.shape[:-1], device=value.device, dtype=value.dtype)
            gated_value = value * 0.5
        elif engram_static_gate and readout_kind == "lm":
            gate = self._static_gate_logits(layer_id, value.size(0), dtype=value.dtype, device=value.device)
            gated_value = value * torch.sigmoid(gate).unsqueeze(-1)
        else:
            key_memory = memory.detach() if engram_detach_key_memory and readout_kind == "lm" else memory
            key_raw = readout.key_proj(key_memory)
            if readout_delta is not None:
                key_raw = key_raw + readout_delta_key_scale * readout_delta.key_proj(key_memory)
            key = norm(key_raw)
            debug_check_finite("engram.key", key)
            query = norm(hidden_states)
            debug_check_finite("engram.query", query)
            if self.per_head:
                gate = (key * query.unsqueeze(1)).sum(dim=-1) / math.sqrt(hidden_states.size(-1))
            else:
                gate = (key * query).sum(dim=-1) / math.sqrt(hidden_states.size(-1))
            debug_check_finite("engram.dot_gate", gate)
            gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
            debug_check_finite("engram.signed_sqrt_gate", gate)
            gated_value = value * torch.sigmoid(gate).unsqueeze(-1)
        debug_check_finite("engram.gated_value", gated_value)
        layer_sign_aux_scale = self._layer_sign_aux_scale()
        if (
            readout_kind == "lm"
            and layer_id is not None
            and layer_sign_aux_scale > 0
        ):
            layer_signs = self._raw_layer_signs(
                layer_id,
                device=memory_heads.device,
                dtype=memory_heads.dtype,
            )
            aux_memory_heads = memory_heads * layer_signs
            aux_value_memory_heads = (value_memory_heads if value_memory_heads is not None else memory_heads) * layer_signs
            if self.per_head:
                aux_memory = aux_memory_heads
                aux_value_memory_base = aux_value_memory_heads
            else:
                aux_memory = aux_memory_heads.flatten(start_dim=-2)
                aux_value_memory_base = aux_value_memory_heads.flatten(start_dim=-2)
            if self.memory_proj is not None:
                aux_memory = self.memory_proj(aux_memory)
                aux_value_memory_base = self.memory_proj(aux_value_memory_base)
            aux_value_memory = aux_value_memory_base.detach() if engram_detach_value_memory else aux_value_memory_base
            aux_value = readout.value_proj(aux_value_memory)
            if readout_delta is not None:
                aux_value = aux_value + readout_delta_value_scale * readout_delta.value_proj(aux_value_memory)
            debug_check_finite("engram.layer_sign_aux_value", aux_value)
            if engram_fixed_half_gate:
                aux_gate = torch.zeros(aux_value.shape[:-1], device=aux_value.device, dtype=aux_value.dtype)
                aux_gated_value = aux_value * 0.5
            elif engram_static_gate:
                aux_gate = self._static_gate_logits(layer_id, aux_value.size(0), dtype=aux_value.dtype, device=aux_value.device)
                aux_gated_value = aux_value * torch.sigmoid(aux_gate).unsqueeze(-1)
            else:
                aux_key_memory = aux_memory.detach() if engram_detach_key_memory else aux_memory
                aux_key_raw = readout.key_proj(aux_key_memory)
                if readout_delta is not None:
                    aux_key_raw = aux_key_raw + readout_delta_key_scale * readout_delta.key_proj(aux_key_memory)
                aux_key = norm(aux_key_raw)
                debug_check_finite("engram.layer_sign_aux_key", aux_key)
                aux_query = norm(hidden_states)
                if self.per_head:
                    aux_gate = (aux_key * aux_query.unsqueeze(1)).sum(dim=-1) / math.sqrt(hidden_states.size(-1))
                else:
                    aux_gate = (aux_key * aux_query).sum(dim=-1) / math.sqrt(hidden_states.size(-1))
                aux_gate = aux_gate.abs().clamp_min(1e-6).sqrt() * aux_gate.sign()
                debug_check_finite("engram.layer_sign_aux_gate", aux_gate)
                aux_gated_value = aux_value * torch.sigmoid(aux_gate).unsqueeze(-1)
            debug_check_finite("engram.layer_sign_aux_gated_value", aux_gated_value)
            gated_value = (gated_value + layer_sign_aux_scale * aux_gated_value) / math.sqrt(1.0 + layer_sign_aux_scale * layer_sign_aux_scale)
            debug_check_finite("engram.layer_sign_aux_merged_value", gated_value)
        if read_hit_scale is not None:
            if self.per_head:
                scale = read_hit_scale
                if scale.ndim == gated_value.ndim:
                    scale = scale.mean(dim=-1)
                gated_value = gated_value * scale.to(dtype=gated_value.dtype).unsqueeze(-1)
            else:
                scale = read_hit_scale
                while scale.ndim > 1:
                    scale = scale.mean(dim=-1)
                gated_value = gated_value * scale.to(dtype=gated_value.dtype).unsqueeze(-1)
            debug_check_finite("engram.read_hit_scaled_value", gated_value)
        if self.per_head:
            if self.training and engram_head_dropout_current > 0:
                keep_prob = 1.0 - engram_head_dropout_current
                head_mask = torch.empty(
                    gated_value.shape[:2],
                    device=gated_value.device,
                    dtype=torch.float32,
                ).bernoulli_(keep_prob).to(dtype=gated_value.dtype).unsqueeze(-1)
                gated_value = gated_value * (head_mask / keep_prob)
                debug_check_finite("engram.head_dropout_value", gated_value)
            head_mix_weights = self._head_mix_weights(layer_id, dtype=gated_value.dtype, device=gated_value.device)
            if head_mix_weights is not None:
                gated_value = gated_value * head_mix_weights
                debug_check_finite("engram.head_mixed_value", gated_value)
            ngram_read_scale_weights = self._ngram_read_scale_weights(dtype=gated_value.dtype, device=gated_value.device)
            if ngram_read_scale_weights is not None:
                gated_value = gated_value * ngram_read_scale_weights
                debug_check_finite("engram.ngram_read_scaled_value", gated_value)
            if return_banks:
                ngram_heads = self.max_ngram - 1
                bank_value = gated_value.view(gated_value.size(0), ngram_heads, self.num_heads, gated_value.size(-1))
                bank_value = bank_value.sum(dim=1) / math.sqrt(ngram_heads)
                bank_outputs = []
                for bank_idx in range(self.num_heads):
                    bank_output = apply_rom_short_conv(bank_value[:, bank_idx], readout.short_conv_norm, readout.short_conv)
                    bank_outputs.append(bank_output)
                output = torch.stack(bank_outputs, dim=1)
                debug_check_finite("engram.bank_outputs", output)
                if self.training and engram_output_dropout_current > 0:
                    keep_prob = 1.0 - engram_output_dropout_current
                    output_mask = torch.empty(
                        output.shape[:-1],
                        device=output.device,
                        dtype=torch.float32,
                    ).bernoulli_(keep_prob).to(dtype=output.dtype).unsqueeze(-1)
                    output = output * (output_mask / keep_prob)
                    debug_check_finite("engram.bank_output_dropout", output)
                if engram_normalize_readout:
                    output = norm(output)
                    debug_check_finite("engram.normalized_bank_outputs", output)
                output = self._register_debug_backward(f"{debug_label}.bank_output", output)
                record_engram_output_grad_metrics(layer_id, output)
                return output
            merged_value = gated_value.sum(dim=1)
            if head_mix_weights is None:
                merged_value = merged_value / math.sqrt(self.total_hash_heads)
            debug_check_finite("engram.merged_head_value", merged_value)
            output = apply_rom_short_conv(merged_value, readout.short_conv_norm, readout.short_conv)
            record_engram_analysis(input_ids, addresses, gate, value, output, gated_value, layer_id=layer_id, readout_kind=readout_kind)
        else:
            output = apply_rom_short_conv(gated_value, readout.short_conv_norm, readout.short_conv)
            record_engram_analysis(input_ids, addresses, gate, value, output, layer_id=layer_id, readout_kind=readout_kind)
        if self.training and engram_output_dropout_current > 0:
            keep_prob = 1.0 - engram_output_dropout_current
            output_mask = torch.empty(
                output.shape[:-1],
                device=output.device,
                dtype=torch.float32,
            ).bernoulli_(keep_prob).to(dtype=output.dtype).unsqueeze(-1)
            output = output * (output_mask / keep_prob)
            debug_check_finite("engram.output_dropout", output)
        if engram_normalize_readout:
            output = norm(output)
            debug_check_finite("engram.normalized_output", output)
            output = self._apply_shadow_grad(addresses, output)
            output = self._register_debug_backward(f"{debug_label}.output", output)
            record_engram_output_grad_metrics(layer_id, output)
            return output
        output = rom_output_scale * output
        output = self._apply_shadow_grad(addresses, output)
        output = self._register_debug_backward(f"{debug_label}.output", output)
        record_engram_output_grad_metrics(layer_id, output)
        return output


class RomTokenBigramMemory(nn.Module):
    """Compose per-token ROM states into a causal bigram state, then read it."""

    def __init__(self, vocab_size: int, model_dim: int, num_heads: int, key_dim: int, value_dim: int, *, enable_write: bool = False):
        super().__init__()
        if min(vocab_size, model_dim, num_heads, key_dim, value_dim) <= 0:
            raise ValueError("ROM dimensions must be positive")
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.enable_write = enable_write
        self.state = nn.Parameter(torch.zeros(vocab_size, num_heads, key_dim, value_dim, dtype=torch.bfloat16))
        self.q_proj = nn.Linear(model_dim, num_heads * key_dim, bias=False)
        self.compose_gate_proj = nn.Linear(model_dim, num_heads, bias=False)
        self.compose_decay_proj = nn.Linear(model_dim, num_heads, bias=False)
        if rom_engram_gate:
            self.engram_key_proj = nn.Linear(num_heads * value_dim, model_dim, bias=False)
        else:
            self.gate_proj = nn.Linear(model_dim, num_heads, bias=False)
        self.read_dim = num_heads * value_dim
        if rom_read_mlp:
            read_hidden_dim = max(1, int(round(self.read_dim * rom_read_mlp_hidden_mult)))
            self.read_mlp_norm = nn.RMSNorm(self.read_dim)
            self.read_mlp_fc1 = nn.Linear(self.read_dim, read_hidden_dim, bias=False)
            self.read_mlp_fc2 = nn.Linear(read_hidden_dim, self.read_dim, bias=False)
            nn.init.zeros_(self.read_mlp_fc2.weight)
        self.out_proj = nn.Linear(num_heads * value_dim, model_dim, bias=False)
        nn.init.zeros_(self.out_proj.weight)
        if rom_short_conv:
            self.short_conv_norm = nn.RMSNorm(model_dim)
            self.short_conv = nn.Conv1d(model_dim, model_dim, rom_short_conv_kernel, groups=model_dim, bias=False, padding=rom_short_conv_kernel - 1)
            nn.init.zeros_(self.short_conv.weight)
        if enable_write:
            self.k_proj = nn.Linear(model_dim, num_heads * key_dim, bias=False)
            self.v_proj = nn.Linear(model_dim, num_heads * value_dim, bias=False)
            self.beta_proj = nn.Linear(model_dim, num_heads, bias=False)
            self.decay_proj = nn.Linear(model_dim, num_heads, bias=False)
            self.write_gate_proj = nn.Linear(model_dim, num_heads, bias=False)
            self.write_alpha = nn.Parameter(torch.zeros(1))

    def forward(self, token_ids: Tensor, hidden_states: Tensor) -> Tensor:
        assert token_ids.ndim == 1
        assert hidden_states.ndim == 2
        assert token_ids.size(0) == hidden_states.size(0)
        token_ids = token_ids.long()
        current_state = self.state[token_ids]
        previous_state = torch.cat([torch.zeros_like(current_state[:1]), self.state[token_ids[:-1]]], dim=0)
        debug_check_finite("rom_token.current_state", current_state)
        debug_check_finite("rom_token.previous_state", previous_state)
        compose_gate = torch.sigmoid(self.compose_gate_proj(hidden_states)).unsqueeze(-1).unsqueeze(-1)
        compose_decay = torch.exp(-F.softplus(self.compose_decay_proj(hidden_states))).unsqueeze(-1).unsqueeze(-1)
        state = compose_decay * previous_state + compose_gate * current_state
        debug_check_finite("rom_token.composed_state", state)
        q = self.q_proj(hidden_states).view(-1, self.num_heads, self.key_dim)
        q = F.normalize(q, p=2, dim=-1)
        if self.enable_write:
            k = self.k_proj(hidden_states).view(-1, self.num_heads, self.key_dim)
            k = F.normalize(k, p=2, dim=-1)
            v = self.v_proj(hidden_states).view(-1, self.num_heads, self.value_dim)
            beta = torch.sigmoid(self.beta_proj(hidden_states)).unsqueeze(-1)
            decay = torch.exp(-F.softplus(self.decay_proj(hidden_states))).unsqueeze(-1).unsqueeze(-1)
            write_gate = torch.sigmoid(self.write_gate_proj(hidden_states)).unsqueeze(-1).unsqueeze(-1)
            cell = state * decay
            prediction = torch.einsum("thkv,thk->thv", cell, k)
            delta_v = beta * (v - prediction)
            write = k.unsqueeze(-1) * delta_v.unsqueeze(-2)
            state = cell + torch.tanh(self.write_alpha)[0].to(dtype=write.dtype) * write_gate * write
            debug_check_finite("rom_token.write_state", state)
        read = torch.einsum("thk,thkv->thv", q, state)
        debug_check_finite("rom_token.read", read)
        read = read.reshape(hidden_states.size(0), self.num_heads * self.value_dim)
        read = apply_rom_read_mlp(read, getattr(self, "read_mlp_norm", None), getattr(self, "read_mlp_fc1", None), getattr(self, "read_mlp_fc2", None))
        if rom_engram_gate:
            return engram_gate_value(read, hidden_states, self.out_proj, self.engram_key_proj, getattr(self, "short_conv_norm", None), getattr(self, "short_conv", None))
        gate = torch.sigmoid(self.gate_proj(hidden_states)).unsqueeze(-1)
        output = self.out_proj((read.view(hidden_states.size(0), self.num_heads, self.value_dim) * gate).reshape(hidden_states.size(0), self.num_heads * self.value_dim))
        output = apply_rom_short_conv(output, getattr(self, "short_conv_norm", None), getattr(self, "short_conv", None))
        output = apply_rom_ema_smooth(output)
        return rom_output_scale * output


def chunked_softcapped_cross_entropy(x: Tensor, targets: Tensor, mtp_weights: Tensor, lm_head: nn.Module) -> Tensor:
    x_flat = x.view(-1, x.size(-1))
    targets_flat = targets.view(-1)
    chunk_rows = int(os.environ.get("PLAIN_CE_CHUNK_ROWS", "4096"))
    losses = torch.empty(targets_flat.shape, dtype=torch.float32, device=targets_flat.device)
    for start in range(0, x_flat.size(0), chunk_rows):
        end = min(start + chunk_rows, x_flat.size(0))
        logits = lm_head(x_flat[start:end])
        logits = 23 * torch.sigmoid((logits + 5) / 7.5)
        log_probs = F.log_softmax(logits.float(), dim=-1)
        chunk_loss = torch.zeros(end - start, dtype=torch.float32, device=targets_flat.device)
        for k, weight in enumerate(mtp_weights):
            valid = min(end, targets_flat.size(0) - k) - start
            if valid > 0:
                target_k = targets_flat[start + k:start + k + valid]
                chunk_loss[:valid] -= weight.float() * log_probs[:valid].gather(1, target_k[:, None]).squeeze(1)
        losses[start:end] = chunk_loss
    return losses


class CastedLinearT(nn.Module):
    """
    Linear layer with transposed weight storage (in_features, out_features) which
    addresses the slow kernel that was used for gradient accumulation. @chrisjmccormick
    """
    def __init__(self, in_features: int, out_features: int, use_fp8=False, x_s=1.0, w_s=1.0, grad_s=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_fp8 = use_fp8
        self.x_s = x_s
        self.w_s = w_s
        self.grad_s = grad_s

        self.weight = nn.Parameter(torch.empty(in_features, out_features, dtype=torch.bfloat16))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            nn.init.zeros_(self.weight) # @Grad62304977 and others

    def forward(self, x: Tensor):
        if self.use_fp8 and self.training:
            _x = x.flatten(0, -2)
            out = torch.ops.nanogpt.mm_t(_x, self.weight, x_s=self.x_s, w_s=self.w_s, grad_s=self.grad_s)[0]
            return out.reshape(*x.shape[:-1], -1)
        else:
            return x @ self.weight.type_as(x)

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the model

class Yarn(nn.Module):
    def __init__(self, head_dim, max_seq_len, paired=False):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.paired = paired
        self.reset()

    def rotary(self, x_BTHD):
        assert self.factor1.size(0) >= x_BTHD.size(-3)
        factor1, factor2 = (
            self.factor1[None, : x_BTHD.size(-3), None, :],
            self.factor2[None, : x_BTHD.size(-3), None, :],
        )
        x_flip = x_BTHD.view(*x_BTHD.shape[:-1], x_BTHD.shape[-1] // 2, 2).flip(-1).view(x_BTHD.shape)
        return factor1 * x_BTHD + factor2 * x_flip

    def reset(self):
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=self.head_dim//4, dtype=torch.float32, device=device)
        angular_freq = angular_freq.repeat_interleave(2)
        # half-truncate RoPE by @YouJiacheng (w/ base freq tuning)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(self.head_dim//2)])
        t = torch.arange(2*self.max_seq_len, dtype=torch.float32, device=device)
        if not self.paired:
            theta = torch.outer(t, angular_freq)
            self.factor1 = nn.Buffer(
                theta.cos().to(torch.bfloat16), persistent=False
            )
            self.factor2 = nn.Buffer(
                theta.sin().to(torch.bfloat16), persistent=False
            )
        else:
            t_even = 2 * t
            t_odd = t_even + 1
            theta1 = torch.outer(t_even, angular_freq)
            theta2 = torch.outer(t_odd, angular_freq)
            self.factor1 = nn.Buffer(
                torch.cat((theta1.cos(), theta2.cos()), dim=-1).to(torch.bfloat16),
                persistent=False
            )
            self.factor2 = nn.Buffer(
                torch.cat((theta1.sin(), theta2.sin()), dim=-1).to(torch.bfloat16),
                persistent=False
            )
        self.factor2[..., 1::2] *= -1
        self.angular_freq = angular_freq
        # start with 0.1, inspired by 0.12 from @leloykun and learnable scalars used by @brendanh0gan https://x.com/hi_tysam/status/1879693583898591283
        self.attn_scale = 0.1

    def apply(self, old_window: int, new_window: int, alpha: int=1, beta: int=32):
        rotations = old_window * self.angular_freq / (2 * torch.pi)
        scaling_factor = old_window / new_window
        interpolation_weight = torch.clamp((rotations - alpha) / (beta - alpha), 0, 1)
        self.angular_freq *= scaling_factor + interpolation_weight * (1 - scaling_factor)
        t = torch.arange(2*self.max_seq_len, dtype=torch.float32, device=self.angular_freq.device)
        if not self.paired:
            theta = torch.outer(t, self.angular_freq)
            self.factor1.copy_(theta.cos())
            self.factor2.copy_(theta.sin())
        else:
            t_even = 2 * t
            t_odd = t_even + 1
            theta1 = torch.outer(t_even, self.angular_freq)
            theta2 = torch.outer(t_odd, self.angular_freq)
            self.factor1.copy_(torch.cat((theta1.cos(), theta2.cos()), dim=-1))
            self.factor2.copy_(torch.cat((theta1.sin(), theta2.sin()), dim=-1))
        self.factor2[..., 1::2] *= -1
        self.attn_scale *= 0.2 * math.log(new_window / old_window) + 1

@dataclass(slots=True)
class AttnArgs:
    ve: torch.Tensor
    sa_lambdas: torch.Tensor
    block_mask: BlockMask
    yarn: Yarn
    key_offset: bool
    attn_gate_w: torch.Tensor
    ve_gate_w: torch.Tensor
    train_max_seq_len: torch.Tensor

compiled_create_block_mask = torch.compile(create_block_mask, dynamic=False)
compiled_flex_attention = torch.compile(flex_attention, dynamic=False)

def build_flex_block_mask(seqlens: Tensor, bm_size: int, total_tokens: int) -> BlockMask:
    positions = torch.arange(total_tokens, device=seqlens.device, dtype=seqlens.dtype)
    docs = torch.searchsorted(seqlens, positions, right=True) - 1

    def document_causal_window(b, h, q_idx, kv_idx):
        return (q_idx >= kv_idx) & ((q_idx - kv_idx) <= bm_size) & (docs[q_idx] == docs[kv_idx])

    return compiled_create_block_mask(
        document_causal_window,
        B=None,
        H=None,
        Q_LEN=total_tokens,
        KV_LEN=total_tokens,
        device=seqlens.device,
        BLOCK_SIZE=128,
    )

def build_flex_block_masks(seqlens: Tensor, schedule_cfg: "ForwardScheduleConfig", total_tokens: int) -> tuple[BlockMask, BlockMask, BlockMask]:
    return (
        build_flex_block_mask(seqlens, schedule_cfg.ws_short, total_tokens),
        build_flex_block_mask(seqlens, schedule_cfg.ws_long, total_tokens),
        build_flex_block_mask(2 * seqlens, schedule_cfg.ws_short, 2 * total_tokens),
    )

def flex_varlen_window_attention(q: Tensor, k: Tensor, v: Tensor, block_mask: BlockMask, scale: float) -> Tensor:
    q = q.transpose(1, 2).contiguous()
    k = k.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous()
    return compiled_flex_attention(
        q,
        k,
        v,
        block_mask=block_mask,
        scale=scale,
    ).transpose(1, 2)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, head_dim: int, num_heads: int, paired: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dim = dim
        self.hdim = num_heads * head_dim
        self.paired = paired
        assert self.hdim == self.dim, "num_heads * head_dim must equal model_dim"
        # Weights are stored in parameter banks and passed via forward()

    def forward(self, x: Tensor, attn_args: AttnArgs, qkvo_w: Tensor):
        B, T = x.size(0), x.size(1) # batch size, sequence length
        assert B == 1, "varlen sequences requires B == 1"
        assert T % 16 == 0
        # unpack attention args
        yarn = attn_args.yarn
        ve, sa_lambdas, key_offset = attn_args.ve, attn_args.sa_lambdas, attn_args.key_offset
        block_mask = attn_args.block_mask
        # sparse gated attention to enable context based no-op by @classiclarryd
        # only include gates on layers with value embeds used on forward pass
        attn_gate_w, ve_gate_w = attn_args.attn_gate_w, attn_args.ve_gate_w
        train_max_seq_len = attn_args.train_max_seq_len

        q, k, v = F.linear(x, sa_lambdas[0] * qkvo_w[:self.dim * 3].type_as(x)).view(B, T, 3 * self.num_heads, self.head_dim).chunk(3, dim=-2)
        max_len = train_max_seq_len if self.training else (args.val_batch_size // (grad_accum_steps * world_size))

        q, k = norm(q), norm(k) # QK norm @Grad62304977

        if not self.paired:
            q, k = yarn.rotary(q), yarn.rotary(k)

            if key_offset:
                # shift keys forward for the stationary head dims. Enables 1-layer induction.
                k[:, 1:, :, self.head_dim // 2:] = k[:, :-1, :, self.head_dim // 2:]

            if ve is not None:
                # gate pattern g(x[:6] + ve[:6]) by @photomz
                ve_gate_out = 2 * torch.sigmoid(F.linear(torch.cat([x[..., :6], ve[None, ..., :6]], dim=-1), ve_gate_w)).view(B, T, self.num_heads, 1)
                v = v + ve_gate_out * ve.view_as(v) # @ KoszarskyB & @Grad62304977

        else:
            # Paired heads: adjacent heads' queries attend to each other's keys.
            # Two copies of the input stream are interleaved to achieve this, which:
            # - doubles the length of each sequence
            # - halves the effective window size
            q = q.view(B, T, self.num_heads // 2, self.head_dim * 2)
            k = k.view(B, T, self.num_heads // 2, self.head_dim * 2)
            v = v.reshape(B, T * 2, self.num_heads // 2, self.head_dim)

            q, k = yarn.rotary(q), yarn.rotary(k)

            q = q.view(B, T * 2, self.num_heads // 2, self.head_dim)
            k = k.view(B, T * 2, self.num_heads // 2, self.head_dim)

            if ve is not None:
                ve_gate_out = 2 * torch.sigmoid(F.linear(x[..., :12], ve_gate_w)).view(B, T * 2, self.num_heads // 2, 1)
                v = v + ve_gate_out * ve.view_as(v)

            max_len = 2 * max_len

        y = flex_varlen_window_attention(q, k, v, block_mask, yarn.attn_scale)
        y = y.reshape(B, T, self.num_heads, self.head_dim)
        y = y * torch.sigmoid(F.linear(x[..., :12], attn_gate_w)).view(B, T, self.num_heads, 1)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim) # re-assemble all head outputs side by side
        y = F.linear(y, sa_lambdas[1] * qkvo_w[self.dim * 3:].type_as(y))  # sa_lambdas[1] pre-multiplied to O @shenberg
        return y


# -----------------------------------------------------------------------------
# The main model

def next_multiple_of_n(v: float | int, *, n: int):
    return math.ceil(v / n) * n

@dataclass(slots=True)
class ForwardScheduleConfig:
    mtp_weights: torch.Tensor
    ws_short: int
    ws_long: int
    train_max_seq_len: int

class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, num_heads: int, head_dim: int, model_dim: int, max_seq_len: int):
        super().__init__()
        if rom_layers:
            bad_layers = [i for i in rom_layers if i < 0 or i >= num_layers]
            if bad_layers:
                raise ValueError(f"ROM_LAYERS entries must be in [0, {num_layers - 1}], got {bad_layers}")
            if len(set(rom_layers)) != len(rom_layers):
                raise ValueError(f"ROM_LAYERS must not contain duplicates, got {rom_layers}")
            self.bigram_layer_ids = rom_layers
        else:
            if rom_layer_only < -1 or rom_layer_only >= num_layers:
                raise ValueError(f"ROM_LAYER_ONLY must be -1 or in [0, {num_layers - 1}]")
            self.bigram_layer_ids = tuple(range(num_layers)) if rom_layer_only < 0 else (rom_layer_only,)
        self.num_layers = num_layers
        self.last_cache_recon_loss = None
        self.vocab_size = next_multiple_of_n(vocab_size, n=128)

        self.smear_gate = nn.Linear(12, 1, bias=False)
        nn.init.zeros_(self.smear_gate.weight)

        self.skip_gate = nn.Linear(12, 1, bias=False)
        nn.init.zeros_(self.skip_gate.weight)

        # token value embeddings by @KoszarskyB - inspired by @Grad62304977's value residual implementation following https://arxiv.org/abs/2410.17897
        # value embedding code simplification inspired by @ragulpr https://github.com/KellerJordan/modded-nanogpt/pull/78
        # spherical gaussian init by @photomz
        self.value_embeds = nn.Parameter(0.01 * torch.randn(5 * self.vocab_size, model_dim, dtype=torch.bfloat16))

        # parameter banks for attention and value embedding gate weights
        self.attn_gate_bank = nn.Parameter(torch.zeros(10, num_heads, 12)) # 10 layers
        self.ve_gate_bank = nn.Parameter(torch.zeros(5, num_heads, 12)) # 5 unique gates
        self.gate_filler_nones = [None] * (num_layers - 6)

        # -----------------------------------
        # Parameter banks for sharded optimization, by @chrisjmccormick

        # Identify which layers have attention/MLP
        # Attention is skipped in layer 6 by @YouJiacheng
        num_attn_layers = num_layers - 1
        # All layers have MLP (At 11 layers--dropped first layer @EmelyanenkoK)
        num_mlp_layers = num_layers

        hdim = num_heads * head_dim
        mlp_hdim = 4 * model_dim

        # QK bank: per-head-pair Muon groups for Q, K weights
        # Each pair of adjacent heads gets its own independent polar express orthogonalization
        self._num_attn_layers = num_attn_layers
        num_qk_groups = num_attn_layers * 2 * (num_heads // 2)  # 10 * 2 * 3 = 60
        self._num_qk_groups = num_qk_groups
        num_qk_padded = next_multiple_of_n(num_qk_groups, n=world_size)  # 64
        self.qk_bank = nn.Parameter(torch.empty(num_qk_padded, head_dim * 2, model_dim))
        self.qk_bank.reshape = (num_qk_padded, head_dim * 2, model_dim)

        # VO bank: per-layer Muon groups for V and O weights
        num_vo_real = num_attn_layers * 2  # 20
        num_vo_padded = next_multiple_of_n(num_vo_real, n=world_size)  # 24
        self.vo_bank = nn.Parameter(torch.empty(num_vo_padded, hdim, hdim))
        self.vo_bank.reshape = (num_vo_padded, hdim, hdim)

        # MLP bank: stores c_fc and c_proj for all MLP layers
        # We add 1 padding layer (index 11) to get 12*2=24 matrices for even distribution across 8 GPUs
        self.mlp_bank = nn.Parameter(torch.empty(12, 2, mlp_hdim, model_dim))  # (12, 2, 3072, 768)
        self.mlp_bank.reshape = (24, mlp_hdim, model_dim)  # Shape for sharding: (24, 3072, 768)

        # improved init scale by @YouJiacheng and @srashedll
        std = 0.5 * model_dim ** -0.5
        bound = (3 ** 0.5) * std
        with torch.no_grad():
            self.qk_bank[:num_qk_groups].uniform_(-bound, bound)
            self.qk_bank[num_qk_groups:].zero_()
            self.vo_bank[:num_vo_real].uniform_(-bound, bound)
            self.vo_bank[num_vo_real:].zero_()
            self.mlp_bank[:, 0, :, :].uniform_(-bound, bound)  # c_fc
            self.mlp_bank[:, 1, :, :].zero_()  # c_proj - zero init suggested by @Grad62304977

        # Attention modules (no learned params -- weights come from qk_bank/vo_bank)
        self.paired_head_layers = [0, 2, 5, 9]
        self.attn = CausalSelfAttention(model_dim, head_dim, num_heads, paired=False)
        self.attn_paired = CausalSelfAttention(model_dim, head_dim, num_heads, paired=True)
        self.yarn = Yarn(head_dim, max_seq_len)
        self.yarn_paired_head = Yarn(head_dim, max_seq_len, paired=True)
        # there are only 50257 unique GPT-2 tokens; we extend to nearest multiple of 128 for efficiency.
        # suggested to me by @Grad62304977. this originates from Karpathy's experiments.
        use_fp8 = not os.environ.get("DISABLE_FP8", False)
        # Transposed weight storage for faster gradient accumulation
        self.lm_head = CastedLinearT(model_dim, self.vocab_size, use_fp8=use_fp8, x_s=100/448, w_s=1.6/448, grad_s=grad_scale * 0.75/448)

        nn.init.normal_(self.lm_head.weight, mean=0, std=0.005)

        self.embed = nn.Embedding(self.vocab_size, model_dim)
        with torch.no_grad():
            self.embed.weight.copy_(self.lm_head.weight.T)

        if rom_bigram and not rom_token and not rom_single_token and rom_table_nonemb_mult > 0:
            state_width = rom_heads * rom_key_dim * rom_value_dim
            future_scalar_params = num_layers * 2 + num_layers + num_layers + num_layers * 2 + (num_layers * 2 + 3)
            non_embedding_params = sum(
                p.numel()
                for name, p in self.named_parameters()
                if not name.startswith(("embed.", "lm_head.", "value_embeds", "bigram_embed."))
            ) + future_scalar_params
            target_state_params = math.ceil(non_embedding_params * rom_table_nonemb_mult)
            args.bigram_vocab_size = max(2, next_multiple_of_n(math.ceil(target_state_params / state_width), n=128))
            self.rom_non_embedding_params = non_embedding_params
            self.rom_target_state_params = args.bigram_vocab_size * state_width

        if engram_bigram:
            resolved_engram_dim = engram_dim if engram_dim > 0 else model_dim // 2
            engram_partition_group_ids: tuple[int, ...] = ()
            if engram_layer_partitions and engram_layer_partition_groups > 0:
                partition_group_count = min(engram_layer_partition_groups, len(self.bigram_layer_ids))
                engram_partition_group_ids = tuple(
                    min((idx * partition_group_count) // len(self.bigram_layer_ids), partition_group_count - 1)
                    for idx in range(len(self.bigram_layer_ids))
                )
            self.bigram_embed = EngramBigramMemory(
                args.bigram_vocab_size,
                model_dim,
                resolved_engram_dim,
                engram_heads,
                engram_max_ngram,
                seed=engram_hash_seed + 10007 * self.bigram_layer_ids[0],
                pad_id=engram_pad_id,
                token_vocab_size=vocab_size,
                layer_hash_ids=self.bigram_layer_ids if engram_layer_hashes else (),
                layer_readout_ids=self.bigram_layer_ids if engram_layer_readouts else (),
                layer_readout_delta_ids=self.bigram_layer_ids if engram_layer_readout_delta else (),
                layer_partition_ids=self.bigram_layer_ids if engram_layer_partitions else (),
                layer_partition_group_ids=engram_partition_group_ids,
            )
        elif rom_single_token:
            self.bigram_embed = RomBigramMemory(self.vocab_size, model_dim, rom_heads, rom_key_dim, rom_value_dim, enable_write=rom_write)
        elif rom_token:
            self.bigram_embed = RomTokenBigramMemory(self.vocab_size, model_dim, rom_heads, rom_key_dim, rom_value_dim, enable_write=rom_write)
        elif rom_bigram:
            self.bigram_embed = RomBigramMemory(args.bigram_vocab_size, model_dim, rom_heads, rom_key_dim, rom_value_dim, enable_write=rom_write)
        else:
            self.bigram_embed = nn.Embedding(args.bigram_vocab_size, model_dim)
            nn.init.zeros_(self.bigram_embed.weight)

        self.post_lambdas = nn.Parameter(torch.ones(num_layers, 2))

        # Per-layer injection coefficients for x0 and bigram
        self.x0_lambdas = nn.Parameter(torch.zeros(num_layers))
        self.bigram_lambdas = nn.Parameter(0.05 * torch.ones(num_layers))

        # Per-sublayer residual scaling: [num_layers, 2] where [:,0]=attn, [:,1]=mlp
        # sqrt(1.1) per sublayer so cumulative per-layer scaling is 1.1
        self.resid_lambdas = nn.Parameter(torch.full((num_layers, 2), 1.1**0.5))
        if engram_mhc:
            if engram_mhc_identity:
                self.mhc_logit = None
                self.mhc_router = None
            else:
                mhc_init_logit = math.log(engram_mhc_init / (1.0 - engram_mhc_init))
                self.mhc_logit = nn.Parameter(torch.full((num_layers,), mhc_init_logit))
                self.mhc_router = nn.Linear(12, num_layers, bias=False) if engram_mhc_dynamic else None
                if self.mhc_router is not None:
                    nn.init.zeros_(self.mhc_router.weight)
            self.mhc_identity_router = nn.Linear(
                12 * engram_mhc_streams,
                2 * num_layers * engram_mhc_streams,
                bias=True,
            ) if engram_mhc_identity else None
            if self.mhc_identity_router is not None:
                nn.init.zeros_(self.mhc_identity_router.weight)
                with torch.no_grad():
                    bias = self.mhc_identity_router.bias.view(num_layers, 2, engram_mhc_streams)
                    bias[:, 0, :] = -4.0
                    bias[:, 0, 0] = 4.0
                    bias[:, 1, :] = -6.0
                    bias[:, 1, 0] = 0.0
        else:
            self.mhc_logit = None
            self.mhc_router = None
            self.mhc_identity_router = None
        self.engram_attnres_query = nn.Parameter(torch.zeros(num_layers, model_dim, dtype=torch.bfloat16)) if engram_attnres_merge else None
        self.engram_direct_resid = nn.Parameter(torch.full((num_layers,), engram_attnres_direct_init, dtype=torch.float32)) if engram_attnres_direct_residual else None
        if self.engram_direct_resid is not None and engram_attnres_direct_layers:
            bad_layers = [i for i in engram_attnres_direct_layers if i < 0 or i >= num_layers]
            if bad_layers:
                raise ValueError(f"ENGRAM_ATTNRES_DIRECT_LAYERS entries must be in [0, {num_layers - 1}], got {bad_layers}")
        self.engram_attnres_layer_gain_log = nn.Parameter(torch.full((num_layers,), engram_attnres_layer_gain_init, dtype=torch.float32)) if engram_attnres_layer_gain else None
        self.engram_attnres_extra_bias = nn.Parameter(torch.full((num_layers,), engram_attnres_extra_bias_init, dtype=torch.float32)) if engram_attnres_extra_target_layer >= 0 else None
        self.final_smear_mtp_lambda = nn.Parameter(torch.tensor([final_smear_mtp_init], dtype=torch.float32)) if final_smear_mtp else None
        self.last_engram_attnres_stats: dict[int, Tensor] = {}
        self.last_engram_attnres_extra_stats: dict[int, Tensor] = {}
        self.last_engram_attnres_cos_stats: dict[int, Tensor] = {}
        self.last_engram_attnres_rms_ratio_stats: dict[int, Tensor] = {}
        if engram_cache_readout and engram_cache_learned_cfg:
            self.cache_cfg_lambdas = nn.Parameter(torch.full((num_layers,), engram_cache_cfg_scale))
        else:
            self.cache_cfg_lambdas = None

        pad = (-num_layers * 2 - 3) % dist.get_world_size()
        self.scalars = nn.Parameter(
            torch.cat(
                [
                    *[torch.tensor([0.5, 1.0]) for _ in range(num_layers)],  # SA lambdas
                    torch.zeros(1), # smear_lambda
                    0.5*torch.ones(1), # backout_lambda
                    -1.5 * torch.ones(1),  # skip_lambda -> σ(-1.5) ≈ 0.18
                    torch.ones(pad),
                ]
            )
        )
        # Auto-label parameters
        for name, param in self.named_parameters():
            param.label = name.replace('.weight', '')

    def bigram_memory_param(self) -> nn.Parameter:
        if engram_bigram:
            if self.bigram_embed.offload:
                return self.bigram_embed.first_readout_param()
            return self.bigram_embed.embedding.weight
        if rom_bigram:
            state = self.bigram_embed.state
            return state.weight if isinstance(state, nn.Embedding) else state
        return self.bigram_embed.weight

    def _engram_mhc_mix(self, layer_id: int, main: Tensor, aux: Tensor, memory: Tensor) -> tuple[Tensor, Tensor]:
        logit = self.mhc_logit[layer_id]
        if self.mhc_router is not None:
            route = self.mhc_router(norm(main)[..., :self.mhc_router.weight.size(-1)])[..., layer_id]
            logit = logit + route
        a = torch.sigmoid(logit).to(dtype=main.dtype).unsqueeze(-1)
        if engram_mhc_streams > 2:
            if aux.ndim == main.ndim:
                aux = aux.unsqueeze(0).expand(engram_mhc_streams - 1, *aux.shape).clone()
            memory_stream = aux.clone()
            memory_stream[0] = memory_stream[0] + memory
            streams = torch.cat((main.unsqueeze(0), memory_stream), dim=0)
            stream_sum = streams.sum(dim=0, keepdim=True)
            mixed = (1.0 - a) * streams + (a / float(engram_mhc_streams - 1)) * (stream_sum - streams)
            return mixed[0], mixed[1:]
        memory_stream = aux + memory
        mixed_main = main + a * (memory_stream - main)
        mixed_aux = memory_stream + a * (main - memory_stream)
        return mixed_main, mixed_aux

    def _engram_identity_hc_read(self, layer_id: int, main: Tensor, aux: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if aux.ndim == main.ndim:
            aux = aux.unsqueeze(0).expand(engram_mhc_streams - 1, *aux.shape).clone()
        streams = torch.cat((main.unsqueeze(0), aux), dim=0)
        stream_features = norm(streams)[..., :12].permute(1, 2, 0, 3).flatten(-2)
        router_dtype = main.dtype
        stream_features = stream_features.to(dtype=router_dtype)
        logits = F.linear(
            stream_features,
            self.mhc_identity_router.weight.to(dtype=router_dtype),
            self.mhc_identity_router.bias.to(dtype=router_dtype),
        ).view(
            *stream_features.shape[:-1],
            self.num_layers,
            2,
            engram_mhc_streams,
        )[..., layer_id, :, :]
        h_pre = F.softmax(logits[..., 0, :], dim=-1).to(dtype=main.dtype)
        h_post = F.softmax(logits[..., 1, :], dim=-1).to(dtype=main.dtype)
        read = (streams.permute(1, 2, 0, 3) * h_pre.unsqueeze(-1)).sum(dim=-2)
        return read, aux, h_post

    def _engram_identity_hc_write(self, main: Tensor, aux: Tensor, memory: Tensor, h_post: Tensor) -> tuple[Tensor, Tensor]:
        update = h_post.unsqueeze(-1) * memory.unsqueeze(-2)
        main = main + update[..., 0, :]
        aux = aux + update[..., 1:, :].permute(2, 0, 1, 3)
        return main, aux

    def _engram_attnres_merge(self, layer_id: int, main: Tensor, memory: Tensor, extra: Tensor | None = None) -> Tensor:
        query = self.engram_attnres_query[layer_id].to(dtype=main.dtype)
        if engram_attnres_delta:
            zero_delta = torch.zeros_like(memory)
            if extra is None:
                sources = torch.stack((zero_delta, memory), dim=-2)
                extra_value = None
            else:
                extra_value = extra - main
                sources = torch.stack((zero_delta, memory, extra_value), dim=-2)
        elif extra is None:
            sources = torch.stack((main, memory), dim=-2)
            extra_value = None
        else:
            sources = torch.stack((main, memory, extra), dim=-2)
            extra_value = extra
        keys = norm(sources)
        logits = (keys * query.view(*([1] * (keys.ndim - 2)), 1, -1)).sum(dim=-1) / math.sqrt(main.size(-1))
        if extra is not None and self.engram_attnres_extra_bias is not None:
            logits = logits.clone()
            logits[..., 2] = logits[..., 2] + self.engram_attnres_extra_bias[layer_id].to(dtype=logits.dtype)
        weights = F.softmax(logits, dim=-1).to(dtype=main.dtype)
        memory_weight = weights[..., 1:2] * engram_attnres_merge_gain_current
        if self.engram_attnres_layer_gain_log is not None:
            layer_gain = self.engram_attnres_layer_gain_log[layer_id].float().exp().to(dtype=main.dtype)
            memory_weight = memory_weight * layer_gain
        if not is_torch_compiling():
            self.last_engram_attnres_stats[layer_id] = weights[..., 1].detach()
            if engram_attnres_metrics:
                main_float = main.detach().float()
                memory_float = memory.detach().float()
                memory_cos = F.cosine_similarity(main_float, memory_float, dim=-1, eps=1e-12)
                self.last_engram_attnres_cos_stats[layer_id] = memory_cos
                main_rms = main_float.square().mean(dim=-1).sqrt()
                memory_rms = memory_float.square().mean(dim=-1).sqrt()
                memory_rms_ratio = memory_rms / main_rms.clamp_min(1e-12)
                self.last_engram_attnres_rms_ratio_stats[layer_id] = memory_rms_ratio
                record_engram_attnres_analysis(layer_id, memory_weight, memory_cos, memory_rms_ratio)
            if extra is not None:
                self.last_engram_attnres_extra_stats[layer_id] = weights[..., 2].detach()
        out = main + memory_weight * memory
        if self.engram_direct_resid is not None and (not engram_attnres_direct_layers or layer_id in engram_attnres_direct_layers):
            out = out + self.engram_direct_resid[layer_id].to(dtype=main.dtype) * memory
        if extra_value is not None:
            extra_scale = engram_attnres_extra_scale
            if engram_attnres_extra_scale_schedule_steps > 0:
                schedule_step = float(rom_debug_nan_current_step - engram_attnres_extra_scale_schedule_start)
                progress = min(max(schedule_step / float(engram_attnres_extra_scale_schedule_steps), 0.0), 1.0)
                extra_scale = engram_attnres_extra_scale + progress * (engram_attnres_extra_scale_final - engram_attnres_extra_scale)
            out = out + weights[..., 2:3] * engram_attnres_merge_gain_current * extra_scale * extra_value
        return out

    def _engram_bank_attnres_merge(self, layer_id: int, main: Tensor, memories: Tensor) -> Tensor:
        query = self.engram_attnres_query[layer_id].to(dtype=main.dtype)
        if engram_attnres_delta:
            sources = torch.cat((torch.zeros_like(main).unsqueeze(-2), memories), dim=-2)
        else:
            sources = torch.cat((main.unsqueeze(-2), memories), dim=-2)
        keys = norm(sources)
        logits = (keys * query.view(*([1] * (keys.ndim - 2)), 1, -1)).sum(dim=-1) / math.sqrt(main.size(-1))
        weights = F.softmax(logits, dim=-1).to(dtype=main.dtype)
        memory_weights = weights[..., 1:] * engram_attnres_merge_gain_current
        if self.engram_attnres_layer_gain_log is not None:
            layer_gain = self.engram_attnres_layer_gain_log[layer_id].float().exp().to(dtype=main.dtype)
            memory_weights = memory_weights * layer_gain
        if not is_torch_compiling():
            self.last_engram_attnres_stats[layer_id] = weights[..., 1:].sum(dim=-1).detach()
            self.last_engram_attnres_extra_stats[layer_id] = weights[..., 1:].amax(dim=-1).detach()
            if engram_attnres_metrics:
                main_float = main.detach().float().unsqueeze(-2)
                memories_float = memories.detach().float()
                memory_cos = F.cosine_similarity(main_float, memories_float, dim=-1, eps=1e-12).mean(dim=-1)
                self.last_engram_attnres_cos_stats[layer_id] = memory_cos
                main_rms = main_float.square().mean(dim=-1).sqrt()
                memory_rms = memories_float.square().mean(dim=-1).sqrt()
                memory_rms_ratio = (memory_rms / main_rms.clamp_min(1e-12)).mean(dim=-1)
                self.last_engram_attnres_rms_ratio_stats[layer_id] = memory_rms_ratio
        return main + (memory_weights.unsqueeze(-1) * memories).sum(dim=-2)

    def _dense_attn_mlp_layer(
        self,
        x: Tensor,
        attn_in: Tensor,
        attn_args: AttnArgs,
        qkvo_w: Tensor,
        c_fc: Tensor,
        c_proj: Tensor,
        resid_lambda_attn: Tensor,
        post_lambda_attn: Tensor,
        x0_inject: Tensor,
        resid_lambda_mlp: Tensor,
        post_lambda_mlp: Tensor,
        paired: bool,
    ) -> Tensor:
        attn = self.attn_paired if paired else self.attn
        attn_out = attn(norm(attn_in), attn_args, qkvo_w)
        x = resid_lambda_attn * x + post_lambda_attn * attn_out + x0_inject
        mlp_in = norm(x)
        if self.training and not plain_mlp_train:
            mlp_out = ReLUSqrdMLP(mlp_in, c_fc, c_proj)
        else:
            mlp_out = F.relu(F.linear(mlp_in, c_fc)).square() @ c_proj
        return resid_lambda_mlp * x + post_lambda_mlp * mlp_out

    def _dense_mlp_layer(
        self,
        x: Tensor,
        c_fc: Tensor,
        c_proj: Tensor,
        resid_lambda_mlp: Tensor,
        post_lambda_mlp: Tensor,
    ) -> Tensor:
        mlp_in = norm(x)
        if self.training and not plain_mlp_train:
            mlp_out = ReLUSqrdMLP(mlp_in, c_fc, c_proj)
        else:
            mlp_out = F.relu(F.linear(mlp_in, c_fc)).square() @ c_proj
        return resid_lambda_mlp * x + post_lambda_mlp * mlp_out

    def forward(self, input_seq: Tensor, target_seq: Tensor, seqlens: Tensor, bigram_input_seq: Tensor, schedule_cfg: ForwardScheduleConfig, block_masks: tuple[BlockMask, BlockMask, BlockMask]):
        assert input_seq.ndim == 1
        if engram_cache_recon and not is_torch_compiling():
            self.last_cache_recon_loss = None
        if engram_attnres_merge and not is_torch_compiling():
            self.last_engram_attnres_stats = {}
            self.last_engram_attnres_extra_stats = {}
            self.last_engram_attnres_cos_stats = {}
            self.last_engram_attnres_rms_ratio_stats = {}

        # ---- Schedule and layer topology ----
        mtp_weights, train_max_seq_len = schedule_cfg.mtp_weights, schedule_cfg.train_max_seq_len
        ws_short, ws_long = schedule_cfg.ws_short, schedule_cfg.ws_long
        block_mask_short, block_mask_long, block_mask_paired_short = block_masks
        # set block masks and key shift
        bm_sizes = [ws_short, ws_short, ws_short, ws_long, ws_short, ws_short, None, ws_short, ws_short, ws_short, ws_long]
        assert len(bm_sizes) == self.num_layers
        key_offset = [b==ws_long for b in bm_sizes] # apply partial key offset to long windows

        # ---- Unbind parameters (avoid select_backward kernels) ----
        sa_lambdas = self.scalars[: 2 * self.num_layers].view(-1, 2)
        smear_lambda = self.scalars[2 * self.num_layers]
        backout_lambda = self.scalars[2 * self.num_layers + 1]
        skip_lambda = self.scalars[2 * self.num_layers + 2]
        resid_lambdas_attn = self.resid_lambdas[:, 0].bfloat16().unbind(0)
        resid_lambdas_mlp  = self.resid_lambdas[:, 1].bfloat16().unbind(0)
        post_lambdas_attn = self.post_lambdas[:, 0].bfloat16().unbind(0)
        post_lambdas_mlp  = self.post_lambdas[:, 1].bfloat16().unbind(0)
        x0_lambdas = self.x0_lambdas.bfloat16().unbind(0)
        bigram_lambdas = self.bigram_lambdas.bfloat16().unbind(0)
        use_bigram_layer = [i in self.bigram_layer_ids for i in range(self.num_layers)]
        ag = self.attn_gate_bank.unbind(0)
        veg = self.ve_gate_bank.unbind(0)
        attn_gates = [*ag[:6], None, *ag[6:]]
        ve_gates = [None, veg[0], veg[1], *self.gate_filler_nones, veg[2], veg[3], veg[4]]
        assert len(attn_gates) == self.num_layers
        assert len(ve_gates) == self.num_layers
        qk_all = self.qk_bank[:self._num_qk_groups].view(self._num_attn_layers, -1, self.qk_bank.shape[-1])
        vo_flat = self.vo_bank[:self._num_attn_layers * 2].view(self._num_attn_layers, 2, *self.vo_bank.shape[1:]).flatten(1, 2)
        attn_weights = torch.cat([qk_all, vo_flat], dim=1).unbind(0)
        mlp_all = self.mlp_bank.flatten(0, 1).unbind(0)  # 24 tensors of [mlp_hdim, dim]
        mlp_fcs = mlp_all[0::2]    # even indices: c_fc
        mlp_projs = mlp_all[1::2]  # odd indices: c_proj

        if engram_bigram and not engram_layer_hashes and not engram_latent:
            self.bigram_embed.prefetch(input_seq)

        # ---- Embeddings and input preparation ----
        x = self.embed(input_seq) # embed is synced from lm_head during tied phase by optimizer
        
        # Value embeddings - always computed (not precomputed)
        ve = self.value_embeds.view(5, self.vocab_size, -1)[:, input_seq]
        # Shifted .01 ... 234 structure on token value embeddings by @photomz
        ve = [None, ve[0], ve[1], *self.gate_filler_nones, ve[2], ve[3], ve[4]]
        assert len(ve) == self.num_layers

        # smear token embed forward 1 position @classiclarryd
        smear_gate_out = smear_lambda * torch.sigmoid(self.smear_gate(x[1:, :self.smear_gate.weight.size(-1)]))
        x = torch.cat([x[:1], x[1:] + smear_gate_out * x[:-1]])
        x = x0 = norm(x[None])
        if engram_bigram or rom_single_token:
            x0_bigram = None
        elif rom_token:
            x0_bigram = self.bigram_embed(input_seq, x0.squeeze(0))[None]
        elif rom_bigram:
            x0_bigram = self.bigram_embed(bigram_input_seq, x0.squeeze(0))[None]
        else:
            x0_bigram = self.bigram_embed(bigram_input_seq)[None]

        # Initialize residual stream with pre-layer-0 bigram injection
        if not engram_bigram and use_bigram_layer[0]:
            x = x + x0_bigram * bigram_lambdas[0]

        # Precompute x0/bigram injection (added to attention output each layer)
        # Layer 0: bigram already injected above, so only x0 component
        if engram_bigram or rom_single_token:
            x0_inject = tuple(x0 * x0_lambdas[i] for i in range(self.num_layers))
        else:
            x0_inject = (x0 * x0_lambdas[0],) + tuple(
                x0 * x0_lambdas[i] + (x0_bigram * bigram_lambdas[i] if use_bigram_layer[i] else 0)
                for i in range(1, self.num_layers)
            )
        skip_gate_out = torch.sigmoid(skip_lambda) * 2 * torch.sigmoid(self.skip_gate(x0[..., :self.skip_gate.weight.size(-1)]))
        if engram_mhc:
            if engram_mhc_streams > 2:
                mhc_aux = x.unsqueeze(0).expand(engram_mhc_streams - 1, *x.shape).clone()
            else:
                mhc_aux = x
        else:
            mhc_aux = None
        
        # ---- Transformer layers ----
        x_backout = None
        skip_connection = None
        cache_recon_source = None
        cache_recon_read = None
        cache_recon_loss = None
        engram_attnres_extra_source = None
        for i in range(self.num_layers):
            mhc_layer_start = x if mhc_aux is not None and engram_mhc_delta and not engram_mhc_identity else None
            if engram_cache_recon and i == engram_cache_recon_target_layer and cache_recon_source is not None and cache_recon_read is not None:
                cache_recon_target = x.detach() - cache_recon_source
                if engram_cache_recon_mode == "cosine":
                    cache_recon_loss = 1.0 - (
                        F.normalize(cache_recon_read.float(), dim=-1) *
                        F.normalize(cache_recon_target.float(), dim=-1)
                    ).sum(dim=-1).mean()
                elif engram_cache_recon_mode == "direction_mse":
                    cache_recon_loss = F.mse_loss(
                        F.normalize(cache_recon_read.float(), dim=-1),
                        F.normalize(cache_recon_target.float(), dim=-1),
                    )
                else:
                    cache_recon_loss = F.mse_loss(cache_recon_read.float(), cache_recon_target.float())
                self.last_cache_recon_loss = cache_recon_loss.detach()
                debug_check_finite("gpt.cache_recon_loss", cache_recon_loss)
            yarn = self.yarn_paired_head if i in self.paired_head_layers else self.yarn
            block_mask = block_mask_paired_short if i in self.paired_head_layers else (block_mask_long if bm_sizes[i] == ws_long else block_mask_short)
            attn_args = AttnArgs(
                ve=ve[i],
                sa_lambdas=sa_lambdas[i],
                block_mask=block_mask,
                yarn=yarn,
                key_offset=key_offset[i],
                attn_gate_w=attn_gates[i],
                ve_gate_w=ve_gates[i],
                train_max_seq_len=train_max_seq_len
            )
            # Select weights from banks
            attn_idx = i - (i > 6) if i != 6 else None
            qkvo_w = attn_weights[attn_idx] if attn_idx is not None else None
            c_fc = mlp_fcs[i]
            c_proj = mlp_projs[i]

            # Select attention variant for this layer
            attn = self.attn_paired if i in self.paired_head_layers else self.attn

            if engram_bigram and use_bigram_layer[i]:
                engram_layer_id = i if (engram_layer_hashes or engram_layer_readouts or engram_layer_readout_delta or engram_layer_partitions or engram_layer_signs or engram_layer_row_signs) else None
                if engram_cache_recon and i == engram_cache_recon_source_layer:
                    cache_recon_source = x.detach()
                if engram_mhc_identity:
                    engram_read, mhc_aux, mhc_h_post = self._engram_identity_hc_read(i, x, mhc_aux)
                    debug_check_finite(f"gpt.layer{i}.engram_identity_read", engram_read)
                    engram_query = engram_read
                else:
                    mhc_h_post = None
                    engram_query = x
                if engram_bank_attnres:
                    engram_out = self.bigram_embed(
                        input_seq,
                        norm(engram_query.squeeze(0)),
                        layer_id=engram_layer_id,
                        return_banks=True,
                    )[None]
                    debug_check_finite(f"gpt.layer{i}.engram_bank_out", engram_out)
                else:
                    engram_out = self.bigram_embed(input_seq, norm(engram_query.squeeze(0)), layer_id=engram_layer_id)[None]
                    debug_check_finite(f"gpt.layer{i}.engram_out", engram_out)
                if engram_cache_recon and i == engram_cache_recon_source_layer:
                    if engram_bank_attnres:
                        raise ValueError("ENGRAM_BANK_ATTNRES is not implemented with ENGRAM_CACHE_RECON")
                    if engram_cache_readout:
                        cache_read = self.bigram_embed(input_seq, norm(engram_query.squeeze(0)), layer_id=engram_layer_id, readout_kind="cache")[None]
                        debug_check_finite(f"gpt.layer{i}.cache_read", cache_read)
                        cache_recon_read = cache_read
                        if self.cache_cfg_lambdas is not None or engram_cache_cfg_scale != 0.0:
                            if self.cache_cfg_lambdas is not None:
                                cache_cfg_scale = self.cache_cfg_lambdas[i].to(dtype=cache_read.dtype)
                            else:
                                cache_cfg_scale = torch.as_tensor(
                                    engram_cache_cfg_scale,
                                    dtype=cache_read.dtype,
                                    device=cache_read.device,
                                )
                            engram_out = engram_out + cache_read * cache_cfg_scale
                            debug_check_finite(f"gpt.layer{i}.cfg_engram_out", engram_out)
                    else:
                        cache_recon_read = engram_out
                if engram_mhc_identity:
                    x, mhc_aux = self._engram_identity_hc_write(x, mhc_aux, engram_out, mhc_h_post)
                    debug_check_finite(f"gpt.layer{i}.post_engram_identity_hc_aux", mhc_aux)
                elif mhc_aux is not None:
                    x, mhc_aux = self._engram_mhc_mix(i, x, mhc_aux, engram_out)
                    debug_check_finite(f"gpt.layer{i}.post_engram_mhc_aux", mhc_aux)
                elif engram_bank_attnres:
                    x = self._engram_bank_attnres_merge(i, x, engram_out)
                elif engram_attnres_merge:
                    extra = engram_attnres_extra_source if i == engram_attnres_extra_target_layer else None
                    x = self._engram_attnres_merge(i, x, engram_out, extra)
                else:
                    x = x + engram_out
                debug_check_finite(f"gpt.layer{i}.post_engram_resid", x)
            elif rom_single_token and use_bigram_layer[i]:
                rom_out = self.bigram_embed(input_seq, norm(x.squeeze(0)))[None]
                debug_check_finite(f"gpt.layer{i}.rom_single_token_out", rom_out)
                if engram_attnres_merge:
                    x = self._engram_attnres_merge(i, x, rom_out)
                else:
                    x = x + rom_out
                debug_check_finite(f"gpt.layer{i}.post_rom_single_token_resid", x)

            # Skip attention on layer 6 @YouJiacheng. Instead pull skip connection from prior long window
            if i == 6:
                x = x + skip_gate_out * skip_connection
                if compile_dense_layer_body:
                    x = self.compiled_dense_mlp_layer(x, c_fc, c_proj, resid_lambdas_mlp[i], post_lambdas_mlp[i])
                    debug_check_finite(f"gpt.layer{i}.post_mlp_resid", x)
                    if i == engram_attnres_extra_source_layer:
                        engram_attnres_extra_source = x
                    if mhc_aux is not None and mhc_layer_start is not None:
                        layer_delta = x - mhc_layer_start
                        if engram_mhc_streams > 2:
                            layer_delta = layer_delta.unsqueeze(0)
                        mhc_aux = mhc_aux + layer_delta
                        debug_check_finite(f"gpt.layer{i}.post_mlp_mhc_aux", mhc_aux)
                    if i == 3:
                        skip_connection = x
                    if i == 7:
                        x_backout = x
                    continue
            else:
                attn_in = x_backout if x_backout is not None else x
                if compile_dense_layer_body:
                    x = self.compiled_dense_attn_mlp_layer(
                        x,
                        attn_in,
                        attn_args,
                        qkvo_w,
                        c_fc,
                        c_proj,
                        resid_lambdas_attn[i],
                        post_lambdas_attn[i],
                        x0_inject[i],
                        resid_lambdas_mlp[i],
                        post_lambdas_mlp[i],
                        i in self.paired_head_layers,
                    )
                    debug_check_finite(f"gpt.layer{i}.post_mlp_resid", x)
                    if i == engram_attnres_extra_source_layer:
                        engram_attnres_extra_source = x
                    if mhc_aux is not None and mhc_layer_start is not None:
                        layer_delta = x - mhc_layer_start
                        if engram_mhc_streams > 2:
                            layer_delta = layer_delta.unsqueeze(0)
                        mhc_aux = mhc_aux + layer_delta
                        debug_check_finite(f"gpt.layer{i}.post_mlp_mhc_aux", mhc_aux)
                    if i == 3:
                        skip_connection = x
                    if i == 7:
                        x_backout = x
                    continue
                debug_check_finite(f"gpt.layer{i}.attn_in", attn_in)
                attn_norm = norm(attn_in)
                debug_check_finite(f"gpt.layer{i}.attn_norm", attn_norm)
                if qkvo_w is not None:
                    debug_check_finite(f"gpt.layer{i}.qkvo_w", qkvo_w)
                attn_out = attn(attn_norm, attn_args, qkvo_w)
                debug_check_finite(f"gpt.layer{i}.attn_out", attn_out)
                x = resid_lambdas_attn[i] * x + post_lambdas_attn[i] * attn_out + x0_inject[i]
                debug_check_finite(f"gpt.layer{i}.post_attn_resid", x)
            mlp_in = norm(x)
            debug_check_finite(f"gpt.layer{i}.mlp_in", mlp_in)
            if self.training and not plain_mlp_train:
                mlp_out = ReLUSqrdMLP(mlp_in, c_fc, c_proj)
            else:
                mlp_out = F.relu(F.linear(mlp_in, c_fc)).square() @ c_proj
            debug_check_finite(f"gpt.layer{i}.mlp_out", mlp_out)
            x = resid_lambdas_mlp[i] * x + post_lambdas_mlp[i] * mlp_out
            debug_check_finite(f"gpt.layer{i}.post_mlp_resid", x)
            if i == engram_attnres_extra_source_layer:
                engram_attnres_extra_source = x
            if mhc_aux is not None and mhc_layer_start is not None:
                layer_delta = x - mhc_layer_start
                if engram_mhc_streams > 2:
                    layer_delta = layer_delta.unsqueeze(0)
                mhc_aux = mhc_aux + layer_delta
                debug_check_finite(f"gpt.layer{i}.post_mlp_mhc_aux", mhc_aux)
            if i == 3:
                skip_connection = x
            if i == 7:
                x_backout = x

        # back out contributions from first 7 layers
        debug_check_finite("gpt.pre_backout", x)
        debug_check_finite("gpt.x_backout", x_backout)
        x -= backout_lambda * x_backout
        debug_check_finite("gpt.post_backout_pre_norm", x)
        x = norm(x)
        if self.final_smear_mtp_lambda is not None:
            final_smear_mtp_weight = self.final_smear_mtp_lambda.to(dtype=x.dtype)
            x = torch.cat([x[:, :1], x[:, 1:] + final_smear_mtp_weight * x[:, :-1]], dim=1)
            debug_check_finite("gpt.final_smear_mtp", x)
        debug_check_finite("gpt.final_hidden_before_ce", x)
        debug_check_finite("gpt.lm_head_weight", self.lm_head.weight)
        # @Grad62304977 added tanh softcapping following Gemma 2 paper, @KoszarskyB reduced it from 30 to 15
        # @YouJiacheng shifted it by +15 (2*sigmoid(2*x)=tanh(x)+1). @classiclarryd updated to 23*sigmoid((logits+5)/7.5)
        if self.training or fused_ce_eval:
            loss_per_token = FusedSoftcappedCrossEntropy.apply(x.view(-1, x.size(-1)), target_seq, mtp_weights, self.lm_head.weight, self.lm_head.x_s, self.lm_head.w_s, self.lm_head.grad_s, grad_scale)
        else:
            loss_per_token = chunked_softcapped_cross_entropy(x, target_seq, mtp_weights, self.lm_head)
        if self.training and cache_recon_loss is not None and engram_cache_recon_weight > 0:
            loss_per_token = loss_per_token + (engram_cache_recon_weight * cache_recon_loss / loss_per_token.numel()).to(loss_per_token.dtype)
        debug_check_finite("gpt.loss_per_token", loss_per_token)
        return loss_per_token
# -----------------------------------------------------------------------------
# Distributed data loader

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32) # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True) # avoid pin_memory copy by @YouJiacheng
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy()) # avoid bytes->array copy by @YouJiacheng
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

BOS_ID = 50256
TRAIN_MAX_NUM_DOCS = {16384: 64, 32768: 96, 49152: 128}

class Shard:
    def __init__(self, tokens: Tensor, world_size: int = 1):
        self.tokens = tokens
        self.size = tokens.numel()
        self.world_size = world_size
        self.i = 0

        # Partial index now, full index async
        self.bos_idx = (tokens[:6_000_000] == BOS_ID).nonzero(as_tuple=True)[0].to(torch.int64).cpu().numpy()
        self._full_idx = None
        self._loader_thread = None
        self._ready = threading.Event()
        self._loader_thread = threading.Thread(target=self._scan)
        self._loader_thread.start()

    def _scan(self):
        self._full_idx = (self.tokens == BOS_ID).nonzero(as_tuple=True)[0].to(torch.int64).cpu().numpy()
        self._ready.set()

    def _maybe_switch(self):
        # Switch to full index as soon as async scan completes
        if self.bos_idx is not self._full_idx and self._ready.is_set():
            self._loader_thread.join()
            self.bos_idx = self._full_idx

    def next_batch(self, num_tokens_local: int, max_seq_len: int):
        self._maybe_switch()
        n = len(self.bos_idx)
        starts = [[] for _ in range(self.world_size)]
        ends = [[] for _ in range(self.world_size)]

        idx = self.i
        for r in range(self.world_size):
            cur_len = 0
            while cur_len <= num_tokens_local:
                if idx >= n:
                    raise StopIteration(f"Insufficient BOS ahead; hit tail of shard.")
                cur = self.bos_idx[idx]
                starts[r].append(cur)
                idx += 1
                end = min(self.bos_idx[idx] if idx < n else self.size,
                          cur + max_seq_len,
                          cur + num_tokens_local - cur_len + 1)
                ends[r].append(end)
                cur_len += end - cur

            assert cur_len == num_tokens_local + 1
        self.i = idx
        return starts, ends

    @staticmethod
    def load_async(file: Path, world_size: int = 1):
        """Returns getter function for async shard loading"""
        result = {}
        ready = threading.Event()
        def load():
            tokens = _load_data_shard(file)
            result['shard'] = Shard(tokens, world_size)
            ready.set()
        thread = threading.Thread(target=load)
        thread.start()
        def get():
            ready.wait()
            thread.join()
            return result['shard']
        return get

def get_bigram_hash(x):
    """
    Computes bigram hash for each position using [prev_token, curr_token].
    Multiply by arbitary large ints to get even spread over int32 range.
    Position 0 is mapped to the reserved index (vocab_size - 1).
    BOS_tokens within the batch will hash based on last token of prior doc. Masking this ran slower and showed no improvement.
    """
    rand_int_1 = 36313
    rand_int_2 = 27191
    mod = args.bigram_vocab_size-1
    x = x.to(torch.int32)
    out = torch.empty_like(x, pin_memory=True)
    out.copy_(x)
    out[0] = mod
    out[1:] = torch.bitwise_xor(rand_int_1 * out[1:], rand_int_2 * out[:-1]) % mod
    return out

def distributed_data_generator(filename_pattern: str, num_tokens: int, max_seq_len: int, grad_accum_steps: int = 1, align_to_bos: bool = True, data_seed: int = 0):
    # align_to_bos: each sequence begins with Beginning of Sequence token, sequences truncated to max_seq_len
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    assert num_tokens % (world_size * grad_accum_steps) == 0, "Batch size must be divisible by world size"
    num_tokens = num_tokens // grad_accum_steps

    files = [Path(file) for file in sorted(glob.glob(filename_pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {filename_pattern}")

    file_iter = iter(files)  # Use itertools.cycle(files) for multi-epoch training
    tokens = _load_data_shard(next(file_iter))
    if align_to_bos:
        shard = Shard(tokens, world_size)
        if data_seed:
            max_start = max(0, len(shard.bos_idx) - TRAIN_MAX_NUM_DOCS.get(num_tokens // world_size, next_multiple_of_n((num_tokens // world_size) // 300, n=128)) - 2)
            if max_start > 0:
                rng = np.random.default_rng(data_seed)
                shard.i = int(rng.integers(0, max_start + 1))
        next_shard_getter = Shard.load_async(next(file_iter), world_size)
    else:
        pos = 0  # for unaligned case

    while True:
        num_tokens_local = num_tokens // world_size
        max_num_docs = TRAIN_MAX_NUM_DOCS.get(num_tokens_local, next_multiple_of_n(num_tokens_local // 300, n=128))

        if align_to_bos:
            try:
                seq_starts, seq_ends = shard.next_batch(num_tokens_local, max_seq_len)
                start_idxs, end_idxs = torch.tensor(seq_starts[rank]), torch.tensor(seq_ends[rank])
            except StopIteration:
                # This shard is exhausted, load the next one in the next loop iteration.
                shard = next_shard_getter()
                tokens = shard.tokens
                try:
                    next_shard_getter = Shard.load_async(next(file_iter), world_size)
                except StopIteration:
                    next_shard_getter = None  # no more shards to preload
                continue

            buf = torch.cat([tokens[i:j] for i, j in zip(start_idxs, end_idxs)])
            _inputs = buf[:-1]
            _targets = buf[1:]
            end_idxs[-1] -= 1  # last document was too long to account for _targets offset
            cum_lengths = (end_idxs - start_idxs).cumsum(0)

        else:
            if pos + num_tokens + 1 >= len(tokens):  # should not occur for val data
                tokens, pos = _load_data_shard(next(file_iter)), 0

            pos_local = pos + rank * num_tokens_local
            buf = tokens[pos_local: pos_local + num_tokens_local + 1]
            _inputs = buf[:-1].view(num_tokens_local, )
            _targets = buf[1:].view(num_tokens_local, )

            cum_lengths = torch.nonzero(_inputs == BOS_ID)[:, 0]
            pos += num_tokens


        _cum_lengths = torch.full((max_num_docs,), num_tokens_local)
        _cum_lengths[0] = 0
        _cum_lengths[1:len(cum_lengths) + 1] = cum_lengths

        # Cast to int32 on CPU before transfer to avoid dtype conversion during .to()
        _inputs = _inputs.to(dtype=torch.int32)
        _targets = _targets.to(dtype=torch.int64)
        _cum_lengths = _cum_lengths.to(dtype=torch.int32)
        _bigram_inputs = get_bigram_hash(_inputs)
        _memory_inputs = _inputs if rom_token else _bigram_inputs

        new_params = yield (
            _inputs.to(device="cuda", non_blocking=True),
            _targets.to(device="cuda", non_blocking=True),
            _cum_lengths.to(device="cuda", non_blocking=True),
            _bigram_inputs.to(device="cuda", non_blocking=True),
            _memory_inputs.numpy(),
        )

        if new_params is not None:
            # makes it possible for generator to receive new (num_tokens, max_seq_len, grad_accum_steps) via .send()
            new_num_tokens, new_max_seq_len, new_grad_accum_steps = new_params
            assert new_num_tokens % (world_size * new_grad_accum_steps) == 0, "Num tokens must be divisible by world size"
            num_tokens = new_num_tokens // new_grad_accum_steps
            max_seq_len = new_max_seq_len

# -----------------------------------------------------------------------------
# Training Management

@dataclass(slots=True)
class Hyperparameters:
    # data
    data_path = os.environ.get("DATA_PATH", ".")
    train_files: str = os.path.join(data_path, "data/fineweb10B/fineweb_train_*.bin") # input .bin to train on
    val_files: str = os.path.join(data_path, "data/fineweb10B/fineweb_val_*.bin") # input .bin to eval validation loss on
    val_tokens: int = int(os.environ.get("VAL_TOKENS", "10485760")) # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    # batch sizes
    val_batch_size: int = int(os.environ.get("VAL_BATCH_SIZE", str(4 * 64 * 1024 * 8)))
    # schedule
    num_scheduled_iterations: int = int(os.environ.get("NUM_SCHEDULED_ITERATIONS", "1440"))  # number of steps to complete lr and ws schedule
    num_extension_iterations: int = int(os.environ.get("NUM_EXTENSION_ITERATIONS", "40"))  # number of steps to continue training at final lr and ws
    # evaluation and logging
    run_id: str = os.environ.get("RUN_ID", f"{uuid.uuid4()}")
    val_loss_every: int = int(os.environ.get("VAL_LOSS_EVERY", "250"))  # every how many steps to evaluate val loss? 0 for only at the end
    save_checkpoint: bool = env_flag("SAVE_CHECKPOINT", False)
    save_checkpoint_every: int = int(os.environ.get("SAVE_CHECKPOINT_EVERY", "0"))  # checkpoint every N steps; 0 means final only
    run_evals: bool = False  # run additional evaluations after training is completed
    # bigram hash embedding
    bigram_vocab_size: int = 50304 * int(os.environ.get("BIGRAM_FACTOR", "5"))

args = Hyperparameters()

@dataclass(slots=True)
class TrainingStage:
    lr_mul: float
    batch_size: int
    window_sizes: tuple[int, int]  # (short, long) in block units
    mtp_weights_start: list[float]
    mtp_weights_end: list[float]
    train_max_seq_len: int
    duration: float = None

class TrainingSchedule:
    """
    Training schedule initialized via TRAINING_STAGES
        1. Multi Token Prediction schedule of [1, 0.5, 0.25->0] -> [1, 0.5->0] -> [1] @varunneal
        2. Sliding Attention window schedule of [1,3] -> [3,7] -> [5,11] -> [6,13]
        3. YaRN updates to RoPE on window changes
        4. Split embed and lm head at 2/3 of training
        5. Batch size schedule of 8 -> 16 -> 24
        6. Post training extension of long windows from 13 to 20
        7. Seq len updates from 896 to 2048 at 1/3 of training
    """

    def __init__(self, stages: list[TrainingStage], scheduled_iterations: int, extension_iterations: int,
                 cooldown_frac: float = 0.5, split_embed_stage: int = 2, ws_post_yarn_ext: int = 20):
        self.stages = stages
        self.scheduled_iterations = scheduled_iterations
        self.cooldown_frac = cooldown_frac
        # increase final validation ws, used for YaRN extension and short window size @classiclarryd
        self.ws_post_yarn_ext = ws_post_yarn_ext

        self.total_steps = self.scheduled_iterations + extension_iterations

        # Build stage boundaries (last is extension stage)
        ends = [0, *[round(c * scheduled_iterations) for c in accumulate(s.duration for s in stages[:-1])], self.total_steps]
        assert self.scheduled_iterations == ends[-2]
        self.boundaries = list(pairwise(ends))

        # Split embed at specified stage (ensure odd step for Adam)
        self.split_step = self.boundaries[split_embed_stage][0] | 1

        # Precompute MTP weights for all steps
        self.mtp_weights = []
        for step in range(self.total_steps + 1):
            stage, t = self.lookup(step)
            w = [a + (b - a) * t for a, b in zip(stage.mtp_weights_start, stage.mtp_weights_end)]
            self.mtp_weights.append(torch.tensor(w, device=device))

    def lookup(self, step: int) -> tuple[TrainingStage, float]:
        # Returns stage and % of the way through that stage
        for i, (start, end) in enumerate(self.boundaries):
            if step < end:
                t = (step - start) / (end - start)
                return self.stages[i], t
        return self.stages[-1], 1.0

    def get_lr(self, step: int) -> float:
        # learning rate schedule: tied to batch size schedule, with cooldown at the end
        stage, _ = self.lookup(step)
        lr = stage.lr_mul
        cd_start = int(self.scheduled_iterations * (1 - self.cooldown_frac))
        if step >= cd_start:
            t = min(1.0, (step - cd_start) / (self.scheduled_iterations - cd_start))
            lr = lr * (1 - t) + 0.15 * t
        return lr

# window_sizes are in units of `block_size` tokens (defined in TrainingManager)
TRAINING_STAGES = [
    TrainingStage(duration=1/3, train_max_seq_len=896, batch_size=8 * 2048 * 8, window_sizes=(1, 3), lr_mul=1.0,
                  mtp_weights_start=[1.0, 0.5, 0.25], mtp_weights_end=[1.0, 0.5, 0.0]),
    TrainingStage(duration=1/3, train_max_seq_len=2048, batch_size=16 * 2048 * 8, window_sizes=(3, 7), lr_mul=1.52,  # (16/8)**0.6
                  mtp_weights_start=[1.0, 0.5], mtp_weights_end=[1.0, 0.0]),
    TrainingStage(duration=1/3, train_max_seq_len=2048, batch_size=24 * 2048 * 8, window_sizes=(5, 11), lr_mul=1.73,  # (24/8)**0.5
                  mtp_weights_start=[1.0], mtp_weights_end=[1.0]),
    # extension stage
    TrainingStage(train_max_seq_len=2048, batch_size=24 * 2048 * 8, window_sizes=(6, 13), lr_mul=1.0,  # lr_mul is not used
                  mtp_weights_start=[1.0], mtp_weights_end=[1.0]),
]

def _parse_stage_override(raw: str, cast, name: str) -> list:
    vals = [cast(x) for x in raw.replace(",", " ").split() if x]
    if len(vals) not in (1, len(TRAINING_STAGES)):
        raise ValueError(f"{name} must have length 1 or {len(TRAINING_STAGES)}")
    if len(vals) == 1:
        vals = vals * len(TRAINING_STAGES)
    return vals


def _round_batch_to_divisible(batch_size: int) -> int:
    # Attention kernels require per-rank microbatches to be a multiple of 16.
    divisor = world_size * grad_accum_steps * 16
    return max(divisor, (batch_size // divisor) * divisor)


train_stage_batch_sizes_raw = os.environ.get("TRAIN_STAGE_BATCH_SIZES", "")
train_stage_batch_mults_raw = os.environ.get("TRAIN_STAGE_BATCH_MULTS", "")
if train_stage_batch_sizes_raw:
    for stage, batch_size in zip(TRAINING_STAGES, _parse_stage_override(train_stage_batch_sizes_raw, int, "TRAIN_STAGE_BATCH_SIZES")):
        if batch_size <= 0:
            raise ValueError("TRAIN_STAGE_BATCH_SIZES values must be positive")
        stage.batch_size = _round_batch_to_divisible(batch_size)
elif train_stage_batch_mults_raw:
    for stage, mult in zip(TRAINING_STAGES, _parse_stage_override(train_stage_batch_mults_raw, float, "TRAIN_STAGE_BATCH_MULTS")):
        if mult <= 0:
            raise ValueError("TRAIN_STAGE_BATCH_MULTS values must be positive")
        stage.batch_size = _round_batch_to_divisible(round(stage.batch_size * mult))

# TODO - Confirm.
training_schedule = TrainingSchedule(TRAINING_STAGES, args.num_scheduled_iterations, args.num_extension_iterations, cooldown_frac=0.60)
#training_schedule = TrainingSchedule(TRAINING_STAGES, args.num_scheduled_iterations, args.num_extension_iterations, cooldown_frac=0.55)

def get_muon_momentum(step: int, muon_warmup_steps=300, muon_cooldown_steps=50, momentum_min=0.85, momentum_max=0.95):
    # warmup phase: linearly increase momentum from min to max
    # cooldown phase: linearly decrease momentum from max to min
    momentum_cd_start = training_schedule.total_steps - muon_cooldown_steps
    if step < muon_warmup_steps:
        frac = step / muon_warmup_steps
        momentum = momentum_min + frac * (momentum_max - momentum_min)
    elif step > momentum_cd_start:
        frac = (step - momentum_cd_start) / muon_cooldown_steps
        momentum = momentum_max - frac * (momentum_max - momentum_min)
    else:
        momentum = momentum_max
    return momentum

class TrainingManager():
    """
    Manages the NorMuonAndAdam for all parameters with explicit ordering.
        1. Scalars are given higher momentum terms to smooth learning @ChrisJMcCormick
        2. Adam optimizers are only stepped on odd steps @classiclarryd
        3. Explicit scatter_order and work_order for communication scheduling (no backward hooks)
        4. Muon has a linear momentum warmup and cooldown schedule
        5. Learning rates follow a linear decay schedule
        6. Embed is tied to lm_head until split step (2/3 of training), then untied @classiclarryd
    """
    def __init__(self, model):
        self.model = model
        self.block_size = 128

        # - Ordering dictates when to launch reduce/reduce_scatter operations
        # - "sharded" parameters use reduce_scatter/all_gather and "replicated" ones use all_reduce
        # - lr_mul and wd_mul are per-parameter learning rate and weight decay multipliers
        self.param_table = {
            "qk_bank":        {"optim": "normuon", "comms": "sharded",    "adam_betas": None},
            "vo_bank":        {"optim": "normuon", "comms": "sharded",    "adam_betas": None},
            "mlp_bank":       {"optim": "normuon", "comms": "sharded",    "adam_betas": None},
            "scalars":        {"optim": "adam",    "comms": "replicated", "adam_betas": [0.9,  0.99], "lr_mul": 5.0,  "wd_mul": 0.0},
            "smear_gate":     {"optim": "adam",    "comms": "replicated", "adam_betas": [0.9,  0.99], "lr_mul": 0.01, "wd_mul": 0.0},
            "skip_gate":      {"optim": "adam",    "comms": "replicated", "adam_betas": [0.9,  0.99], "lr_mul": 0.05, "wd_mul": 0.0},
            "attn_gate_bank": {"optim": "adam",    "comms": "replicated", "adam_betas": [0.9,  0.99]},
            "ve_gate_bank":   {"optim": "adam",    "comms": "replicated", "adam_betas": [0.9,  0.99]},
            "lm_head":        {"optim": "adam",    "comms": "sharded",    "adam_betas": [0.5,  0.95], "wd_mul": 150.},
            "bigram_embed":   {"optim": "adam",    "comms": "sharded_sparse", "adam_betas": [0.75, 0.95], "lr_mul": 75.,  "wd_mul": 5.0},
            "post_lambdas":   {"optim": "adam",    "comms": "replicated",     "adam_betas": [0.9,  0.95], "lr_mul": 1.0,  "wd_mul": 0.0},
            "x0_lambdas":     {"optim": "adam",    "comms": "replicated",     "adam_betas": [0.9,  0.95], "lr_mul": 1.0,  "wd_mul": 0.0},
            "bigram_lambdas": {"optim": "adam",    "comms": "replicated",     "adam_betas": [0.9,  0.95], "lr_mul": 1.0,  "wd_mul": 0.0},
            "resid_lambdas":  {"optim": "adam",    "comms": "replicated",     "adam_betas": [0.9,  0.95], "lr_mul": 5.0,  "wd_mul": 0.0},
            "value_embeds":   {"optim": "adam",    "comms": "sharded",    "adam_betas": [0.75, 0.95], "lr_mul": 75.,  "wd_mul": 5.0},
            "embed":          {"optim": "adam",    "comms": "sharded",    "adam_betas": [0.5,  0.95], "wd_mul": 150.},
        }
        if engram_bigram:
            del self.param_table["bigram_embed"]
            base_model_for_engram = model._orig_mod if hasattr(model, "_orig_mod") else model
            layer_readout_ids = tuple(getattr(base_model_for_engram, "bigram_layer_ids", ())) if engram_layer_readouts else ()
            layer_readout_delta_ids = tuple(getattr(base_model_for_engram, "bigram_layer_ids", ())) if engram_layer_readout_delta else ()
            engram_has_memory_proj = engram_store_dim > 0 and engram_store_dim != engram_dim // ((engram_max_ngram - 1) * engram_heads)
            if not (engram_freeze_memory or engram_offload or engram_sparse_adam or engram_sparse_vector_adam or engram_sparse_scalar_adam or engram_sparse_row_adagrad):
                self.param_table["bigram_embed.embedding"] = {"optim": "adam", "comms": "sharded_sparse", "adam_betas": [0.75, 0.95], "lr_mul": engram_lr_mul, "wd_mul": 0.0}
            engram_readout_labels = []
            if layer_readout_ids:
                for layer_id in layer_readout_ids:
                    prefix = f"bigram_embed.layer_readouts.{layer_id}"
                    engram_readout_labels.extend([
                        f"{prefix}.value_proj",
                        f"{prefix}.key_proj",
                    ])
                    self.param_table.update({
                        f"{prefix}.value_proj":      {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.95], "lr_mul": 0.5, "wd_mul": 0.0},
                        f"{prefix}.key_proj":        {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.95], "lr_mul": 0.5, "wd_mul": 0.0},
                    })
                    if engram_short_conv:
                        engram_readout_labels.extend([
                            f"{prefix}.short_conv_norm",
                            f"{prefix}.short_conv",
                        ])
                        self.param_table.update({
                            f"{prefix}.short_conv_norm": {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.95], "lr_mul": 0.5, "wd_mul": 0.0},
                            f"{prefix}.short_conv":      {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.95], "lr_mul": 0.5, "wd_mul": 0.0},
                        })
            else:
                engram_readout_labels = ["bigram_embed.value_proj", "bigram_embed.key_proj"]
                self.param_table.update({
                    "bigram_embed.value_proj":      {"optim": "adam", "comms": "replicated",     "adam_betas": [0.9,  0.95], "lr_mul": 0.5, "wd_mul": 0.0},
                    "bigram_embed.key_proj":        {"optim": "adam", "comms": "replicated",     "adam_betas": [0.9,  0.95], "lr_mul": 0.5, "wd_mul": 0.0},
                })
                if engram_short_conv:
                    engram_readout_labels.extend(["bigram_embed.short_conv_norm", "bigram_embed.short_conv"])
                    self.param_table.update({
                        "bigram_embed.short_conv_norm": {"optim": "adam", "comms": "replicated",     "adam_betas": [0.9,  0.95], "lr_mul": 0.5, "wd_mul": 0.0},
                        "bigram_embed.short_conv":      {"optim": "adam", "comms": "replicated",     "adam_betas": [0.9,  0.95], "lr_mul": 0.5, "wd_mul": 0.0},
                    })
            if layer_readout_delta_ids:
                for layer_id in layer_readout_delta_ids:
                    prefix = f"bigram_embed.layer_readout_deltas.{layer_id}"
                    engram_readout_labels.extend([
                        f"{prefix}.value_proj",
                        f"{prefix}.key_proj",
                    ])
                    self.param_table.update({
                        f"{prefix}.value_proj": {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.95], "lr_mul": 0.5, "wd_mul": 0.0},
                        f"{prefix}.key_proj":   {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.95], "lr_mul": 0.5, "wd_mul": 0.0},
                    })
                if engram_layer_readout_delta_learned_scale:
                    engram_readout_labels.append("bigram_embed.layer_readout_delta_scale_logits")
                    self.param_table["bigram_embed.layer_readout_delta_scale_logits"] = {
                        "optim": "adam",
                        "comms": "replicated",
                        "adam_betas": [0.9, 0.95],
                        "lr_mul": engram_layer_readout_delta_learned_scale_lr_mul,
                        "wd_mul": 0.0,
                    }
            if engram_has_memory_proj:
                self.param_table["bigram_embed.memory_proj"] = {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.95], "lr_mul": 0.5, "wd_mul": 0.0}
            engram_head_mix_work = []
            if engram_layer_head_mix and not engram_head_mix_freeze:
                engram_head_mix_work.append("bigram_embed.layer_head_mix_logits")
                self.param_table["bigram_embed.layer_head_mix_logits"] = {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.95], "lr_mul": 0.5, "wd_mul": 0.0}
            elif engram_head_mix and not engram_head_mix_freeze:
                engram_head_mix_work.append("bigram_embed.head_mix_logits")
                self.param_table["bigram_embed.head_mix_logits"] = {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.95], "lr_mul": 0.5, "wd_mul": 0.0}
                if engram_layer_head_mix_delta:
                    engram_head_mix_work.append("bigram_embed.layer_head_mix_delta_logits")
                    self.param_table["bigram_embed.layer_head_mix_delta_logits"] = {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.95], "lr_mul": 0.5, "wd_mul": 0.0}
            engram_sketch_slot_mix_work = []
            if engram_sketch_slot_mix or engram_sketch_combine_mix:
                engram_sketch_slot_mix_work.append("bigram_embed.sketch_slot_mix_logits")
                self.param_table["bigram_embed.sketch_slot_mix_logits"] = {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.95], "lr_mul": engram_sketch_mix_lr_mul, "wd_mul": 0.0}
            if engram_sketch_aux_learned_scale:
                engram_sketch_slot_mix_work.append("bigram_embed.sketch_aux_scale_logit")
                self.param_table["bigram_embed.sketch_aux_scale_logit"] = {
                    "optim": "adam",
                    "comms": "replicated",
                    "adam_betas": [0.9, 0.95],
                    "lr_mul": engram_sketch_aux_learned_scale_lr_mul,
                    "wd_mul": 0.0,
                }
            engram_static_gate_work = []
            if engram_static_gate:
                engram_static_gate_work.append("bigram_embed.static_gate_logits")
                self.param_table["bigram_embed.static_gate_logits"] = {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.95], "lr_mul": 0.5, "wd_mul": 0.0}
            engram_cache_readout_labels = []
            if engram_cache_readout:
                prefix = f"bigram_embed.cache_readouts.{engram_cache_recon_source_layer}"
                engram_cache_readout_labels.extend([
                    f"{prefix}.value_proj",
                    f"{prefix}.key_proj",
                ])
                self.param_table.update({
                    f"{prefix}.value_proj":      {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.95], "lr_mul": 0.5, "wd_mul": 0.0},
                    f"{prefix}.key_proj":        {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.95], "lr_mul": 0.5, "wd_mul": 0.0},
                })
                if engram_short_conv:
                    engram_cache_readout_labels.extend([
                        f"{prefix}.short_conv_norm",
                        f"{prefix}.short_conv",
                    ])
                    self.param_table.update({
                        f"{prefix}.short_conv_norm": {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.95], "lr_mul": 0.5, "wd_mul": 0.0},
                        f"{prefix}.short_conv":      {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.95], "lr_mul": 0.5, "wd_mul": 0.0},
                    })
                if engram_cache_learned_cfg:
                    self.param_table["cache_cfg_lambdas"] = {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.95], "lr_mul": 0.5, "wd_mul": 0.0}
            engram_latent_labels = []
            if engram_latent:
                engram_latent_labels.append("bigram_embed.latent_proj")
                self.param_table["bigram_embed.latent_proj"] = {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.95], "lr_mul": 0.5, "wd_mul": 0.0}
                if engram_latent_quantizer == "pkm":
                    engram_latent_labels.append("bigram_embed.latent_pkm_keys")
                    self.param_table["bigram_embed.latent_pkm_keys"] = {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.95], "lr_mul": 0.5, "wd_mul": 0.0}
                if engram_latent_ste_scale > 0:
                    engram_latent_labels.append("bigram_embed.latent_ste_proj")
                    self.param_table["bigram_embed.latent_ste_proj"] = {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.95], "lr_mul": 0.5, "wd_mul": 0.0}
            if engram_mhc:
                if engram_mhc_identity:
                    self.param_table["mhc_identity_router"] = {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.95], "lr_mul": 0.1, "wd_mul": 0.0}
                    self.param_table["mhc_identity_router.bias"] = {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.95], "lr_mul": 0.1, "wd_mul": 0.0}
                else:
                    self.param_table["mhc_logit"] = {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.95], "lr_mul": 1.0, "wd_mul": 0.0}
                if engram_mhc_dynamic and not engram_mhc_identity:
                    self.param_table["mhc_router"] = {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.95], "lr_mul": 0.1, "wd_mul": 0.0}
            if engram_attnres_merge:
                self.param_table["engram_attnres_query"] = {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.95], "lr_mul": 0.1, "wd_mul": 0.0}
                if engram_attnres_direct_residual:
                    self.param_table["engram_direct_resid"] = {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.95], "lr_mul": 0.5, "wd_mul": 0.0}
                if engram_attnres_layer_gain:
                    self.param_table["engram_attnres_layer_gain_log"] = {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.95], "lr_mul": 0.25, "wd_mul": 0.0}
                if engram_attnres_extra_target_layer >= 0:
                    self.param_table["engram_attnres_extra_bias"] = {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.95], "lr_mul": 0.5, "wd_mul": 0.0}
        if final_smear_mtp:
            self.param_table["final_smear_mtp_lambda"] = {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.99], "lr_mul": 1.0, "wd_mul": 0.0}
        if rom_bigram:
            del self.param_table["bigram_embed"]
            rom_state_work = []
            if not (rom_state_sparse_adam or rom_state_sparse_sgd or rom_state_normwrite or rom_state_recovered_normwrite):
                self.param_table["bigram_embed.state"] = {"optim": "adam", "comms": "sharded_sparse", "adam_betas": [0.75, 0.95], "lr_mul": 5., "wd_mul": 0.0}
                rom_state_work = ["bigram_embed.state"]
            self.param_table.update({
                "bigram_embed.q_proj":    {"optim": "adam", "comms": "replicated",     "adam_betas": [0.9,  0.95], "lr_mul": 0.5,  "wd_mul": 0.0},
                "bigram_embed.out_proj":  {"optim": "adam", "comms": "replicated",     "adam_betas": [0.9,  0.95], "lr_mul": 0.5,  "wd_mul": 0.0},
            })
            if rom_engram_gate:
                self.param_table["bigram_embed.engram_key_proj"] = {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.95], "lr_mul": 0.5, "wd_mul": 0.0}
            else:
                self.param_table["bigram_embed.gate_proj"] = {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.95], "lr_mul": 0.5, "wd_mul": 0.0}
            if rom_token:
                self.param_table.update({
                    "bigram_embed.compose_gate_proj":  {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.95], "lr_mul": 0.5, "wd_mul": 0.0},
                    "bigram_embed.compose_decay_proj": {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.95], "lr_mul": 0.5, "wd_mul": 0.0},
                })
            if rom_read_mlp:
                self.param_table.update({
                    "bigram_embed.read_mlp_norm": {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.95], "lr_mul": rom_read_mlp_lr_mul, "wd_mul": 0.0},
                    "bigram_embed.read_mlp_fc1":  {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.95], "lr_mul": rom_read_mlp_lr_mul, "wd_mul": 0.0},
                    "bigram_embed.read_mlp_fc2":  {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.95], "lr_mul": rom_read_mlp_lr_mul, "wd_mul": 0.0},
                })
            if rom_write:
                self.param_table.update({
                    "bigram_embed.write_alpha":     {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.95], "lr_mul": 1.0, "wd_mul": 0.0},
                    "bigram_embed.k_proj":          {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.95], "lr_mul": 0.5, "wd_mul": 0.0},
                    "bigram_embed.v_proj":          {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.95], "lr_mul": 0.5, "wd_mul": 0.0},
                    "bigram_embed.beta_proj":       {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.95], "lr_mul": 0.5, "wd_mul": 0.0},
                    "bigram_embed.decay_proj":      {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.95], "lr_mul": 0.5, "wd_mul": 0.0},
                    "bigram_embed.write_gate_proj": {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.95], "lr_mul": 0.5, "wd_mul": 0.0},
                })
            if rom_short_conv:
                self.param_table.update({
                    "bigram_embed.short_conv_norm": {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.95], "lr_mul": 0.5, "wd_mul": 0.0},
                    "bigram_embed.short_conv":      {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.95], "lr_mul": 0.5, "wd_mul": 0.0},
                })

        final_smear_work = ["final_smear_mtp_lambda"] if final_smear_mtp else []

        # - Process smaller/faster params first while large reduces complete
        # - lm_head must complete before embed sync (when tied)
        self.work_order = [
            "scalars", "smear_gate", "skip_gate", "attn_gate_bank", "ve_gate_bank", "post_lambdas", "x0_lambdas", "bigram_lambdas", "resid_lambdas", *final_smear_work,  # Small, fast
            "value_embeds", "bigram_embed",  # Medium
            "lm_head", "embed",   # lm_head must complete before embed sync (when tied)
            "qk_bank", "vo_bank", "mlp_bank",  # Large, polar express - process last to maximize overlap
        ]
        if engram_bigram:
            engram_state_work = [] if (engram_freeze_memory or engram_offload or engram_sparse_adam or engram_sparse_vector_adam or engram_sparse_scalar_adam or engram_sparse_row_adagrad) else ["bigram_embed.embedding"]
            engram_proj_work = ["bigram_embed.memory_proj"] if "bigram_embed.memory_proj" in self.param_table else []
            engram_latent_work = engram_latent_labels if engram_latent else []
            engram_readout_work = engram_readout_labels
            engram_cache_readout_work = engram_cache_readout_labels if engram_cache_readout else []
            engram_cache_cfg_work = ["cache_cfg_lambdas"] if engram_cache_readout and engram_cache_learned_cfg else []
            engram_attnres_work = ["engram_attnres_query"] if engram_attnres_merge else []
            if engram_attnres_direct_residual:
                engram_attnres_work.append("engram_direct_resid")
            if engram_attnres_layer_gain:
                engram_attnres_work.append("engram_attnres_layer_gain_log")
            if engram_attnres_extra_target_layer >= 0:
                engram_attnres_work.append("engram_attnres_extra_bias")
            engram_mhc_work = []
            if engram_mhc:
                if engram_mhc_identity:
                    engram_mhc_work.append("mhc_identity_router")
                    engram_mhc_work.append("mhc_identity_router.bias")
                else:
                    engram_mhc_work.append("mhc_logit")
                    if engram_mhc_dynamic:
                        engram_mhc_work.append("mhc_router")
            self.work_order = [
                "scalars", "smear_gate", "skip_gate", "attn_gate_bank", "ve_gate_bank", "post_lambdas", "x0_lambdas", "bigram_lambdas", "resid_lambdas", *final_smear_work,
                *engram_mhc_work, *engram_attnres_work, *engram_cache_cfg_work, *engram_state_work, *engram_proj_work, *engram_latent_work, *engram_head_mix_work, *engram_sketch_slot_mix_work, *engram_static_gate_work, *engram_readout_work, *engram_cache_readout_work,
                "value_embeds",
                "lm_head", "embed",
                "qk_bank", "vo_bank", "mlp_bank",
            ]
        if rom_bigram:
            rom_compose_work = ["bigram_embed.compose_gate_proj", "bigram_embed.compose_decay_proj"] if rom_token else []
            rom_gate_work = ["bigram_embed.engram_key_proj"] if rom_engram_gate else ["bigram_embed.gate_proj"]
            rom_read_mlp_work = ["bigram_embed.read_mlp_norm", "bigram_embed.read_mlp_fc1", "bigram_embed.read_mlp_fc2"] if rom_read_mlp else []
            rom_short_conv_work = ["bigram_embed.short_conv_norm", "bigram_embed.short_conv"] if rom_short_conv else []
            rom_state_work = [] if (rom_state_sparse_adam or rom_state_sparse_sgd or rom_state_normwrite or rom_state_recovered_normwrite) else ["bigram_embed.state"]
            rom_attnres_work = ["engram_attnres_query"] if engram_attnres_merge else []
            if engram_attnres_merge:
                self.param_table["engram_attnres_query"] = {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.95], "lr_mul": 0.1, "wd_mul": 0.0}
            self.work_order = [
                "scalars", "smear_gate", "skip_gate", "attn_gate_bank", "ve_gate_bank", "post_lambdas", "x0_lambdas", "bigram_lambdas", "resid_lambdas", *final_smear_work,
                *rom_attnres_work, "bigram_embed.q_proj", *rom_compose_work, *rom_gate_work, *rom_read_mlp_work, "bigram_embed.out_proj", *rom_short_conv_work,
                "value_embeds", *rom_state_work,
                "lm_head", "embed",
                "qk_bank", "vo_bank", "mlp_bank",
            ]
            if rom_write:
                self.work_order = [
                    "scalars", "smear_gate", "skip_gate", "attn_gate_bank", "ve_gate_bank", "post_lambdas", "x0_lambdas", "bigram_lambdas", "resid_lambdas", *final_smear_work,
                    *rom_attnres_work, "bigram_embed.write_alpha", "bigram_embed.q_proj", *rom_compose_work, "bigram_embed.k_proj", "bigram_embed.v_proj", "bigram_embed.beta_proj", "bigram_embed.decay_proj", *rom_gate_work, "bigram_embed.write_gate_proj", *rom_read_mlp_work, "bigram_embed.out_proj", *rom_short_conv_work,
                    "value_embeds", *rom_state_work,
                    "lm_head", "embed",
                    "qk_bank", "vo_bank", "mlp_bank",
                ]

        adam_defaults = dict(
            lr=0.008,
            eps=1e-10,
            weight_decay=0.005,
        )
        self.engram_offload_lr_mul = engram_lr_mul
        self.engram_offload_betas = (0.75, 0.95)
        self.engram_offload_initial_lr = adam_defaults["lr"]
        self.engram_offload_eps = adam_defaults["eps"]

        normuon_defaults = dict(
            lr=0.023,
            momentum=0.95,
            beta2=0.9,
            weight_decay=1.2,
        )

        sparse_state_labels = {"bigram_embed.state"} if (rom_state_sparse_adam or rom_state_sparse_sgd or rom_state_normwrite or rom_state_recovered_normwrite) else set()
        if (engram_sparse_adam or engram_sparse_vector_adam or engram_sparse_scalar_adam or engram_sparse_row_adagrad) and not engram_freeze_memory:
            sparse_state_labels.add("bigram_embed.embedding")
        named_params = [
            (name, param)
            for name, param in model.named_parameters()
            if param.requires_grad and getattr(param, "label", None) not in sparse_state_labels
        ]
        self.optimizer = NorMuonAndAdam(
            named_params,
            param_table=self.param_table,
            scatter_order=list(self.param_table),  # Dict order defines scatter priority
            work_order=self.work_order,
            adam_defaults=adam_defaults,
            normuon_defaults=normuon_defaults,
        )
        self.sparse_adam_param = None
        self.sparse_adam_lr_mul = rom_sparse_adam_lr_mul
        self.sparse_adam = None
        self.sparse_sgd_param = None
        self.sparse_sgd_lr_mul = rom_sparse_sgd_lr_mul
        self.sparse_sgd = None
        self.sparse_normwrite_param = None
        self.sparse_normwrite = None
        self.sparse_recovered_param = None
        self.sparse_recovered_module = None
        self.engram_sparse_adam_param = None
        self.engram_sparse_adam = None
        self.engram_sparse_update_metrics: dict[str, float | int] = {}
        base_model_for_snoo = model._orig_mod if hasattr(model, "_orig_mod") else model
        snoo_params = [
            param for _, param in base_model_for_snoo.named_parameters()
            if param.requires_grad and (snoo_include_engram or getattr(param, "label", None) != "bigram_embed.embedding")
        ]
        self.snoo = Snoo(snoo_params, lr=snoo_lr, momentum=snoo_momentum, k=snoo_k) if snoo_outer else None
        self.adam_every_step_labels = {
            label for label in self.param_table
            if engram_adam_every_step and engram_bigram and label.startswith("bigram_embed.")
        }
        if rom_state_sparse_adam:
            base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            self.sparse_adam_param = base_model.bigram_memory_param()
            if rom_sparse_row_scalar_adam:
                self.sparse_adam = SparseRowScalarAdam(
                    [self.sparse_adam_param],
                    lr=adam_defaults["lr"] * self.sparse_adam_lr_mul,
                    betas=(0.75, rom_sparse_adam_beta2),
                    eps=adam_defaults["eps"],
                    weight_decay=0.0,
                )
            else:
                self.sparse_adam = torch.optim.SparseAdam(
                    [self.sparse_adam_param],
                    lr=adam_defaults["lr"] * self.sparse_adam_lr_mul,
                    betas=(0.75, rom_sparse_adam_beta2),
                    eps=adam_defaults["eps"],
                )
        if rom_state_sparse_sgd:
            base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            self.sparse_sgd_param = base_model.bigram_memory_param()
            self.sparse_sgd = SparseSGDMomentum(
                [self.sparse_sgd_param],
                lr=adam_defaults["lr"] * self.sparse_sgd_lr_mul,
                momentum=rom_sparse_sgd_momentum,
                weight_decay=0.0,
            )
        if rom_state_normwrite:
            base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            self.sparse_normwrite_param = base_model.bigram_memory_param()
            self.sparse_normwrite = SparseNormalizedWrite(
                [self.sparse_normwrite_param],
                write_rms=rom_state_write_rms,
                row_cap=rom_state_row_rms_cap,
            )
        if rom_state_recovered_normwrite:
            base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            self.sparse_recovered_param = base_model.bigram_memory_param()
            self.sparse_recovered_module = base_model.bigram_embed
        if (engram_sparse_adam or engram_sparse_vector_adam or engram_sparse_scalar_adam or engram_sparse_row_adagrad) and not engram_freeze_memory:
            base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            self.engram_sparse_adam_param = base_model.bigram_memory_param()
            if engram_sparse_row_adagrad:
                self.engram_sparse_adam = SparseRowWiseAdagrad(
                    [self.engram_sparse_adam_param],
                    lr=adam_defaults["lr"] * engram_lr_mul,
                    eps=adam_defaults["eps"],
                    weight_decay=engram_sparse_weight_decay,
                )
            elif engram_sparse_vector_adam:
                self.engram_sparse_adam = SparseRowScalarAdam(
                    [self.engram_sparse_adam_param],
                    lr=adam_defaults["lr"] * engram_lr_mul,
                    betas=(engram_sparse_adam_beta1, engram_sparse_adam_beta2),
                    eps=adam_defaults["eps"],
                    weight_decay=engram_sparse_weight_decay,
                )
            elif engram_sparse_scalar_adam:
                self.engram_sparse_adam = SparseScalarAdam(
                    [self.engram_sparse_adam_param],
                    lr=adam_defaults["lr"] * engram_lr_mul,
                    betas=(engram_sparse_adam_beta1, engram_sparse_adam_beta2),
                    eps=adam_defaults["eps"],
                    weight_decay=engram_sparse_weight_decay,
                )
            elif engram_sparse_hit_lr or engram_sparse_fal or engram_sparse_ifal:
                self.engram_sparse_adam = SparseAdamWithHitLR(
                    [self.engram_sparse_adam_param],
                    lr=adam_defaults["lr"] * engram_lr_mul,
                    betas=(engram_sparse_adam_beta1, engram_sparse_adam_beta2),
                    eps=adam_defaults["eps"],
                    weight_decay=engram_sparse_weight_decay,
                    exponent=engram_sparse_hit_lr_exponent,
                    min_scale=engram_sparse_hit_lr_min,
                    max_scale=engram_sparse_hit_lr_max,
                    fal=engram_sparse_fal,
                    ifal=engram_sparse_ifal,
                )
            elif engram_sparse_adam_tail_steps > 0:
                self.engram_sparse_adam = SparseAdamWithTail(
                    [self.engram_sparse_adam_param],
                    lr=adam_defaults["lr"] * engram_lr_mul,
                    betas=(engram_sparse_adam_beta1, engram_sparse_adam_beta2),
                    eps=adam_defaults["eps"],
                    weight_decay=engram_sparse_weight_decay,
                    tail_steps=engram_sparse_adam_tail_steps,
                    tail_scale=engram_sparse_adam_tail_scale,
                )
            else:
                self.engram_sparse_adam = torch.optim.SparseAdam(
                    [self.engram_sparse_adam_param],
                    lr=adam_defaults["lr"] * engram_lr_mul,
                    betas=(engram_sparse_adam_beta1, engram_sparse_adam_beta2),
                    eps=adam_defaults["eps"],
                )

        # Split embed from lm_head at 2/3 of training (on an odd step so Adam updates)
        self.split_step = training_schedule.split_step
        base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        self.bigram_param = base_model.bigram_memory_param()

        self.reset()

    def apply_final_ws_ext(self):
        self.ws_long = training_schedule.ws_post_yarn_ext

    def get_forward_args(self):
        return ForwardScheduleConfig(
            mtp_weights = self.mtp_weights,
            ws_short = self.ws_short * self.block_size,
            ws_long = self.ws_long * self.block_size,
            train_max_seq_len = self.train_max_seq_len
        )

    def _is_adam_step(self, step: int):
        """Adam params are only updated on odd steps."""
        return step % 2 == 1

    def get_transition_steps(self):
        return [start for start, _ in training_schedule.boundaries[1:]]

    def advance_schedule(self, step: int):
        global engram_head_dropout_current, engram_output_dropout_current, engram_attnres_merge_gain_current
        stage, _ = training_schedule.lookup(step)
        if engram_head_dropout_schedule_steps > 0:
            progress = min(max(step / engram_head_dropout_schedule_steps, 0.0), 1.0)
            engram_head_dropout_current = engram_head_dropout + progress * (engram_head_dropout_final - engram_head_dropout)
        else:
            engram_head_dropout_current = engram_head_dropout
        if engram_output_dropout_schedule_steps > 0:
            progress = min(max((step - engram_output_dropout_schedule_start) / engram_output_dropout_schedule_steps, 0.0), 1.0)
            engram_output_dropout_current = engram_output_dropout + progress * (engram_output_dropout_final - engram_output_dropout)
        else:
            engram_output_dropout_current = engram_output_dropout
        if engram_attnres_gain_warmup_steps > 0:
            progress = min(max(step / engram_attnres_gain_warmup_steps, 0.0), 1.0)
            engram_attnres_merge_gain_current = engram_attnres_merge_gain * progress
        else:
            engram_attnres_merge_gain_current = engram_attnres_merge_gain
        if engram_bigram and engram_offload:
            base_model = self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model
            base_model.bigram_embed.set_prefetch_moments((self._is_adam_step(step) or engram_adam_every_step) and engram_offload_gpu_adam and engram_offload_prefetch_moments)
        self.ws_short, new_ws_long = stage.window_sizes
        if new_ws_long != self.ws_long:
            self.model.yarn.apply(self.ws_long * self.block_size, new_ws_long * self.block_size)
            self.model.yarn_paired_head.apply(self.ws_long * self.block_size, new_ws_long * self.block_size)

        new_batch_size = stage.batch_size
        new_train_max_seq_len = stage.train_max_seq_len
        if new_batch_size != self.batch_size or new_train_max_seq_len != self.train_max_seq_len:
            self.train_loader_send_args = (new_batch_size, new_train_max_seq_len, grad_accum_steps)
            self.batch_size = new_batch_size
            self.train_max_seq_len = new_train_max_seq_len
        else:
            self.train_loader_send_args = None

        self.ws_long = new_ws_long
        self.mtp_weights = training_schedule.mtp_weights[min(step, len(training_schedule.mtp_weights) - 1)]

    def step_optimizers(self, step: int):
        opt_profile = profile_step_enabled(step)
        opt_pairs = []
        opt_wall_times: dict[str, float] = {}
        opt_wall_start = time.perf_counter()
        step_lr = training_schedule.get_lr(step)
        engram_step_lr = max(step_lr, engram_lr_floor)
        muon_momentum = get_muon_momentum(step)
        do_adam = self._is_adam_step(step)

        # Update learning rates and momentum for all params
        wall_start = time.perf_counter()
        for param, p_cfg in self.optimizer.param_cfgs.items():
            lr_scale = engram_step_lr if p_cfg.label == "bigram_embed.embedding" else step_lr
            p_cfg.lr = p_cfg.initial_lr * lr_scale
            if p_cfg.optim == "normuon":
                p_cfg.momentum = muon_momentum
        if opt_profile:
            profile_add_wall(opt_wall_times, "lr_setup", wall_start)

        if engram_bigram and engram_offload and engram_offload_async_adam:
            wall_start = time.perf_counter()
            event = profile_start_event(opt_profile)
            base_model = self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model
            base_model.bigram_embed.start_offloaded_adam(
                do_adam=do_adam or engram_adam_every_step,
                lr=self.engram_offload_initial_lr * engram_step_lr * self.engram_offload_lr_mul,
                betas=self.engram_offload_betas,
                eps=self.engram_offload_eps,
                weight_decay=0.0,
            )
            if opt_profile:
                profile_end_event(opt_pairs, "offload_start_adam", event)
                profile_add_wall(opt_wall_times, "offload_start_adam", wall_start)

        # Step optimizer with do_adam flag
        wall_start = time.perf_counter()
        event = profile_start_event(opt_profile)
        self.optimizer.step(do_adam=do_adam, adam_every_step_labels=self.adam_every_step_labels)
        if opt_profile:
            profile_end_event(opt_pairs, "base_optimizer", event)
            profile_add_wall(opt_wall_times, "base_optimizer", wall_start)
        if do_adam and self.sparse_adam is not None and self.sparse_adam_param.grad is not None:
            wall_start = time.perf_counter()
            event = profile_start_event(opt_profile)
            sparse_grad = self.sparse_adam_param.grad
            self.sparse_adam.param_groups[0]["lr"] = self.sparse_adam.defaults["lr"] * step_lr
            self.sparse_adam.step()
            if rom_sparse_sanitize:
                with torch.no_grad():
                    grad = sparse_grad.coalesce()
                    idx = grad.indices()[0]
                    if idx.numel() > 0:
                        rows = self.sparse_adam_param.index_select(0, idx)
                        rows = torch.nan_to_num(rows, nan=0.0, posinf=0.0, neginf=0.0)
                        self.sparse_adam_param.index_copy_(0, idx, rows)
                        state = self.sparse_adam.state.get(self.sparse_adam_param, {})
                        for key in ("exp_avg", "exp_avg_sq"):
                            value = state.get(key)
                            if isinstance(value, torch.Tensor):
                                state_rows = value.index_select(0, idx)
                                state_rows = torch.nan_to_num(state_rows, nan=0.0, posinf=0.0, neginf=0.0)
                                value.index_copy_(0, idx, state_rows)
            self.sparse_adam.zero_grad(set_to_none=True)
            if opt_profile:
                profile_end_event(opt_pairs, "bigram_sparse_adam", event)
                profile_add_wall(opt_wall_times, "bigram_sparse_adam", wall_start)
        if do_adam and self.sparse_sgd is not None and self.sparse_sgd_param.grad is not None:
            wall_start = time.perf_counter()
            event = profile_start_event(opt_profile)
            self.sparse_sgd.param_groups[0]["lr"] = self.sparse_sgd.defaults["lr"] * step_lr
            self.sparse_sgd.step()
            self.sparse_sgd.zero_grad(set_to_none=True)
            if opt_profile:
                profile_end_event(opt_pairs, "bigram_sparse_sgd", event)
                profile_add_wall(opt_wall_times, "bigram_sparse_sgd", wall_start)
        if self.sparse_normwrite is not None and self.sparse_normwrite_param.grad is not None:
            wall_start = time.perf_counter()
            event = profile_start_event(opt_profile)
            self.sparse_normwrite.step()
            self.sparse_normwrite.zero_grad(set_to_none=True)
            if opt_profile:
                profile_end_event(opt_pairs, "bigram_sparse_normwrite", event)
                profile_add_wall(opt_wall_times, "bigram_sparse_normwrite", wall_start)
        if self.sparse_recovered_module is not None:
            wall_start = time.perf_counter()
            event = profile_start_event(opt_profile)
            self.sparse_recovered_module.apply_recovered_normwrite(rom_state_row_rms_cap)
            if self.sparse_recovered_param is not None:
                self.sparse_recovered_param.grad = None
            if opt_profile:
                profile_end_event(opt_pairs, "bigram_sparse_recovered_normwrite", event)
                profile_add_wall(opt_wall_times, "bigram_sparse_recovered_normwrite", wall_start)
        engram_sparse_do_adam = do_adam or engram_adam_every_step
        if engram_sparse_do_adam and self.engram_sparse_adam is not None and self.engram_sparse_adam_param.grad is not None:
            param = self.engram_sparse_adam_param
            wall_start = time.perf_counter()
            event = profile_start_event(opt_profile)
            raw_grad_idx = param.grad._indices()[0] if engram_sparse_batch_freq_norm else None
            grad = coalesce_row_sparse_grad(param.grad)
            if engram_sparse_batch_freq_norm and raw_grad_idx is not None:
                idx = grad.indices()[0]
                if raw_grad_idx.numel() != idx.numel():
                    raw_unique, raw_counts = torch.unique(raw_grad_idx, sorted=True, return_counts=True)
                    row_counts = raw_counts.to(dtype=torch.float32, device=grad.device).index_select(0, torch.searchsorted(raw_unique, idx))
                    grad_values = grad.values().float().div_(row_counts.unsqueeze(1).clamp_min(1.0))
                    param.grad = _sparse_coo_tensor_coalesced(
                        grad.indices(),
                        grad_values.to(dtype=grad.dtype),
                        grad.shape,
                        device=grad.device,
                        dtype=grad.dtype,
                    )
                    grad = param.grad
            if engram_sparse_sanitize:
                grad_values = grad.values().float()
                sanitize_sparse_grad_values_(grad_values)
                param.grad = _sparse_coo_tensor_coalesced(
                    grad.indices(),
                    grad_values.to(dtype=grad.dtype),
                    grad.shape,
                    device=grad.device,
                    dtype=grad.dtype,
                )
                grad = param.grad
            if opt_profile:
                profile_end_event(opt_pairs, "engram_grad_coalesce", event)
                profile_add_wall(opt_wall_times, "engram_grad_coalesce", wall_start)
            metric_rows = None
            metric_before = None
            metric_grad_rms = None
            if (
                (not engram_sparse_vector_adam)
                and (not engram_sparse_scalar_adam)
                and (not engram_sparse_hit_lr)
                and (not engram_sparse_fal)
                and (not engram_sparse_ifal)
                and engram_sparse_adam_tail_steps <= 0
                and engram_update_metrics
                and engram_update_metrics_every > 0
            ):
                state = self.engram_sparse_adam.state[param]
                next_adam_step = int(state.get("step", 0)) + 1
                if next_adam_step % engram_update_metrics_every == 0:
                    wall_start = time.perf_counter()
                    event = profile_start_event(opt_profile)
                    metric_rows = grad.indices()[0]
                    metric_before = param.index_select(0, metric_rows).float()
                    metric_grad_rms = rms_no_alloc(grad.values().float())
                    if opt_profile:
                        profile_end_event(opt_pairs, "engram_metric_before", event)
                        profile_add_wall(opt_wall_times, "engram_metric_before", wall_start)
            self.engram_sparse_adam.param_groups[0]["lr"] = self.engram_sparse_adam.defaults["lr"] * engram_step_lr
            wall_start = time.perf_counter()
            event = profile_start_event(opt_profile)
            self.engram_sparse_adam.step()
            for _ in range(engram_sparse_extra_steps):
                self.engram_sparse_adam.step()
            custom_sparse_row_optimizer = (
                engram_sparse_vector_adam
                or engram_sparse_scalar_adam
                or engram_sparse_row_adagrad
                or engram_sparse_hit_lr
                or engram_sparse_fal
                or engram_sparse_ifal
                or engram_sparse_adam_tail_steps > 0
            )
            needs_post_row_cleanup = (
                engram_sparse_row_rms_cap > 0
                or engram_sparse_row_rms_floor > 0
                or engram_sparse_row_rms_norm > 0
                or engram_sparse_row_decay > 0
                or engram_sparse_param_clamp > 0
                or (engram_sparse_sanitize and not custom_sparse_row_optimizer)
            )
            if needs_post_row_cleanup:
                touched_rows = grad.indices()[0]
                if touched_rows.numel() > 0:
                    with torch.no_grad():
                        touched_rows = torch.unique(touched_rows)
                        if engram_sparse_row_rms_floor > 0 and engram_sparse_row_rms_floor_hit_max > 0:
                            state = self.engram_sparse_adam.state.get(param, {})
                            hit_count = state.get("hit_count")
                            if isinstance(hit_count, torch.Tensor):
                                touched_hits = hit_count.index_select(0, touched_rows)
                                touched_rows = touched_rows[touched_hits <= engram_sparse_row_rms_floor_hit_max]
                        if touched_rows.numel() > 0:
                            sanitize_chunk = max(1, int(engram_touched_update_metrics_chunk))
                            for row_start in range(0, touched_rows.numel(), sanitize_chunk):
                                row_idx = touched_rows[row_start: row_start + sanitize_chunk]
                                param_rows = param.index_select(0, row_idx).float()
                                if engram_sparse_row_decay > 0:
                                    decay = max(0.0, 1.0 - self.engram_sparse_adam.param_groups[0]["lr"] * engram_sparse_row_decay)
                                    param_rows.mul_(decay)
                                if engram_sparse_sanitize:
                                    param_rows = torch.nan_to_num(param_rows, nan=0.0, posinf=0.0, neginf=0.0)
                                if engram_sparse_param_clamp > 0:
                                    param_rows.clamp_(-engram_sparse_param_clamp, engram_sparse_param_clamp)
                                if engram_sparse_row_rms_cap > 0:
                                    row_rms = row_rms_stable(param_rows).unsqueeze(1)
                                    scale = (engram_sparse_row_rms_cap / row_rms.clamp_min(1e-12)).clamp_max(1.0)
                                    param_rows.mul_(scale)
                                if engram_sparse_row_rms_floor > 0:
                                    row_rms = row_rms_stable(param_rows).unsqueeze(1)
                                    scale = (engram_sparse_row_rms_floor / row_rms.clamp_min(1e-12)).clamp_min(1.0)
                                    param_rows.mul_(scale)
                                if engram_sparse_row_rms_norm > 0:
                                    row_rms = row_rms_stable(param_rows).unsqueeze(1)
                                    param_rows.mul_(engram_sparse_row_rms_norm / row_rms.clamp_min(1e-12))
                                index_copy_cast_chunked_(param, 0, row_idx, param_rows)
            if opt_profile:
                profile_end_event(opt_pairs, "engram_sparse_adam", event)
                profile_add_wall(opt_wall_times, "engram_sparse_adam", wall_start)
            if engram_sparse_vector_adam or engram_sparse_scalar_adam or engram_sparse_row_adagrad or engram_sparse_hit_lr or engram_sparse_fal or engram_sparse_ifal or engram_sparse_adam_tail_steps > 0:
                self.engram_sparse_update_metrics = getattr(self.engram_sparse_adam, "last_update_metrics", {})
            elif metric_rows is not None and metric_before is not None and metric_grad_rms is not None:
                wall_start = time.perf_counter()
                event = profile_start_event(opt_profile)
                metric_after = param.index_select(0, metric_rows).float()
                metric_update = metric_after - metric_before
                update_rms = rms_no_alloc(metric_update)
                param_rms = rms_no_alloc(metric_before)
                grad_values = grad.values().float()
                table_numel = max(1, param.numel())
                self.engram_sparse_update_metrics = {
                    "adam_step": int(self.engram_sparse_adam.state[param].get("step", 0)),
                    "rows": int(metric_rows.numel()),
                    "touched_rows": int(metric_rows.numel()),
                    "table_rows": int(param.shape[0]),
                    "table_numel": int(param.numel()),
                    "grad_rms": float(metric_grad_rms.item()),
                    "update_rms": float(update_rms.item()),
                    "param_rms": float(param_rms.item()),
                    "touched_grad_rms": float(metric_grad_rms.item()),
                    "touched_update_rms": float(update_rms.item()),
                    "touched_param_rms": float(param_rms.item()),
                    "table_grad_rms": float(table_rms_no_alloc(grad_values, table_numel).item()),
                    "table_update_rms": float(table_rms_no_alloc(metric_update, table_numel).item()),
                    "lr": float(self.engram_sparse_adam.param_groups[0]["lr"]),
                    "step_size": float(self.engram_sparse_adam.param_groups[0]["lr"]),
                }
                if opt_profile:
                    profile_end_event(opt_pairs, "engram_metric_after", event)
                    profile_add_wall(opt_wall_times, "engram_metric_after", wall_start)
            wall_start = time.perf_counter()
            event = profile_start_event(opt_profile)
            self.engram_sparse_adam.zero_grad(set_to_none=True)
            if opt_profile:
                profile_end_event(opt_pairs, "engram_zero_grad", event)
                profile_add_wall(opt_wall_times, "engram_zero_grad", wall_start)
        if engram_bigram and engram_offload and not engram_offload_async_adam:
            wall_start = time.perf_counter()
            event = profile_start_event(opt_profile)
            base_model = self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model
            base_model.bigram_embed.step_offloaded_adam(
                do_adam=do_adam or engram_adam_every_step,
                lr=self.engram_offload_initial_lr * engram_step_lr * self.engram_offload_lr_mul,
                betas=self.engram_offload_betas,
                eps=self.engram_offload_eps,
                weight_decay=0.0,
            )
            if opt_profile:
                profile_end_event(opt_pairs, "offload_step_adam", event)
                profile_add_wall(opt_wall_times, "offload_step_adam", wall_start)
        if self.snoo is not None:
            wall_start = time.perf_counter()
            event = profile_start_event(opt_profile)
            self.snoo.step()
            if opt_profile:
                profile_end_event(opt_pairs, "snoo_outer", event)
                profile_add_wall(opt_wall_times, "snoo_outer", wall_start)
        # At split step: copy lm_head optimizer state to embed and mark as split
        if step == self.split_step and not disable_embed_split:
            if engram_bigram and engram_offload:
                base_model = self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model
                base_model.bigram_embed.wait_offloaded_adam()
            self.optimizer.copy_lm_state_to_embed()
        if opt_profile:
            profile_print("profile_optimizer", step, opt_pairs, opt_wall_times, 1000 * (time.perf_counter() - opt_wall_start))

    def format_engram_update_metrics(self) -> str:
        if not (engram_update_metrics and engram_bigram):
            return ""
        metrics = None
        if engram_offload:
            base_model = self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model
            if engram_offload_async_adam:
                base_model.bigram_embed.wait_offloaded_adam()
            metrics = getattr(base_model.bigram_embed, "last_update_metrics", None)
        elif engram_sparse_adam or engram_sparse_vector_adam or engram_sparse_scalar_adam or engram_sparse_row_adagrad:
            metrics = self.engram_sparse_update_metrics
        else:
            metrics = self.optimizer.last_update_metrics.get("bigram_embed.embedding")
        if not metrics:
            return ""

        grad_rms = float(metrics.get("grad_rms", 0.0))
        update_rms = float(metrics.get("update_rms", 0.0))
        param_rms = float(metrics.get("param_rms", 0.0))
        update_param = update_rms / max(param_rms, 1e-12)
        grad_param = grad_rms / max(param_rms, 1e-12)
        update_grad = update_rms / max(grad_rms, 1e-12)
        result = (
            f" engram_adam_step:{int(metrics.get('adam_step', 0))}"
            f" engram_rows:{int(metrics.get('rows', 0))}"
            f" engram_grad_rms:{grad_rms:.3e}"
            f" engram_update_rms:{update_rms:.3e}"
            f" engram_param_rms:{param_rms:.3e}"
            f" engram_update_param:{update_param:.3e}"
            f" engram_grad_param:{grad_param:.3e}"
            f" engram_update_grad:{update_grad:.3e}"
            f" engram_lr:{float(metrics.get('lr', 0.0)):.3e}"
        )
        if "touched_rows" in metrics:
            touched_grad_rms = float(metrics.get("touched_grad_rms", 0.0))
            touched_update_rms = float(metrics.get("touched_update_rms", 0.0))
            touched_param_rms = float(metrics.get("touched_param_rms", 0.0))
            table_rows = int(metrics.get("table_rows", 0))
            touched_fraction = int(metrics.get("touched_rows", 0)) / max(table_rows, 1)
            touched_update_param = touched_update_rms / max(touched_param_rms, 1e-12)
            touched_grad_param = touched_grad_rms / max(touched_param_rms, 1e-12)
            touched_update_grad = touched_update_rms / max(touched_grad_rms, 1e-12)
            result += (
                f" engram_touched_rows:{int(metrics.get('touched_rows', 0))}"
                f" engram_touched_fraction:{touched_fraction:.3e}"
                f" engram_touched_grad_rms:{touched_grad_rms:.3e}"
                f" engram_touched_update_rms:{touched_update_rms:.3e}"
                f" engram_touched_param_rms:{touched_param_rms:.3e}"
                f" engram_touched_update_param:{touched_update_param:.3e}"
                f" engram_touched_grad_param:{touched_grad_param:.3e}"
                f" engram_touched_update_grad:{touched_update_grad:.3e}"
            )
        if "active_rows" in metrics:
            result += f" engram_active_rows:{int(metrics.get('active_rows', 0))}"
        if "hit_lr_scale_mean" in metrics:
            result += (
                f" engram_hit_lr_scale_mean:{float(metrics.get('hit_lr_scale_mean', 0.0)):.3e}"
                f" engram_hit_lr_scale_min:{float(metrics.get('hit_lr_scale_min', 0.0)):.3e}"
                f" engram_hit_lr_scale_max:{float(metrics.get('hit_lr_scale_max', 0.0)):.3e}"
                f" engram_hit_lr_blend:{float(metrics.get('hit_lr_blend', 1.0)):.3e}"
                f" engram_hit_count_mean:{float(metrics.get('hit_count_mean', 0.0)):.3e}"
                f" engram_hit_count_max:{float(metrics.get('hit_count_max', 0.0)):.3e}"
            )
        if "table_grad_rms" in metrics and "table_update_rms" in metrics:
            table_grad_rms = float(metrics.get("table_grad_rms", 0.0))
            table_update_rms = float(metrics.get("table_update_rms", 0.0))
            table_update_grad = table_update_rms / max(table_grad_rms, 1e-12)
            result += (
                f" engram_table_rows:{int(metrics.get('table_rows', 0))}"
                f" engram_table_numel:{int(metrics.get('table_numel', 0))}"
                f" engram_table_grad_rms:{table_grad_rms:.3e}"
                f" engram_table_update_rms:{table_update_rms:.3e}"
                f" engram_table_update_grad:{table_update_grad:.3e}"
            )
        return result

    def format_engram_mhc_metrics(self) -> str:
        base_model = self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model
        result = ""
        if engram_head_dropout > 0 or engram_head_dropout_schedule_steps > 0:
            result += f" engram_head_dropout_current:{engram_head_dropout_current:.3e}"
        if engram_output_dropout > 0 or engram_output_dropout_schedule_steps > 0:
            result += f" engram_output_dropout_current:{engram_output_dropout_current:.3e}"
        if engram_ngram_read_scales:
            scales = torch.tensor(engram_ngram_read_scales, dtype=torch.float32)
            if engram_ngram_read_scale_schedule_steps > 0:
                final = torch.tensor(engram_ngram_read_scales_final, dtype=torch.float32)
                progress = min(max(float(rom_debug_nan_current_step) / float(engram_ngram_read_scale_schedule_steps), 0.0), 1.0)
                scales = scales + progress * (final - scales)
            if engram_ngram_read_scale_norm:
                scales = scales / scales.square().mean().sqrt().clamp_min(1e-6)
            for ngram_idx, scale in enumerate(scales.tolist(), start=2):
                result += f" engram_ngram_read_scale_n{ngram_idx}:{scale:.3e}"
        if engram_read_hit_scale_exponent != 0 and engram_bigram and getattr(base_model.bigram_embed, "last_read_hit_scale", None) is not None:
            read_scale = base_model.bigram_embed.last_read_hit_scale.detach().float()
            result += (
                f" engram_read_hit_scale_mean:{read_scale.mean().item():.3e}"
                f" engram_read_hit_scale_min:{read_scale.min().item():.3e}"
                f" engram_read_hit_scale_max:{read_scale.max().item():.3e}"
            )
        if engram_hot_split and engram_bigram and getattr(base_model.bigram_embed, "last_hot_split_metrics", None):
            hot_split_metrics = base_model.bigram_embed.last_hot_split_metrics
            result += (
                f" engram_hot_split_frac:{float(hot_split_metrics.get('frac', 0.0)):.3e}"
                f" engram_hot_split_count:{int(hot_split_metrics.get('count', 0))}"
                f" engram_hot_split_aux_scale:{float(hot_split_metrics.get('aux_scale', 0.0)):.3e}"
                f" engram_hot_split_aux_slots:{int(hot_split_metrics.get('aux_slots', 1))}"
                f" engram_hot_split_mean_hits:{float(hot_split_metrics.get('mean_hits', 0.0)):.3e}"
                f" engram_hot_split_hot_mean_hits:{float(hot_split_metrics.get('hot_mean_hits', 0.0)):.3e}"
                f" engram_hot_split_aux_same_frac:{float(hot_split_metrics.get('aux_same_frac', 0.0)):.3e}"
                f" engram_hot_split_aux_unique_rows:{int(hot_split_metrics.get('aux_unique_rows', 0))}"
                f" engram_hot_split_aux_raw_rows:{int(hot_split_metrics.get('aux_raw_rows', 0))}"
                f" engram_hot_split_aux_unique_frac:{float(hot_split_metrics.get('aux_unique_frac', 0.0)):.3e}"
            )
        if engram_bigram and (engram_head_mix or engram_layer_head_mix):
            bigram_embed = getattr(base_model, "bigram_embed", None)
            if bigram_embed is not None and getattr(bigram_embed, "head_mix_logits", None) is not None:
                weights = F.softmax(bigram_embed.head_mix_logits.detach().float(), dim=-1)
                for head_idx, weight in enumerate(weights):
                    result += f" engram_head_mix_h{head_idx}:{weight.item():.3e}"
                delta_logits = getattr(bigram_embed, "layer_head_mix_delta_logits", None)
                if delta_logits is not None:
                    global_logits = bigram_embed.head_mix_logits.detach().float()
                    for layer_pos, layer_id in enumerate(getattr(bigram_embed, "layer_head_mix_ids", ())):
                        if layer_pos < delta_logits.size(0):
                            layer_weights = F.softmax(global_logits + delta_logits[layer_pos].detach().float(), dim=-1)
                            for head_idx, weight in enumerate(layer_weights):
                                result += f" engram_head_mix_l{layer_id}_h{head_idx}:{weight.item():.3e}"
            elif bigram_embed is not None and getattr(bigram_embed, "layer_head_mix_logits", None) is not None:
                weights = F.softmax(bigram_embed.layer_head_mix_logits.detach().float(), dim=-1)
                for layer_pos, layer_id in enumerate(getattr(bigram_embed, "layer_head_mix_ids", ())):
                    if layer_pos < weights.size(0):
                        for head_idx, weight in enumerate(weights[layer_pos]):
                            result += f" engram_head_mix_l{layer_id}_h{head_idx}:{weight.item():.3e}"
        if engram_bigram and (engram_sketch_slot_mix or engram_sketch_combine_mix):
            bigram_embed = getattr(base_model, "bigram_embed", None)
            weights = bigram_embed.sketch_slot_mix_weights() if bigram_embed is not None else None
            if weights is not None:
                for head_idx in range(weights.size(0)):
                    for slot_idx in range(weights.size(1)):
                        result += f" engram_slot_mix_h{head_idx}_s{slot_idx}:{weights[head_idx, slot_idx].item():.3e}"
        if engram_bigram and engram_sketch_aux_learned_scale:
            bigram_embed = getattr(base_model, "bigram_embed", None)
            scale = bigram_embed.sketch_aux_scale_snapshot() if bigram_embed is not None else None
            if scale is not None:
                result += f" engram_sketch_aux_learned_scale:{scale.item():.3e}"
        if engram_bigram and engram_layer_readout_delta_learned_scale:
            bigram_embed = getattr(base_model, "bigram_embed", None)
            scales = bigram_embed.layer_readout_delta_scales_snapshot() if bigram_embed is not None else None
            if scales is not None:
                for layer_pos, layer_id in enumerate(getattr(bigram_embed, "layer_readout_delta_ids", ())):
                    if layer_pos < scales.size(0):
                        result += (
                            f" engram_layerdelta_l{layer_id}_value_scale:{scales[layer_pos, 0].item():.3e}"
                            f" engram_layerdelta_l{layer_id}_key_scale:{scales[layer_pos, 1].item():.3e}"
                        )
        if engram_shadow_grad and engram_bigram and getattr(base_model.bigram_embed, "last_shadow_metrics", None):
            shadow_metrics = base_model.bigram_embed.last_shadow_metrics
            result += (
                f" engram_shadow_rows:{int(shadow_metrics.get('rows', 0))}"
                f" engram_shadow_kept_rows:{int(shadow_metrics.get('kept_rows', shadow_metrics.get('rows', 0)))}"
                f" engram_shadow_hit_mean:{float(shadow_metrics.get('hit_mean', 0.0)):.3e}"
                f" engram_shadow_grad_rms:{float(shadow_metrics.get('grad_rms', 0.0)):.3e}"
                f" engram_shadow_target_rms:{float(shadow_metrics.get('target_rms', 0.0)):.3e}"
                f" engram_shadow_update_rms:{float(shadow_metrics.get('update_rms', 0.0)):.3e}"
                f" engram_shadow_row_rms:{float(shadow_metrics.get('row_rms', 0.0)):.3e}"
            )
        if engram_mhc:
            if base_model.mhc_logit is not None:
                gate = torch.sigmoid(base_model.mhc_logit.detach().float())
                result += (
                    f" mhc_gate_mean:{gate.mean().item():.3e}"
                    f" mhc_gate_min:{gate.min().item():.3e}"
                    f" mhc_gate_max:{gate.max().item():.3e}"
                )
                for layer_id in getattr(base_model, "bigram_layer_ids", ()):
                    if 0 <= layer_id < gate.numel():
                        result += f" mhc_gate_l{layer_id}:{gate[layer_id].item():.3e}"
            if getattr(base_model, "mhc_router", None) is not None:
                result += f" mhc_router_rms:{rms_no_alloc(base_model.mhc_router.weight.detach().float()).item():.3e}"
            if getattr(base_model, "mhc_identity_router", None) is not None:
                result += f" mhc_identity_router_rms:{rms_no_alloc(base_model.mhc_identity_router.weight.detach().float()).item():.3e}"
        if engram_attnres_merge and getattr(base_model, "engram_attnres_query", None) is not None:
            query = base_model.engram_attnres_query.detach().float()
            result += (
                f" engram_attnres_query_rms:{rms_no_alloc(query).item():.3e}"
                f" engram_attnres_gain:{engram_attnres_merge_gain_current:.3e}"
            )
            direct = getattr(base_model, "engram_direct_resid", None)
            if direct is not None:
                direct_f = direct.detach().float()
                result += (
                    f" engram_direct_resid_mean:{direct_f.mean().item():.3e}"
                    f" engram_direct_resid_l2:{direct_f[2].item():.3e}"
                    f" engram_direct_resid_l8:{direct_f[8].item():.3e}"
                )
            layer_gain_log = getattr(base_model, "engram_attnres_layer_gain_log", None)
            if layer_gain_log is not None:
                layer_gain = layer_gain_log.detach().float().exp()
                result += (
                    f" engram_attnres_layer_gain_mean:{layer_gain.mean().item():.3e}"
                    f" engram_attnres_layer_gain_l2:{layer_gain[2].item():.3e}"
                    f" engram_attnres_layer_gain_l8:{layer_gain[8].item():.3e}"
                )
            stats = getattr(base_model, "last_engram_attnres_stats", {})
            if stats:
                weights = torch.cat([v.float().flatten() for v in stats.values()])
                result += (
                    f" engram_attnres_p_mean:{weights.mean().item():.3e}"
                    f" engram_attnres_p_min:{weights.min().item():.3e}"
                    f" engram_attnres_p_max:{weights.max().item():.3e}"
                )
                for layer_id in getattr(base_model, "bigram_layer_ids", ()):
                    if layer_id in stats:
                        result += f" engram_attnres_p_l{layer_id}:{stats[layer_id].float().mean().item():.3e}"
            extra_stats = getattr(base_model, "last_engram_attnres_extra_stats", {})
            if extra_stats:
                extra_weights = torch.cat([v.float().flatten() for v in extra_stats.values()])
                result += f" engram_attnres_extra_p_mean:{extra_weights.mean().item():.3e}"
                for layer_id in sorted(extra_stats):
                    result += f" engram_attnres_extra_p_l{layer_id}:{extra_stats[layer_id].float().mean().item():.3e}"
            cos_stats = getattr(base_model, "last_engram_attnres_cos_stats", {})
            if cos_stats:
                cos_values = torch.cat([v.float().flatten() for v in cos_stats.values()])
                result += (
                    f" engram_attnres_cos_mean:{cos_values.mean().item():.3e}"
                    f" engram_attnres_cos_min:{cos_values.min().item():.3e}"
                    f" engram_attnres_cos_max:{cos_values.max().item():.3e}"
                )
                for layer_id in getattr(base_model, "bigram_layer_ids", ()):
                    if layer_id in cos_stats:
                        result += f" engram_attnres_cos_l{layer_id}:{cos_stats[layer_id].float().mean().item():.3e}"
            rms_ratio_stats = getattr(base_model, "last_engram_attnres_rms_ratio_stats", {})
            if rms_ratio_stats:
                ratio_values = torch.cat([v.float().flatten() for v in rms_ratio_stats.values()])
                result += (
                    f" engram_attnres_rms_ratio_mean:{ratio_values.mean().item():.3e}"
                    f" engram_attnres_rms_ratio_min:{ratio_values.min().item():.3e}"
                    f" engram_attnres_rms_ratio_max:{ratio_values.max().item():.3e}"
                )
                for layer_id in getattr(base_model, "bigram_layer_ids", ()):
                    if layer_id in rms_ratio_stats:
                        result += f" engram_attnres_rms_ratio_l{layer_id}:{rms_ratio_stats[layer_id].float().mean().item():.3e}"
        if engram_cache_learned_cfg and base_model.cache_cfg_lambdas is not None:
            cfg = base_model.cache_cfg_lambdas.detach().float()
            result += (
                f" cache_cfg_mean:{cfg.mean().item():.3e}"
                f" cache_cfg_min:{cfg.min().item():.3e}"
                f" cache_cfg_max:{cfg.max().item():.3e}"
            )
            if 0 <= engram_cache_recon_source_layer < cfg.numel():
                result += f" cache_cfg_l{engram_cache_recon_source_layer}:{cfg[engram_cache_recon_source_layer].item():.3e}"
        if engram_hit_hist and engram_bigram and getattr(base_model.bigram_embed, "hit_hist", None) is not None:
            summary = base_model.bigram_embed.hit_hist_summary()
            if summary:
                result += (
                    f" engram_hit_frac_ever:{float(summary['frac_ever_hit']):.3e}"
                    f" engram_hit_frac_gt1:{float(summary['frac_hit_gt1']):.3e}"
                    f" engram_hit_total:{int(summary['total_hits'])}"
                    f" engram_hit_max:{int(summary['max_hits'])}"
                    f" engram_hit_mean_touched:{float(summary['mean_hits_per_touched_row']):.3e}"
                )
        return result

    def reset(self, state=None):
        if state is not None:
            if isinstance(state, dict) and "optimizer" in state:
                self.optimizer.load_state_dict(state["optimizer"])
                if self.sparse_adam is not None and state.get("sparse_adam") is not None:
                    self.sparse_adam.load_state_dict(state["sparse_adam"])
                if self.sparse_sgd is not None and state.get("sparse_sgd") is not None:
                    self.sparse_sgd.load_state_dict(state["sparse_sgd"])
                if self.engram_sparse_adam is not None and state.get("engram_sparse_adam") is not None:
                    self.engram_sparse_adam.load_state_dict(state["engram_sparse_adam"])
                if self.snoo is not None and state.get("snoo") is not None:
                    self.snoo.load_state_dict(state["snoo"])
            else:
                self.optimizer.load_state_dict(state)

        # Reset NorMuon momentum buffers and split_embed state
        self.optimizer.reset()
        if state is not None and engram_bigram and engram_offload:
            base_model = self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model
            base_model.bigram_embed.wait_offloaded_adam()
            base_model.bigram_embed.reset_offloaded_embedding()
        self.engram_sparse_update_metrics.clear()

        stage, _ = training_schedule.lookup(0)
        self.ws_short, self.ws_long = stage.window_sizes
        self.batch_size = stage.batch_size
        self.train_max_seq_len = stage.train_max_seq_len
        self.model.yarn.reset()
        self.model.yarn_paired_head.reset()
        if _sparse_comms_active():
            self.row_update_mask = np.zeros(self.bigram_param.shape[0], dtype=np.uint8)
            self.sparse_counts_state = None
            # buffer we use for fast GPU uploads of send indexes
            self.send_idxes_buffer = torch.empty(self.bigram_param.shape[0], dtype=torch.int32, pin_memory=True)


    def get_state(self):
        state = {"optimizer": self.optimizer.state_dict()}
        if self.sparse_adam is not None:
            state["sparse_adam"] = self.sparse_adam.state_dict()
        if self.sparse_sgd is not None:
            state["sparse_sgd"] = self.sparse_sgd.state_dict()
        if self.sparse_normwrite is not None:
            state["sparse_normwrite"] = self.sparse_normwrite.state_dict()
        if self.engram_sparse_adam is not None:
            state["engram_sparse_adam"] = self.engram_sparse_adam.state_dict()
        if self.snoo is not None:
            state["snoo"] = self.snoo.state_dict()
        return copy.deepcopy(state)

    def sparse_index_update(self, step, bigram_indexes):
        if not _sparse_comms_active():
            return

        self.row_update_mask[bigram_indexes] = 1

        if self._is_adam_step(step):
            with torch.no_grad():
                bigram_idx_np = np.flatnonzero(self.row_update_mask).astype(np.int32)
                send_idxes, send_counts, recv_counts, recv_counts_fut = sparse_comms_start(
                    bigram_idx_np, self.bigram_param.shape[0], rank, world_size, self.send_idxes_buffer
                )
                self.sparse_counts_state = (send_idxes, send_counts, recv_counts, recv_counts_fut)

    def sparse_index_share(self, step):
        if not _sparse_comms_active() or not self._is_adam_step(step):
            return

        send_idxes, send_counts, recv_counts, recv_counts_fut = self.sparse_counts_state
        self.sparse_counts_state = None

        recv_counts_fut.wait()
        recv_idxes, sparse_state, idxes_fut = sparse_comms_share_indexes(send_idxes, send_counts, recv_counts)
        self.optimizer._reduce_futures[self.bigram_param] = [idxes_fut, recv_idxes]
        self.optimizer._sparse_async_data[self.bigram_param] = sparse_state

        self.row_update_mask.fill(0)


        

# -----------------------------------------------------------------------------
# int main

# begin logging
logfile = None
if master_process:
    run_id = args.run_id
    os.makedirs("logs", exist_ok=True)
    logfile = f"logs/{run_id}.txt"
    print(logfile)
def print0(s, console=False):
    if master_process:
        with open(logfile, "a") as f:
            if console:
                print(s)
            print(s, file=f)

wandb_run = None

def _parse_metric_suffix(text: str, prefix: str = "") -> dict[str, float]:
    metrics: dict[str, float] = {}
    for part in text.split():
        if ":" not in part:
            continue
        key, raw_value = part.split(":", 1)
        try:
            value = float(raw_value)
        except ValueError:
            continue
        metrics[f"{prefix}{key}"] = value
    return metrics

def _wandb_config() -> dict[str, object]:
    cfg: dict[str, object] = {
        field.name: getattr(args, field.name)
        for field in fields(Hyperparameters)
        if not field.name.endswith("_files")
    }
    cfg.update({
        "world_size": world_size,
        "grad_accum_steps": grad_accum_steps,
        "device_capability": f"sm{device_capability[0]}{device_capability[1]}",
        "compile_model": compile_model,
        "compile_backend": compile_backend or "inductor",
        "compile_layer_modules": compile_layer_modules,
        "compile_dense_layer_body": compile_dense_layer_body,
        "final_smear_mtp": final_smear_mtp,
        "final_smear_mtp_init": final_smear_mtp_init,
        "snoo_outer": snoo_outer,
        "snoo_lr": snoo_lr,
        "snoo_momentum": snoo_momentum,
        "snoo_k": snoo_k,
        "snoo_include_engram": snoo_include_engram,
        "normuon_update_smoothing": normuon_update_smoothing,
        "adam_embed_vector_adam": adam_embed_vector_adam,
        "adam_embed_scalar_adam": adam_embed_scalar_adam,
        "rom_bigram": rom_bigram,
        "rom_single_token": rom_single_token,
        "rom_token": rom_token,
        "rom_layers": ",".join(map(str, rom_layers)),
        "rom_output_scale": rom_output_scale,
        "rom_normalize_readout": rom_normalize_readout,
        "rom_readout_rms": rom_readout_rms,
        "rom_read_mlp": rom_read_mlp,
        "rom_read_mlp_hidden_mult": rom_read_mlp_hidden_mult,
        "rom_read_mlp_lr_mul": rom_read_mlp_lr_mul,
        "rom_ema_smooth": rom_ema_smooth,
        "rom_ema_alpha": rom_ema_alpha,
        "rom_ema_kernel": rom_ema_kernel,
        "rom_state_init_std": rom_state_init_std,
        "rom_state_diag_init": rom_state_diag_init,
        "rom_state_frob_norm": rom_state_frob_norm,
        "rom_state_row_rms_cap": rom_state_row_rms_cap,
        "rom_state_normwrite": rom_state_normwrite,
        "rom_state_recovered_normwrite": rom_state_recovered_normwrite,
        "rom_state_write_rms": rom_state_write_rms,
        "rom_state_hit_rms_low": rom_state_hit_rms_low,
        "rom_state_hit_rms_high": rom_state_hit_rms_high,
        "rom_state_hit_rms_knee": rom_state_hit_rms_knee,
        "rom_sparse_adam_lr_mul": rom_sparse_adam_lr_mul,
        "rom_sparse_adam_beta2": rom_sparse_adam_beta2,
        "rom_sparse_row_scalar_adam": rom_sparse_row_scalar_adam,
        "engram_bigram": engram_bigram,
        "engram_dim": engram_dim,
        "engram_store_dim": engram_store_dim,
        "engram_heads": engram_heads,
        "engram_max_ngram": engram_max_ngram,
        "engram_ngram_row_factors": ",".join(map(str, engram_ngram_row_factors)),
        "engram_ngram_read_scales": ",".join(map(str, engram_ngram_read_scales)),
        "engram_ngram_read_scales_final": ",".join(map(str, engram_ngram_read_scales_final)),
        "engram_ngram_read_scale_schedule_steps": engram_ngram_read_scale_schedule_steps,
        "engram_ngram_read_scale_norm": engram_ngram_read_scale_norm,
        "engram_avalanche_hash": engram_avalanche_hash,
        "engram_layer_hashes": engram_layer_hashes,
        "engram_layer_readouts": engram_layer_readouts,
        "engram_layer_readout_delta": engram_layer_readout_delta,
        "engram_layer_readout_delta_scale": engram_layer_readout_delta_scale,
        "engram_layer_readout_delta_scale_final": engram_layer_readout_delta_scale_final,
        "engram_layer_readout_delta_scale_schedule_steps": engram_layer_readout_delta_scale_schedule_steps,
        "engram_layer_readout_delta_scale_schedule_start": engram_layer_readout_delta_scale_schedule_start,
        "engram_layer_readout_delta_learned_scale": engram_layer_readout_delta_learned_scale,
        "engram_layer_readout_delta_learned_scale_init": engram_layer_readout_delta_learned_scale_init,
        "engram_layer_readout_delta_learned_scale_max": engram_layer_readout_delta_learned_scale_max,
        "engram_layer_readout_delta_learned_scale_lr_mul": engram_layer_readout_delta_learned_scale_lr_mul,
        "engram_layer_partitions": engram_layer_partitions,
        "engram_layer_signs": engram_layer_signs,
        "engram_layer_row_signs": engram_layer_row_signs,
        "engram_layer_row_signs_aux_only": engram_layer_row_signs_aux_only,
        "engram_layer_row_sign_scale": engram_layer_row_sign_scale,
        "engram_layer_row_sign_scale_final": engram_layer_row_sign_scale_final,
        "engram_layer_row_sign_scale_schedule_steps": engram_layer_row_sign_scale_schedule_steps,
        "engram_layer_row_sign_scale_schedule_start": engram_layer_row_sign_scale_schedule_start,
        "engram_layer_partition_groups": engram_layer_partition_groups,
        "engram_per_head": engram_per_head,
        "engram_canonicalize": engram_canonicalize,
        "engram_latent": engram_latent,
        "engram_latent_quantizer": engram_latent_quantizer,
        "engram_latent_fsq_levels": engram_latent_fsq_levels_raw,
        "engram_latent_fsq_eps": engram_latent_fsq_eps,
        "engram_latent_bsq_bits": engram_latent_bsq_bits,
        "engram_latent_rows_per_head": engram_latent_rows_per_head,
        "engram_latent_input_scale": engram_latent_input_scale,
        "engram_latent_ste_scale": engram_latent_ste_scale,
        "engram_latent_mix_ngram": engram_latent_mix_ngram,
        "engram_latent_aux_readout": engram_latent_aux_readout,
        "engram_latent_aux_scale": engram_latent_aux_scale,
        "engram_latent_aux_scale_final": engram_latent_aux_scale_final,
        "engram_latent_aux_scale_schedule_steps": engram_latent_aux_scale_schedule_steps,
        "engram_latent_aux_scale_schedule_start": engram_latent_aux_scale_schedule_start,
        "engram_latent_pkm_subkeys": engram_latent_pkm_subkeys,
        "engram_latent_pkm_key_dim": engram_latent_pkm_key_dim,
        "engram_latent_pkm_topk": engram_latent_pkm_topk,
        "engram_normalize_readout": engram_normalize_readout,
        "engram_normalize_memory_heads": engram_normalize_memory_heads,
        "engram_detach_key_memory": engram_detach_key_memory,
        "engram_detach_value_memory": engram_detach_value_memory,
        "engram_detach_memory_layers": ",".join(map(str, engram_detach_memory_layers)),
        "engram_fixed_half_gate": engram_fixed_half_gate,
        "engram_head_mix": engram_head_mix,
        "engram_layer_head_mix": engram_layer_head_mix,
        "engram_layer_head_mix_delta": engram_layer_head_mix_delta,
        "engram_head_mix_freeze": engram_head_mix_freeze,
        "engram_sketch_k": engram_sketch_k,
        "engram_sketch_dim_signs": engram_sketch_dim_signs,
        "engram_sketch_dim_sign_mode": engram_sketch_dim_sign_mode,
        "engram_sketch_scalar_signs": engram_sketch_scalar_signs,
        "engram_sketch_scalar_sign_mode": engram_sketch_scalar_sign_mode,
        "engram_sketch_include_base": engram_sketch_include_base,
        "engram_sketch_aux_scale": engram_sketch_aux_scale,
        "engram_sketch_aux_scale_final": engram_sketch_aux_scale_final,
        "engram_sketch_aux_scale_schedule_steps": engram_sketch_aux_scale_schedule_steps,
        "engram_sketch_aux_scale_schedule_start": engram_sketch_aux_scale_schedule_start,
        "engram_sketch_aux_learned_scale": engram_sketch_aux_learned_scale,
        "engram_sketch_aux_learned_scale_init": engram_sketch_aux_learned_scale_init,
        "engram_sketch_aux_learned_scale_max": engram_sketch_aux_learned_scale_max,
        "engram_sketch_aux_learned_scale_lr_mul": engram_sketch_aux_learned_scale_lr_mul,
        "engram_sketch_slot_readout": engram_sketch_slot_readout,
        "engram_sketch_slot_attention": engram_sketch_slot_attention,
        "engram_sketch_slot_mix": engram_sketch_slot_mix,
        "engram_sketch_combine_mix": engram_sketch_combine_mix,
        "engram_sketch_combine_mix_mode": engram_sketch_combine_mix_mode,
        "engram_sketch_combine_mix_max_dev": engram_sketch_combine_mix_max_dev,
        "engram_sketch_mix_lr_mul": engram_sketch_mix_lr_mul,
        "engram_superpose_k": engram_superpose_k,
        "engram_superpose_include_base": engram_superpose_include_base,
        "engram_superpose_aux_scale": engram_superpose_aux_scale,
        "engram_superpose_aux_scale_final": engram_superpose_aux_scale_final,
        "engram_superpose_aux_scale_schedule_steps": engram_superpose_aux_scale_schedule_steps,
        "engram_superpose_aux_scale_schedule_start": engram_superpose_aux_scale_schedule_start,
        "engram_superpose_normalize": engram_superpose_normalize,
        "engram_manual_sparse_coalesce": engram_manual_sparse_coalesce,
        "engram_manual_sparse_coalesce_start": engram_manual_sparse_coalesce_start,
        "engram_head_dropout": engram_head_dropout,
        "engram_head_dropout_final": engram_head_dropout_final,
        "engram_head_dropout_schedule_steps": engram_head_dropout_schedule_steps,
        "engram_output_dropout": engram_output_dropout,
        "engram_output_dropout_final": engram_output_dropout_final,
        "engram_output_dropout_schedule_steps": engram_output_dropout_schedule_steps,
        "engram_output_dropout_schedule_start": engram_output_dropout_schedule_start,
        "engram_read_hit_scale_exponent": engram_read_hit_scale_exponent,
        "engram_read_hit_scale_offset": engram_read_hit_scale_offset,
        "engram_read_hit_scale_min": engram_read_hit_scale_min,
        "engram_read_hit_scale_max": engram_read_hit_scale_max,
        "engram_read_hit_scale_norm_mean": engram_read_hit_scale_norm_mean,
        "engram_hit_dropout": engram_hit_dropout,
        "engram_hit_dropout_final": engram_hit_dropout_final,
        "engram_hit_dropout_schedule_steps": engram_hit_dropout_schedule_steps,
        "engram_hit_dropout_schedule_start": engram_hit_dropout_schedule_start,
        "engram_hit_dropout_decay_final": engram_hit_dropout_decay_final,
        "engram_hit_dropout_decay_steps": engram_hit_dropout_decay_steps,
        "engram_hit_dropout_decay_start": engram_hit_dropout_decay_start,
        "engram_hit_dropout_min_hits": engram_hit_dropout_min_hits,
        "engram_hit_dropout_invert_scale": engram_hit_dropout_invert_scale,
        "engram_sketch_hit_hist_base_only": engram_sketch_hit_hist_base_only,
        "engram_hot_split": engram_hot_split,
        "engram_hot_split_value_only": engram_hot_split_value_only,
        "engram_hot_split_min_hits": engram_hot_split_min_hits,
        "engram_hot_split_aux_scale": engram_hot_split_aux_scale,
        "engram_hot_split_aux_scale_final": engram_hot_split_aux_scale_final,
        "engram_hot_split_aux_scale_schedule_steps": engram_hot_split_aux_scale_schedule_steps,
        "engram_hot_split_aux_scale_schedule_start": engram_hot_split_aux_scale_schedule_start,
        "engram_hot_split_aux_slots": engram_hot_split_aux_slots,
        "engram_hot_split_ramp_steps": engram_hot_split_ramp_steps,
        "engram_hot_split_detach_aux": engram_hot_split_detach_aux,
        "engram_hot_split_dedup_aux": engram_hot_split_dedup_aux,
        "engram_hot_split_train_only": engram_hot_split_train_only,
        "engram_hit_hist_kinds": ",".join(engram_hit_hist_kinds),
        "engram_init_std": engram_init_std,
        "engram_init_zero": engram_init_zero,
        "engram_freeze_memory": engram_freeze_memory,
        "engram_attnres_merge": engram_attnres_merge,
        "engram_attnres_merge_gain": engram_attnres_merge_gain,
        "engram_attnres_gain_warmup_steps": engram_attnres_gain_warmup_steps,
        "engram_attnres_direct_residual": engram_attnres_direct_residual,
        "engram_attnres_direct_init": engram_attnres_direct_init,
        "engram_attnres_direct_layers": engram_attnres_direct_layers_raw,
        "engram_attnres_layer_gain": engram_attnres_layer_gain,
        "engram_attnres_layer_gain_init": engram_attnres_layer_gain_init,
        "engram_attnres_extra_source_layer": engram_attnres_extra_source_layer,
        "engram_attnres_extra_target_layer": engram_attnres_extra_target_layer,
        "engram_attnres_extra_bias_init": engram_attnres_extra_bias_init,
        "engram_bank_attnres": engram_bank_attnres,
        "engram_mhc": engram_mhc,
        "engram_mhc_streams": engram_mhc_streams,
        "engram_sparse_adam": engram_sparse_adam,
        "engram_sparse_vector_adam": engram_sparse_vector_adam,
        "engram_sparse_scalar_adam": engram_sparse_scalar_adam,
        "engram_sparse_row_adagrad": engram_sparse_row_adagrad,
        "engram_sparse_adam_tail_scale": engram_sparse_adam_tail_scale,
        "engram_sparse_adam_beta1": engram_sparse_adam_beta1,
        "engram_sparse_adam_beta2": engram_sparse_adam_beta2,
        "engram_sparse_weight_decay": engram_sparse_weight_decay,
        "engram_sparse_hit_lr": engram_sparse_hit_lr,
        "engram_sparse_fal": engram_sparse_fal,
        "engram_sparse_ifal": engram_sparse_ifal,
        "engram_sparse_batch_freq_norm": engram_sparse_batch_freq_norm,
        "engram_sparse_hit_lr_exponent": engram_sparse_hit_lr_exponent,
        "engram_sparse_hit_lr_min": engram_sparse_hit_lr_min,
        "engram_sparse_hit_lr_max": engram_sparse_hit_lr_max,
        "engram_sparse_hit_lr_blend": engram_sparse_hit_lr_blend,
        "engram_sparse_hit_lr_blend_final": engram_sparse_hit_lr_blend_final,
        "engram_sparse_hit_lr_blend_schedule_steps": engram_sparse_hit_lr_blend_schedule_steps,
        "engram_sparse_hit_lr_blend_schedule_start": engram_sparse_hit_lr_blend_schedule_start,
        "engram_sparse_row_rms_cap": engram_sparse_row_rms_cap,
        "engram_sparse_row_rms_floor": engram_sparse_row_rms_floor,
        "engram_sparse_row_rms_floor_hit_max": engram_sparse_row_rms_floor_hit_max,
        "engram_sparse_row_rms_norm": engram_sparse_row_rms_norm,
        "engram_sparse_row_decay": engram_sparse_row_decay,
        "engram_sparse_grad_coalesce_hook": engram_sparse_grad_coalesce_hook,
        "engram_sparse_grad_coalesce_hook_start": engram_sparse_grad_coalesce_hook_start,
        "engram_static_gate": int(engram_static_gate),
        "engram_static_gate_init": engram_static_gate_init,
        "engram_lr_mul": engram_lr_mul,
        "engram_lr_floor": engram_lr_floor,
        "engram_adam_every_step": engram_adam_every_step,
        "engram_update_metrics": engram_update_metrics,
        "engram_hit_hist": engram_hit_hist,
        "engram_save_hit_hist": engram_save_hit_hist,
        "engram_mask_unhit_eval": engram_mask_unhit_eval,
        "engram_mask_unhit_eval_mode": engram_mask_unhit_eval_mode,
        "engram_mask_hit_min_eval": engram_mask_hit_min_eval,
        "engram_mask_hit_max_eval": engram_mask_hit_max_eval,
        "engram_mask_hit_invert_eval": engram_mask_hit_invert_eval,
        "wandb_histograms": wandb_histograms,
        "wandb_hist_every": wandb_hist_every,
        "wandb_hist_rows": wandb_hist_rows,
    })
    return cfg

def wandb_init_if_needed():
    global wandb_run
    if not (master_process and wandb_enabled):
        return
    try:
        import wandb
    except Exception as exc:
        print0(f"WANDB requested but import failed: {exc}", console=True)
        return
    os.makedirs(wandb_dir, exist_ok=True)
    kwargs = {
        "project": wandb_project,
        "entity": wandb_entity,
        "group": wandb_group,
        "name": wandb_name or args.run_id,
        "id": args.run_id,
        "resume": "allow",
        "tags": list(wandb_tags),
        "config": _wandb_config(),
        "dir": wandb_dir,
    }
    if wandb_mode is not None:
        kwargs["mode"] = wandb_mode
    wandb_run = wandb.init(**kwargs)
    if wandb_log_code:
        wandb_run.log_code(".")
    print0(f"WANDB run initialized: project={wandb_project} name={wandb_name or args.run_id} mode={wandb_mode or 'default'}", console=True)

def wandb_log(metrics: dict[str, object], step: int):
    if wandb_run is None:
        return
    clean = {}
    for key, value in metrics.items():
        if value is None:
            continue
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                continue
            value = value.detach().float().item()
        if isinstance(value, (int, float, bool)):
            clean[key] = value
        elif value.__class__.__module__.startswith("wandb"):
            clean[key] = value
    if clean:
        wandb_run.log(clean, step=step)

def _wandb_histogram(values):
    import wandb
    import numpy as np

    arr = np.asarray(values)
    if arr.size == 0:
        return None

    finite = np.isfinite(arr)
    if not bool(finite.any()):
        return None

    return wandb.Histogram(arr[finite])

def _add_engram_hit_bucket_stats(
    logs: dict[str, object],
    prefix: str,
    hits: Tensor,
    row_rms: Tensor,
    m_rms: Tensor | None = None,
    v_mean: Tensor | None = None,
    sqrtv_mean: Tensor | None = None,
    update_rms: Tensor | None = None,
) -> None:
    buckets = [
        ("hit_0", hits == 0),
        ("hit_1", hits == 1),
        ("hit_2_3", (hits >= 2) & (hits <= 3)),
        ("hit_4_15", (hits >= 4) & (hits <= 15)),
        ("hit_16_63", (hits >= 16) & (hits <= 63)),
        ("hit_64_255", (hits >= 64) & (hits <= 255)),
        ("hit_256_1023", (hits >= 256) & (hits <= 1023)),
        ("hit_ge_1024", hits >= 1024),
    ]
    total = max(1, int(hits.numel()))
    for name, mask in buckets:
        count = int(mask.sum().item())
        logs[f"{prefix}/{name}_count"] = count
        logs[f"{prefix}/{name}_frac"] = count / total
        if count == 0:
            continue
        logs[f"{prefix}/{name}_hit_mean"] = float(hits[mask].mean().item())
        logs[f"{prefix}/{name}_row_rms_mean"] = float(row_rms[mask].mean().item())
        if m_rms is not None:
            logs[f"{prefix}/{name}_m_rms_mean"] = float(m_rms[mask].mean().item())
        if v_mean is not None:
            logs[f"{prefix}/{name}_v_mean_mean"] = float(v_mean[mask].mean().item())
        if sqrtv_mean is not None:
            logs[f"{prefix}/{name}_sqrtv_mean_mean"] = float(sqrtv_mean[mask].mean().item())
        if update_rms is not None:
            logs[f"{prefix}/{name}_adam_update_rms_mean"] = float(update_rms[mask].mean().item())

def _fixed_sample_indices(num_rows: int, sample_rows: int, sample_seed: int, target_device: torch.device) -> Tensor:
    count = min(max(0, sample_rows), num_rows)
    if count <= 0:
        return torch.empty((0,), dtype=torch.long, device=target_device)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(sample_seed)
    idx = torch.randint(num_rows, (count,), generator=gen, dtype=torch.long)
    return idx.to(device=target_device, non_blocking=True)

def collect_engram_wandb_histograms(training_manager: "TrainingManager", *, step: int, phase: str) -> dict[str, object]:
    if wandb_run is None or not wandb_histograms or wandb_hist_rows <= 0:
        return {}
    if not (engram_bigram and (engram_sparse_adam or engram_sparse_vector_adam or engram_sparse_scalar_adam or engram_sparse_row_adagrad)):
        return {}
    param = training_manager.engram_sparse_adam_param
    opt = training_manager.engram_sparse_adam
    if param is None or opt is None or param not in opt.state:
        return {}
    state = opt.state[param]
    if not state:
        return {}
    def _maybe_hist(name: str, tensor: Tensor):
        if tensor.numel() == 0:
            return None
        finite_mask = torch.isfinite(tensor)
        finite_count = int(finite_mask.sum().item())
        total_count = int(tensor.numel())
        if finite_count != total_count:
            print0(f"ENGRAM_HIST_NAN phase={phase} step={step} metric={name} finite={finite_count}/{total_count}", console=True)
            if engram_hist_nan_assert:
                raise RuntimeError(f"ENGRAM_HIST_NAN phase={phase} step={step} metric={name} finite={finite_count}/{total_count}")
        if finite_count == 0:
            return None
        return _wandb_histogram(tensor[finite_mask].detach().cpu().numpy())

    with torch.no_grad():
        idx = _fixed_sample_indices(param.shape[0], wandb_hist_rows, wandb_hist_seed, param.device)
        if idx.numel() == 0:
            return {}
        weight_rows = param.index_select(0, idx).float()
        row_rms = row_rms_stable(weight_rows)
        logs: dict[str, object] = {
            f"{phase}/engram_row_rms_hist": _maybe_hist("row_rms", row_rms),
            f"{phase}/engram_row_rms_sample_mean": float(row_rms.mean().item()),
            f"{phase}/engram_row_rms_sample_p50": float(torch.quantile(row_rms, 0.50).item()),
            f"{phase}/engram_row_rms_sample_p90": float(torch.quantile(row_rms, 0.90).item()),
            f"{phase}/engram_row_rms_sample_p99": float(torch.quantile(row_rms, 0.99).item()),
        }
        m_rms = None
        v_mean = None
        sqrtv_mean = None
        update_rms = None
        exp_avg = state.get("exp_avg")
        exp_avg_sq = state.get("exp_avg_sq")
        if exp_avg is not None:
            m_rows = exp_avg.index_select(0, idx).float()
            m_rms = row_rms_stable(m_rows)
            logs.update({
                f"{phase}/engram_m_rms_hist": _maybe_hist("m_rms", m_rms),
                f"{phase}/engram_m_rms_sample_mean": float(m_rms.mean().item()),
                f"{phase}/engram_m_rms_sample_p99": float(torch.quantile(m_rms, 0.99).item()),
            })
        elif state.get("exp_avg_row") is not None:
            m_rows = state["exp_avg_row"].index_select(0, idx).float()
            m_rms = m_rows.abs()
            logs.update({
                f"{phase}/engram_m_rms_hist": _maybe_hist("m_rms", m_rms),
                f"{phase}/engram_m_rms_sample_mean": float(m_rms.mean().item()),
                f"{phase}/engram_m_rms_sample_p99": float(torch.quantile(m_rms, 0.99).item()),
            })
        if exp_avg_sq is not None:
            v_rows = exp_avg_sq.index_select(0, idx).float()
            v_mean = v_rows.mean(dim=1)
            sqrtv_mean = v_rows.sqrt().mean(dim=1)
            logs.update({
                f"{phase}/engram_v_mean_hist": _maybe_hist("v_mean", v_mean),
                f"{phase}/engram_sqrtv_mean_hist": _maybe_hist("sqrtv_mean", sqrtv_mean),
                f"{phase}/engram_v_mean_sample_mean": float(v_mean.mean().item()),
                f"{phase}/engram_sqrtv_mean_sample_mean": float(sqrtv_mean.mean().item()),
            })
            if exp_avg is not None:
                update = m_rows / (v_rows.sqrt() + opt.param_groups[0].get("eps", 1e-10))
                update_rms = row_rms_stable(update)
                logs.update({
                    f"{phase}/engram_adam_update_rms_hist": _maybe_hist("adam_update_rms", update_rms),
                    f"{phase}/engram_adam_update_rms_sample_mean": float(update_rms.mean().item()),
                    f"{phase}/engram_adam_update_rms_sample_p99": float(torch.quantile(update_rms, 0.99).item()),
                })
        elif state.get("exp_avg_sq_row") is not None:
            v_row = state["exp_avg_sq_row"].index_select(0, idx).float()
            v_mean = v_row
            sqrtv_mean = v_row.clamp_min(0.0).sqrt()
            logs.update({
                f"{phase}/engram_v_row_hist": _maybe_hist("v_row", v_row),
                f"{phase}/engram_v_row_sample_mean": float(v_row.mean().item()),
            })
            if m_rms is not None:
                update_rms = m_rms / (sqrtv_mean + opt.param_groups[0].get("eps", 1e-10))
                logs.update({
                    f"{phase}/engram_adam_update_rms_hist": _maybe_hist("adam_update_rms", update_rms),
                    f"{phase}/engram_adam_update_rms_sample_mean": float(update_rms.mean().item()),
                    f"{phase}/engram_adam_update_rms_sample_p99": float(torch.quantile(update_rms, 0.99).item()),
                })
        hit_count = state.get("hit_count")
        if hit_count is not None:
            hits = hit_count.index_select(0, idx).float()
            logs.update({
                f"{phase}/engram_optimizer_hit_count_hist": _maybe_hist("hit_count", hits),
                f"{phase}/engram_optimizer_hit_count_sample_mean": float(hits.mean().item()),
                f"{phase}/engram_optimizer_hit_count_sample_p99": float(torch.quantile(hits, 0.99).item()),
                f"{phase}/engram_optimizer_hit_count_sample_frac_ever": float((hits > 0).float().mean().item()),
            })
            _add_engram_hit_bucket_stats(
                logs,
                f"{phase}/engram_hit_bucket",
                hits,
                row_rms,
                m_rms=m_rms,
                v_mean=v_mean,
                sqrtv_mean=sqrtv_mean,
                update_rms=update_rms,
            )
        base_model = training_manager.model._orig_mod if hasattr(training_manager.model, "_orig_mod") else training_manager.model
        bigram_embed = getattr(base_model, "bigram_embed", None)
        hit_hist = bigram_embed._active_hit_hist() if bigram_embed is not None and hasattr(bigram_embed, "_active_hit_hist") else None
        if hit_hist is not None and hit_hist.numel() == param.shape[0]:
            hist_idx = idx.to(device=hit_hist.device)
            train_hits = hit_hist.index_select(0, hist_idx).float()
            logs.update({
                f"{phase}/engram_train_hit_hist": _maybe_hist("train_hit", train_hits),
                f"{phase}/engram_train_hit_sample_mean": float(train_hits.mean().item()),
                f"{phase}/engram_train_hit_sample_frac_ever": float((train_hits > 0).float().mean().item()),
            })
        return logs

def profile_step_enabled(step: int) -> bool:
    return profile_events and profile_events_start <= step < profile_events_start + profile_events_steps

def profile_start_event(enabled: bool):
    if not enabled:
        return None
    event = torch.cuda.Event(enable_timing=True)
    event.record()
    return event

def profile_end_event(pairs: list, name: str, start_event):
    if start_event is None:
        return
    end_event = torch.cuda.Event(enable_timing=True)
    end_event.record()
    pairs.append((name, start_event, end_event))

def profile_add_wall(wall_times: dict[str, float], name: str, start_time: float):
    wall_times[name] = wall_times.get(name, 0.0) + 1000 * (time.perf_counter() - start_time)

def profile_print(prefix: str, step: int, pairs: list, wall_times: dict[str, float], wall_total_ms: float):
    torch.cuda.synchronize()
    cuda_times: dict[str, float] = {}
    for name, start_event, end_event in pairs:
        cuda_times[name] = cuda_times.get(name, 0.0) + start_event.elapsed_time(end_event)
    cuda_total_ms = sum(cuda_times.values())
    fields = [f"{prefix} step:{step}", f"wall_total_ms:{wall_total_ms:.2f}", f"cuda_total_ms:{cuda_total_ms:.2f}"]
    for name, elapsed_ms in wall_times.items():
        fields.append(f"wall_{name}_ms:{elapsed_ms:.2f}")
    for name, elapsed_ms in cuda_times.items():
        fields.append(f"cuda_{name}_ms:{elapsed_ms:.2f}")
    print0(" ".join(fields), console=True)

def debug_check_model_params(model: nn.Module, step: int):
    if not rom_debug_nan or step < rom_debug_nan_min_step:
        return
    base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    for name, param in base_model.named_parameters():
        if torch.isfinite(param).all():
            continue
        p = param.detach().float()
        finite_frac = torch.isfinite(p).float().mean().item()
        absmax = torch.nan_to_num(p, nan=0.0, posinf=float("inf"), neginf=float("inf")).abs().amax().item()
        raise RuntimeError(f"ROM_DEBUG_NAN nonfinite param at step {step}: {name} shape={tuple(param.shape)} dtype={param.dtype} finite_frac={finite_frac:.6f} absmax={absmax:.6g}")

def debug_check_param_grad(model: nn.Module, step: int, accum_idx: int, label: str):
    if not rom_debug_nan or step not in rom_debug_grad_steps:
        return
    base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    param = getattr(base_model, label)
    grad = param.grad
    if grad is None:
        print0(f"ROM_DEBUG_NAN grad {label} at step {step} accum {accum_idx}: None", console=True)
        return
    if torch.isfinite(grad).all():
        print0(f"ROM_DEBUG_NAN grad {label} at step {step} accum {accum_idx}: finite", console=True)
        return
    g = grad.detach().float()
    finite_frac = torch.isfinite(g).float().mean().item()
    absmax = torch.nan_to_num(g, nan=0.0, posinf=float("inf"), neginf=float("inf")).abs().amax().item()
    raise RuntimeError(f"ROM_DEBUG_NAN nonfinite grad {label} after backward at step {step} accum {accum_idx}: shape={tuple(grad.shape)} dtype={grad.dtype} finite_frac={finite_frac:.6f} absmax={absmax:.6g}")

def load_engram_hit_hist_for_checkpoint(base_model: nn.Module, checkpoint_path: str):
    if not (engram_bigram and engram_hit_hist and getattr(base_model.bigram_embed, "hit_hist", None) is not None):
        return None
    hist_path = engram_hit_hist_load
    if not hist_path:
        ckpt_path = Path(checkpoint_path)
        if ckpt_path.name.startswith("state_step"):
            hist_path = str(ckpt_path.with_name(ckpt_path.name.replace("state_step", "engram_hit_hist_step", 1)))
    if hist_path and os.path.exists(hist_path):
        hit_payload = torch.load(hist_path, map_location=device, weights_only=False)
        base_model.bigram_embed.hit_hist.copy_(hit_payload["hit_hist"].to(device=device, dtype=base_model.bigram_embed.hit_hist.dtype))
        return hist_path
    return None


# begin by printing this file (the Python code)
print0(code)
print0("="*100)
# log information about the hardware/software environment this is running on
print0(f"Running Python {sys.version}")
print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}")
print0(f"Running Triton version {triton.__version__}")

def nvidia_smi():
    import subprocess  # avoid top level import
    return subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout
print0(nvidia_smi())

if model_seed is not None:
    torch.manual_seed(model_seed)
    torch.cuda.manual_seed_all(model_seed)

model: nn.Module = GPT(
    vocab_size=50257,
    num_layers=11,
    num_heads=6,
    head_dim=128,
    model_dim=768,
    max_seq_len=args.val_batch_size // (grad_accum_steps * world_size)
).cuda()
if engram_cache_recon:
    base_model_for_cache_recon = model._orig_mod if hasattr(model, "_orig_mod") else model
    if engram_cache_recon_target_layer >= base_model_for_cache_recon.num_layers:
        raise ValueError(f"ENGRAM_CACHE_RECON_TARGET_LAYER must be in [0, {base_model_for_cache_recon.num_layers - 1}]")
    if engram_cache_recon_source_layer not in base_model_for_cache_recon.bigram_layer_ids:
        raise ValueError("ENGRAM_CACHE_RECON_SOURCE_LAYER must be included in ROM_LAYERS")
    if not engram_cache_readout and tuple(base_model_for_cache_recon.bigram_layer_ids) != (engram_cache_recon_source_layer,):
        raise ValueError("ENGRAM_CACHE_RECON currently expects exactly one read layer; set ROM_LAYERS to ENGRAM_CACHE_RECON_SOURCE_LAYER")
print0(f"Experiment config: world_size={world_size} grad_accum_steps={grad_accum_steps} bigram_vocab_size={args.bigram_vocab_size} run_id={args.run_id} model_seed={model_seed if model_seed is not None else ''} train_data_seed={train_data_seed} device_capability=sm{device_capability[0]}{device_capability[1]} compile_model={int(compile_model)} compile_backend={compile_backend or 'inductor'} compile_layer_modules={int(compile_layer_modules)} compile_dense_layer_body={int(compile_dense_layer_body)} plain_mlp_train={int(plain_mlp_train)} fused_ce_eval={int(fused_ce_eval)} final_smear_mtp={int(final_smear_mtp)} final_smear_mtp_init={final_smear_mtp_init} snoo_outer={int(snoo_outer)} snoo_lr={snoo_lr} snoo_momentum={snoo_momentum} snoo_k={snoo_k} snoo_include_engram={int(snoo_include_engram)} normuon_update_smoothing={normuon_update_smoothing} adam_embed_vector_adam={int(adam_embed_vector_adam)} adam_embed_scalar_adam={int(adam_embed_scalar_adam)} rom_bigram={int(rom_bigram)} engram_bigram={int(engram_bigram)} engram_analyze={int(engram_analyze)} engram_analyze_tokens={engram_analyze_tokens} engram_hit_hist={int(engram_hit_hist)} engram_mask_unhit_eval={int(engram_mask_unhit_eval)} engram_eval_ckpt={engram_eval_ckpt} disable_embed_split={int(disable_embed_split)} rom_debug_nan={int(rom_debug_nan)} rom_debug_nan_min_step={rom_debug_nan_min_step} rom_single_token={int(rom_single_token)} rom_token={int(rom_token)} rom_engram_gate={int(rom_engram_gate)} rom_write={int(rom_write)} rom_heads={rom_heads} rom_key_dim={rom_key_dim} rom_value_dim={rom_value_dim} rom_mqa={int(rom_mqa)} rom_layer_only={rom_layer_only} rom_layers={','.join(map(str, rom_layers))} bigram_layer_ids={','.join(map(str, getattr(model, 'bigram_layer_ids', ())))} rom_output_scale={rom_output_scale} rom_short_conv={int(rom_short_conv)} rom_short_conv_kernel={rom_short_conv_kernel} rom_state_sparse_embedding={int(rom_state_sparse_embedding)} rom_state_sparse_adam={int(rom_state_sparse_adam)} rom_state_sparse_sgd={int(rom_state_sparse_sgd)} rom_state_normwrite={int(rom_state_normwrite)} rom_state_recovered_normwrite={int(rom_state_recovered_normwrite)} rom_state_write_rms={rom_state_write_rms} rom_state_init_std={rom_state_init_std} rom_state_diag_init={int(rom_state_diag_init)} rom_state_frob_norm={rom_state_frob_norm} rom_state_row_rms_cap={rom_state_row_rms_cap} rom_sparse_sanitize={int(rom_sparse_sanitize)} rom_sparse_adam_lr_mul={rom_sparse_adam_lr_mul} rom_sparse_adam_beta2={rom_sparse_adam_beta2} rom_sparse_row_scalar_adam={int(rom_sparse_row_scalar_adam)} rom_sparse_sgd_lr_mul={rom_sparse_sgd_lr_mul} rom_sparse_sgd_momentum={rom_sparse_sgd_momentum} rom_table_nonemb_mult={rom_table_nonemb_mult} engram_dim={engram_dim} engram_store_dim={engram_store_dim} engram_init_std={engram_init_std} engram_init_zero={int(engram_init_zero)} engram_heads={engram_heads} engram_max_ngram={engram_max_ngram} engram_ngram_row_factors={engram_ngram_row_factors_raw} engram_short_conv={int(engram_short_conv)} engram_short_conv_kernel={engram_short_conv_kernel} engram_normalize_readout={int(engram_normalize_readout)} engram_normalize_memory_heads={int(engram_normalize_memory_heads)} engram_sketch_k={engram_sketch_k} engram_sketch_dim_signs={int(engram_sketch_dim_signs)} engram_sketch_dim_sign_mode={engram_sketch_dim_sign_mode} engram_sketch_scalar_sign_mode={engram_sketch_scalar_sign_mode} engram_sketch_include_base={int(engram_sketch_include_base)} engram_sketch_slot_readout={int(engram_sketch_slot_readout)} engram_sketch_slot_mix={int(engram_sketch_slot_mix)} engram_sketch_combine_mix={int(engram_sketch_combine_mix)} engram_superpose_k={engram_superpose_k} engram_superpose_include_base={int(engram_superpose_include_base)} engram_head_dropout={engram_head_dropout} engram_head_dropout_final={engram_head_dropout_final} engram_head_dropout_schedule_steps={engram_head_dropout_schedule_steps} engram_output_dropout={engram_output_dropout} engram_output_dropout_final={engram_output_dropout_final} engram_output_dropout_schedule_steps={engram_output_dropout_schedule_steps} engram_read_hit_scale_exponent={engram_read_hit_scale_exponent} engram_read_hit_scale_offset={engram_read_hit_scale_offset} engram_read_hit_scale_min={engram_read_hit_scale_min} engram_read_hit_scale_max={engram_read_hit_scale_max} engram_read_hit_scale_norm_mean={int(engram_read_hit_scale_norm_mean)} engram_hash_seed={engram_hash_seed} engram_layer_hashes={int(engram_layer_hashes)} engram_layer_readouts={int(engram_layer_readouts)} engram_layer_partitions={int(engram_layer_partitions)} engram_layer_signs={int(engram_layer_signs)} engram_static_gate={int(engram_static_gate)} engram_static_gate_init={engram_static_gate_init} engram_layer_partition_groups={engram_layer_partition_groups} engram_per_head={int(engram_per_head)} engram_canonicalize={int(engram_canonicalize)} engram_latent={int(engram_latent)} engram_latent_quantizer={engram_latent_quantizer} engram_latent_fsq_levels={engram_latent_fsq_levels_raw} engram_latent_fsq_eps={engram_latent_fsq_eps} engram_latent_bsq_bits={engram_latent_bsq_bits} engram_latent_rows_per_head={engram_latent_rows_per_head} engram_latent_input_scale={engram_latent_input_scale} engram_latent_ste_scale={engram_latent_ste_scale} engram_latent_mix_ngram={int(engram_latent_mix_ngram)} engram_latent_aux_readout={int(engram_latent_aux_readout)} engram_latent_aux_scale={engram_latent_aux_scale} engram_latent_aux_scale_final={engram_latent_aux_scale_final} engram_latent_aux_scale_schedule_steps={engram_latent_aux_scale_schedule_steps} engram_latent_aux_scale_schedule_start={engram_latent_aux_scale_schedule_start} engram_latent_pkm_subkeys={engram_latent_pkm_subkeys} engram_latent_pkm_key_dim={engram_latent_pkm_key_dim} engram_latent_pkm_topk={engram_latent_pkm_topk} engram_cache_recon={int(engram_cache_recon)} engram_cache_recon_source_layer={engram_cache_recon_source_layer} engram_cache_recon_target_layer={engram_cache_recon_target_layer} engram_cache_recon_weight={engram_cache_recon_weight} engram_cache_recon_mode={engram_cache_recon_mode} engram_cache_readout={int(engram_cache_readout)} engram_cache_cfg_scale={engram_cache_cfg_scale} engram_cache_detach_memory={int(engram_cache_detach_memory)} engram_cache_learned_cfg={int(engram_cache_learned_cfg)} engram_mhc={int(engram_mhc)} engram_mhc_init={engram_mhc_init} engram_mhc_streams={engram_mhc_streams} engram_mhc_identity={int(engram_mhc_identity)} engram_mhc_dynamic={int(engram_mhc_dynamic)} engram_mhc_delta={int(engram_mhc_delta)} engram_attnres_merge={int(engram_attnres_merge)} engram_attnres_merge_gain={engram_attnres_merge_gain} engram_attnres_extra_source_layer={engram_attnres_extra_source_layer} engram_attnres_extra_target_layer={engram_attnres_extra_target_layer} engram_bank_attnres={int(engram_bank_attnres)} engram_attnres_delta={int(engram_attnres_delta)} engram_untied_proj={int(engram_untied_proj)} engram_adam_every_step={int(engram_adam_every_step)} engram_lr_mul={engram_lr_mul} engram_lr_floor={engram_lr_floor} engram_update_metrics={int(engram_update_metrics)} engram_update_metrics_every={engram_update_metrics_every} engram_disable_compile_region={int(engram_disable_compile_region)} profile_events={int(profile_events)} profile_events_start={profile_events_start} profile_events_steps={profile_events_steps} torch_profiler={int(torch_profiler)} torch_profiler_start={torch_profiler_start} torch_profiler_steps={torch_profiler_steps} torch_profiler_record_shapes={int(torch_profiler_record_shapes)} torch_profiler_profile_memory={int(torch_profiler_profile_memory)} torch_profiler_with_stack={int(torch_profiler_with_stack)} engram_touched_update_metrics={int(engram_touched_update_metrics)} engram_touched_update_metrics_chunk={engram_touched_update_metrics_chunk} engram_sparse_adam={int(engram_sparse_adam)} engram_sparse_vector_adam={int(engram_sparse_vector_adam)} engram_sparse_scalar_adam={int(engram_sparse_scalar_adam)} engram_sparse_adam_tail_steps={engram_sparse_adam_tail_steps} engram_sparse_extra_steps={engram_sparse_extra_steps} engram_sparse_hit_lr={int(engram_sparse_hit_lr)} engram_sparse_hit_lr_exponent={engram_sparse_hit_lr_exponent} engram_sparse_hit_lr_min={engram_sparse_hit_lr_min} engram_sparse_hit_lr_max={engram_sparse_hit_lr_max} engram_sparse_sanitize={int(engram_sparse_sanitize)} engram_sparse_param_clamp={engram_sparse_param_clamp} engram_sparse_row_rms_cap={engram_sparse_row_rms_cap} engram_sparse_row_rms_floor={engram_sparse_row_rms_floor} engram_sparse_row_rms_floor_hit_max={engram_sparse_row_rms_floor_hit_max} engram_sparse_row_rms_norm={engram_sparse_row_rms_norm} engram_sparse_row_decay={engram_sparse_row_decay} engram_sparse_grad_coalesce_hook={int(engram_sparse_grad_coalesce_hook)} engram_sparse_grad_coalesce_hook_start={engram_sparse_grad_coalesce_hook_start} engram_manual_sparse_coalesce_start={engram_manual_sparse_coalesce_start} engram_offload={int(engram_offload)} engram_offload_pin={int(engram_offload_pin)} engram_offload_pin_staging={int(engram_offload_pin_staging)} engram_offload_prefetch={int(engram_offload_prefetch)} engram_offload_async_adam={int(engram_offload_async_adam)} engram_offload_gpu_adam={int(engram_offload_gpu_adam)} engram_offload_prefetch_moments={int(engram_offload_prefetch_moments)} engram_offload_merge_pending={int(engram_offload_merge_pending)} engram_offload_lazy_moments={int(engram_offload_lazy_moments)} engram_offload_second_moment={int(engram_offload_second_moment)} engram_offload_moment_dtype={engram_offload_moment_dtype_name} rom_non_embedding_params={getattr(model, 'rom_non_embedding_params', '')} rom_target_state_params={getattr(model, 'rom_target_state_params', '')}", console=True)
print0(f"Engram latent aux config: aux_readout={int(engram_latent_aux_readout)} aux_scale={engram_latent_aux_scale}->{engram_latent_aux_scale_final} schedule_steps={engram_latent_aux_scale_schedule_steps} schedule_start={engram_latent_aux_scale_schedule_start}", console=True)
print0(f"Engram memory trainability: freeze_memory={int(engram_freeze_memory)}", console=True)
print0(f"Engram hash config: avalanche_hash={int(engram_avalanche_hash)} hash_seed={engram_hash_seed} layer_hashes={int(engram_layer_hashes)} layer_partitions={int(engram_layer_partitions)} layer_signs={int(engram_layer_signs)} layer_sign_scale={engram_layer_sign_scale}->{engram_layer_sign_scale_final} layer_sign_scale_schedule_steps={engram_layer_sign_scale_schedule_steps} layer_sign_scale_schedule_start={engram_layer_sign_scale_schedule_start} layer_sign_aux_scale={engram_layer_sign_aux_scale}->{engram_layer_sign_aux_scale_final} layer_sign_aux_scale_schedule_steps={engram_layer_sign_aux_scale_schedule_steps} layer_sign_aux_scale_schedule_start={engram_layer_sign_aux_scale_schedule_start} layer_row_signs={int(engram_layer_row_signs)} layer_row_signs_aux_only={int(engram_layer_row_signs_aux_only)} layer_row_sign_scale={engram_layer_row_sign_scale}->{engram_layer_row_sign_scale_final} layer_row_sign_scale_schedule_steps={engram_layer_row_sign_scale_schedule_steps} layer_row_sign_scale_schedule_start={engram_layer_row_sign_scale_schedule_start}", console=True)
print0(f"Engram ngram scale config: row_factors={engram_ngram_row_factors_raw} read_scales={engram_ngram_read_scales_raw} read_scales_final={engram_ngram_read_scales_final_raw} read_scale_schedule_steps={engram_ngram_read_scale_schedule_steps} read_scale_norm={int(engram_ngram_read_scale_norm)}", console=True)
print0(f"Engram sketch config: k={engram_sketch_k} include_base={int(engram_sketch_include_base)} aux_scale={engram_sketch_aux_scale} aux_scale_final={engram_sketch_aux_scale_final} schedule_steps={engram_sketch_aux_scale_schedule_steps} schedule_start={engram_sketch_aux_scale_schedule_start} aux_learned_scale={int(engram_sketch_aux_learned_scale)} aux_learned_scale_init={engram_sketch_aux_learned_scale_init} aux_learned_scale_max={engram_sketch_aux_learned_scale_max} aux_learned_scale_lr_mul={engram_sketch_aux_learned_scale_lr_mul} slot_readout={int(engram_sketch_slot_readout)} slot_attention={int(engram_sketch_slot_attention)} slot_mix={int(engram_sketch_slot_mix)} combine_mix={int(engram_sketch_combine_mix)} hit_hist_base_only={int(engram_sketch_hit_hist_base_only)}", console=True)
print0(f"Engram hit dropout config: dropout={engram_hit_dropout} final={engram_hit_dropout_final} schedule_steps={engram_hit_dropout_schedule_steps} schedule_start={engram_hit_dropout_schedule_start} decay_final={engram_hit_dropout_decay_final} decay_steps={engram_hit_dropout_decay_steps} decay_start={engram_hit_dropout_decay_start} min_hits={engram_hit_dropout_min_hits} invert_scale={int(engram_hit_dropout_invert_scale)}", console=True)
print0(f"Engram superpose config: k={engram_superpose_k} include_base={int(engram_superpose_include_base)} aux_scale={engram_superpose_aux_scale} aux_scale_final={engram_superpose_aux_scale_final} schedule_steps={engram_superpose_aux_scale_schedule_steps} schedule_start={engram_superpose_aux_scale_schedule_start} normalize={int(engram_superpose_normalize)}", console=True)
print0(f"Engram hot split config: hot_split={int(engram_hot_split)} value_only={int(engram_hot_split_value_only)} min_hits={engram_hot_split_min_hits} aux_scale={engram_hot_split_aux_scale} aux_scale_final={engram_hot_split_aux_scale_final} schedule_steps={engram_hot_split_aux_scale_schedule_steps} schedule_start={engram_hot_split_aux_scale_schedule_start} aux_slots={engram_hot_split_aux_slots} ramp_steps={engram_hot_split_ramp_steps} detach_aux={int(engram_hot_split_detach_aux)} dedup_aux={int(engram_hot_split_dedup_aux)} train_only={int(engram_hot_split_train_only)}", console=True)
print0(f"Engram sparse optimizer flags: adam={int(engram_sparse_adam)} vector_adam={int(engram_sparse_vector_adam)} scalar_adam={int(engram_sparse_scalar_adam)} row_adagrad={int(engram_sparse_row_adagrad)} hit_lr={int(engram_sparse_hit_lr)} fal={int(engram_sparse_fal)} ifal={int(engram_sparse_ifal)} batch_freq_norm={int(engram_sparse_batch_freq_norm)} tail_steps={engram_sparse_adam_tail_steps}", console=True)
print0(f"Engram sparse hit LR schedule: blend={engram_sparse_hit_lr_blend}->{engram_sparse_hit_lr_blend_final} schedule_steps={engram_sparse_hit_lr_blend_schedule_steps} schedule_start={engram_sparse_hit_lr_blend_schedule_start} exponent={engram_sparse_hit_lr_exponent} clamp=[{engram_sparse_hit_lr_min},{engram_sparse_hit_lr_max}]", console=True)
print0(f"Engram gate/mix config: detach_key_memory={int(engram_detach_key_memory)} detach_value_memory={int(engram_detach_value_memory)} detach_memory_layers={','.join(map(str, engram_detach_memory_layers))} fixed_half_gate={int(engram_fixed_half_gate)} head_mix={int(engram_head_mix)} layer_head_mix={int(engram_layer_head_mix)} layer_head_mix_delta={int(engram_layer_head_mix_delta)} head_mix_init={','.join(map(str, engram_head_mix_init))} head_mix_freeze={int(engram_head_mix_freeze)} layer_readout_delta={int(engram_layer_readout_delta)} layer_readout_delta_scale={engram_layer_readout_delta_scale}->{engram_layer_readout_delta_scale_final} layer_readout_delta_scale_schedule_steps={engram_layer_readout_delta_scale_schedule_steps} layer_readout_delta_scale_schedule_start={engram_layer_readout_delta_scale_schedule_start} layer_readout_delta_learned_scale={int(engram_layer_readout_delta_learned_scale)} layer_readout_delta_learned_scale_init={engram_layer_readout_delta_learned_scale_init} layer_readout_delta_learned_scale_max={engram_layer_readout_delta_learned_scale_max} layer_readout_delta_learned_scale_lr_mul={engram_layer_readout_delta_learned_scale_lr_mul} head_dropout={engram_head_dropout}->{engram_head_dropout_final} output_dropout={engram_output_dropout}->{engram_output_dropout_final} output_dropout_schedule_steps={engram_output_dropout_schedule_steps} output_dropout_schedule_start={engram_output_dropout_schedule_start} attnres_direct_residual={int(engram_attnres_direct_residual)} attnres_direct_init={engram_attnres_direct_init} attnres_direct_layers={engram_attnres_direct_layers_raw} attnres_layer_gain={int(engram_attnres_layer_gain)} attnres_layer_gain_init={engram_attnres_layer_gain_init} attnres_extra_source={engram_attnres_extra_source_layer} attnres_extra_target={engram_attnres_extra_target_layer} attnres_extra_bias_init={engram_attnres_extra_bias_init} attnres_extra_scale={engram_attnres_extra_scale}->{engram_attnres_extra_scale_final} attnres_extra_scale_schedule_steps={engram_attnres_extra_scale_schedule_steps} attnres_extra_scale_schedule_start={engram_attnres_extra_scale_schedule_start}", console=True)
print0(f"ROM readout config: normalize_readout={int(rom_normalize_readout)} readout_rms={rom_readout_rms} read_mlp={int(rom_read_mlp)} read_mlp_hidden_mult={rom_read_mlp_hidden_mult} read_mlp_lr_mul={rom_read_mlp_lr_mul} ema_smooth={int(rom_ema_smooth)} ema_alpha={rom_ema_alpha} ema_kernel={rom_ema_kernel}", console=True)
print0(f"ROM state write config: write_rms={rom_state_write_rms} hit_rms_low={rom_state_hit_rms_low} hit_rms_high={rom_state_hit_rms_high} hit_rms_knee={rom_state_hit_rms_knee}", console=True)
print0(f"Engram sparse Adam config: beta1={engram_sparse_adam_beta1} beta2={engram_sparse_adam_beta2} weight_decay={engram_sparse_weight_decay} tail_scale={engram_sparse_adam_tail_scale} extra_steps={engram_sparse_extra_steps}", console=True)
print0(f"Engram grad/shadow config: output_grad_metrics={int(engram_output_grad_metrics)} shadow_grad={int(engram_shadow_grad)} shadow_only={int(engram_shadow_only)} shadow_scale={engram_shadow_scale} shadow_write_rms={engram_shadow_write_rms} shadow_write_alpha={engram_shadow_write_alpha} shadow_decay={engram_shadow_decay} shadow_row_rms_cap={engram_shadow_row_rms_cap} shadow_hit_max={engram_shadow_hit_max}", console=True)
print0(f"Checkpoint config: save_checkpoint={int(args.save_checkpoint)} save_checkpoint_every={args.save_checkpoint_every} engram_save_hit_hist={int(engram_save_hit_hist)}", console=True)
if engram_bigram:
    base_model_for_print = model._orig_mod if hasattr(model, "_orig_mod") else model
    engram_module = base_model_for_print.bigram_embed
    embedding_shape = tuple(engram_module.embedding.weight.shape) if getattr(engram_module, "embedding", None) is not None else tuple(getattr(engram_module, "offload_weight", torch.empty(0)).shape)
    embedding_dtype = str(engram_module.embedding.weight.dtype) if getattr(engram_module, "embedding", None) is not None else str(getattr(engram_module, "offload_weight", torch.empty(0)).dtype)
    print0(f"Engram memory shape: rows={getattr(engram_module, 'num_memory_rows', '')} store_dim={getattr(engram_module, 'store_dim', '')} embedding_shape={embedding_shape} embedding_dtype={embedding_dtype} total_hash_heads={getattr(engram_module, 'total_hash_heads', '')} latent_codebook_size={getattr(engram_module, 'latent_codebook_size', '')} head_mods_min={int(engram_module.head_mods.min().item()) if hasattr(engram_module, 'head_mods') else ''} head_mods_max={int(engram_module.head_mods.max().item()) if hasattr(engram_module, 'head_mods') else ''}", console=True)
print0("="*100)
for m in model.modules():
    if isinstance(m, (nn.Conv1d, nn.Embedding, nn.Linear, nn.RMSNorm)):
        m.weight.data = m.weight.data.bfloat16()
if engram_bigram and engram_memory_fp32:
    base_model_for_dtype = model._orig_mod if hasattr(model, "_orig_mod") else model
    if getattr(base_model_for_dtype.bigram_embed, "embedding", None) is not None:
        base_model_for_dtype.bigram_embed.embedding.weight.data = base_model_for_dtype.bigram_embed.embedding.weight.data.float()
if engram_bigram:
    base_model_for_dtype = model._orig_mod if hasattr(model, "_orig_mod") else model
    if getattr(base_model_for_dtype.bigram_embed, "embedding", None) is not None:
        print0(f"Engram memory dtype after cast: {base_model_for_dtype.bigram_embed.embedding.weight.dtype}", console=True)
model.attn_gate_bank.data = model.attn_gate_bank.data.bfloat16()
model.ve_gate_bank.data = model.ve_gate_bank.data.bfloat16()
model.qk_bank.data = model.qk_bank.data.bfloat16()
model.vo_bank.data = model.vo_bank.data.bfloat16()
model.mlp_bank.data = model.mlp_bank.data.bfloat16()
for param in model.parameters():
    dist.broadcast(param.detach(), 0)

if compile_layer_modules:
    model.attn = torch.compile(model.attn, dynamic=False, fullgraph=False)
    model.attn_paired = torch.compile(model.attn_paired, dynamic=False, fullgraph=False)

if compile_dense_layer_body:
    dense_compile_kwargs = {"dynamic": False, "fullgraph": False}
    if compile_backend is not None:
        dense_compile_kwargs["backend"] = compile_backend
    model.compiled_dense_attn_mlp_layer = torch.compile(model._dense_attn_mlp_layer, **dense_compile_kwargs)
    model.compiled_dense_mlp_layer = torch.compile(model._dense_mlp_layer, **dense_compile_kwargs)

if compile_model:
    model: nn.Module = torch.compile(model, dynamic=False, fullgraph=not engram_disable_compile_region, backend=compile_backend)
training_manager = TrainingManager(model)
wandb_init_if_needed()

if engram_eval_ckpt:
    print0(f"Loading Engram checkpoint for eval: {engram_eval_ckpt}", console=True)
    checkpoint = torch.load(engram_eval_ckpt, map_location="cpu", mmap=True, weights_only=False)
    model_state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    if any(k.startswith("_orig_mod.") for k in model_state):
        model_state = {k.removeprefix("_orig_mod."): v for k, v in model_state.items()}
    model.load_state_dict(model_state, strict=True)
    if engram_offload and isinstance(checkpoint, dict):
        base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        base_model.bigram_embed.load_offload_state_dict(checkpoint.get("offloaded_engram", {}))
    checkpoint_step = int(checkpoint.get("step", training_schedule.total_steps)) if isinstance(checkpoint, dict) else training_schedule.total_steps
    del model_state, checkpoint

    training_manager.advance_schedule(min(checkpoint_step, training_schedule.total_steps))
    if checkpoint_step >= training_schedule.total_steps:
        training_manager.apply_final_ws_ext()
    base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    loaded_hist = load_engram_hit_hist_for_checkpoint(base_model, engram_eval_ckpt)
    if loaded_hist:
        print0(f"Loaded Engram hit histogram for eval: {loaded_hist}", console=True)
    elif engram_mask_unhit_eval:
        print0("Warning: ENGRAM_MASK_UNHIT_EVAL is set but no Engram hit histogram was loaded", console=True)

    model.eval()
    assert args.val_tokens % args.val_batch_size == 0
    val_steps = grad_accum_steps * args.val_tokens // args.val_batch_size
    val_loader = distributed_data_generator(args.val_files, args.val_batch_size, -1, grad_accum_steps=grad_accum_steps, align_to_bos=False)
    val_loss = 0
    with torch.no_grad():
        eval_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        for _ in range(val_steps):
            inputs, targets, cum_seqlens, bigram_inputs, _ = next(val_loader)
            forward_args = training_manager.get_forward_args()
            block_masks = build_flex_block_masks(cum_seqlens, forward_args, inputs.numel())
            batch_val_loss = eval_model(inputs, targets, cum_seqlens, bigram_inputs, forward_args, block_masks).mean()
            debug_check_finite("engram_eval_batch_loss", batch_val_loss)
            val_loss += batch_val_loss
    val_loss /= val_steps
    dist.reduce(val_loss, 0, op=dist.ReduceOp.AVG)
    result = {
        "checkpoint_path": engram_eval_ckpt,
        "checkpoint_step": checkpoint_step,
        "val_tokens": args.val_tokens,
        "val_loss": float(val_loss.item()),
        "engram_mask_unhit_eval": bool(engram_mask_unhit_eval),
        "engram_mask_unhit_eval_mode": engram_mask_unhit_eval_mode,
        "engram_mask_hit_min_eval": engram_mask_hit_min_eval,
        "engram_mask_hit_max_eval": engram_mask_hit_max_eval,
        "engram_mask_hit_invert_eval": bool(engram_mask_hit_invert_eval),
        "engram_eval_hit_scale": engram_eval_hit_scale,
        "engram_eval_hit_scale_min": engram_eval_hit_scale_min,
        "engram_eval_hit_scale_max": engram_eval_hit_scale_max,
        "engram_eval_hit_scale_invert": bool(engram_eval_hit_scale_invert),
        "engram_hit_hist": loaded_hist,
    }
    out_path = engram_eval_out or f"logs/{args.run_id}/engram_eval.json"
    if master_process:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
    eval_log = {
        "eval/val_loss": val_loss,
        "eval/val_tokens": args.val_tokens,
        "eval/mask_unhit": engram_mask_unhit_eval,
        "eval/mask_hit_min": engram_mask_hit_min_eval,
        "eval/mask_hit_max": engram_mask_hit_max_eval,
        "eval/mask_hit_invert": engram_mask_hit_invert_eval,
        "eval/hit_scale": engram_eval_hit_scale,
        "eval/hit_scale_min": engram_eval_hit_scale_min,
        "eval/hit_scale_max": engram_eval_hit_scale_max,
        "eval/hit_scale_invert": engram_eval_hit_scale_invert,
    }
    if wandb_hist_every > 0:
        eval_log.update(collect_engram_wandb_histograms(training_manager, step=checkpoint_step, phase="eval"))
    wandb_log(eval_log, checkpoint_step)
    print0(f"Engram eval val_loss:{val_loss:.4f} tokens:{args.val_tokens} mask_unhit:{int(engram_mask_unhit_eval)} mask_mode:{engram_mask_unhit_eval_mode} mask_hit_min:{engram_mask_hit_min_eval} mask_hit_max:{engram_mask_hit_max_eval} mask_hit_invert:{int(engram_mask_hit_invert_eval)} hit_scale:{engram_eval_hit_scale} hit_scale_min:{engram_eval_hit_scale_min} hit_scale_max:{engram_eval_hit_scale_max} hit_scale_invert:{int(engram_eval_hit_scale_invert)} hist:{loaded_hist or ''} out:{out_path}", console=True)
    if wandb_run is not None:
        wandb_run.finish()
    dist.barrier()
    dist.destroy_process_group()
    sys.exit(0)

if engram_analyze:
    print0(f"Loading Engram checkpoint for analysis: {engram_analyze_ckpt}", console=True)
    checkpoint = torch.load(engram_analyze_ckpt, map_location="cpu", mmap=True, weights_only=False)
    model_state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    if any(k.startswith("_orig_mod.") for k in model_state):
        model_state = {k.removeprefix("_orig_mod."): v for k, v in model_state.items()}
    model.load_state_dict(model_state, strict=True)
    if engram_bigram and engram_hit_hist:
        base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        loaded_hist = load_engram_hit_hist_for_checkpoint(base_model, engram_analyze_ckpt)
        if loaded_hist:
            print0(f"Loaded Engram hit histogram for analysis: {loaded_hist}", console=True)
        elif engram_mask_unhit_eval:
            print0("Warning: ENGRAM_MASK_UNHIT_EVAL is set but no Engram hit histogram was loaded", console=True)
    if engram_offload and isinstance(checkpoint, dict):
        base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        base_model.bigram_embed.load_offload_state_dict(checkpoint.get("offloaded_engram", {}))
    checkpoint_step = int(checkpoint.get("step", training_schedule.total_steps)) if isinstance(checkpoint, dict) else training_schedule.total_steps
    del model_state, checkpoint

    training_manager.advance_schedule(min(checkpoint_step, training_schedule.total_steps))
    if checkpoint_step >= training_schedule.total_steps:
        training_manager.apply_final_ws_ext()
    model.eval()

    ENGRAM_ANALYSIS_CHUNKS.clear()
    analysis_out = engram_analyze_out or f"logs/{args.run_id}/engram_analysis.json"
    if engram_analyze_prompts_file:
        prompts = load_engram_prompt_texts(engram_analyze_prompts_file)
        if not prompts:
            raise ValueError(f"No prompts found in {engram_analyze_prompts_file}")
        loss_by_prompt = []
        with torch.no_grad():
            eval_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            for prompt_idx, prompt in enumerate(prompts):
                ENGRAM_ANALYSIS_PROMPT_INDEX = prompt_idx
                ENGRAM_ANALYSIS_PROMPT_TEXT = prompt
                inputs, targets, cum_seqlens, bigram_inputs = build_prompt_batch(prompt)
                forward_args = training_manager.get_forward_args()
                block_masks = build_flex_block_masks(cum_seqlens, forward_args, inputs.numel())
                prompt_loss = eval_model(inputs, targets, cum_seqlens, bigram_inputs, forward_args, block_masks).mean()
                debug_check_finite("engram_prompt_analysis_loss", prompt_loss)
                loss_by_prompt.append(float(prompt_loss.item()))
        ENGRAM_ANALYSIS_PROMPT_INDEX = -1
        ENGRAM_ANALYSIS_PROMPT_TEXT = ""
        html_out = engram_analyze_html or analysis_out.replace(".json", ".html")
        write_engram_prompt_gating_report(
            html_out,
            json_path=analysis_out,
            run_id=args.run_id,
            checkpoint_path=engram_analyze_ckpt,
            loss_by_prompt=loss_by_prompt,
        )
        avg_prompt_loss = sum(loss_by_prompt) / max(1, len(loss_by_prompt))
        wandb_log({"analysis/prompt_avg_loss": avg_prompt_loss, "analysis/prompt_count": len(prompts)}, checkpoint_step)
        print0(f"Engram prompt analysis prompts:{len(prompts)} avg_loss:{avg_prompt_loss:.4f} json:{analysis_out} html:{html_out}", console=True)
    else:
        analysis_tokens = max(world_size, engram_analyze_tokens)
        analysis_tokens -= analysis_tokens % world_size
        val_loader = distributed_data_generator(args.val_files, analysis_tokens, -1, grad_accum_steps=1, align_to_bos=False)
        inputs, targets, cum_seqlens, bigram_inputs, _ = next(val_loader)
        forward_args = training_manager.get_forward_args()
        block_masks = build_flex_block_masks(cum_seqlens, forward_args, inputs.numel())
        with torch.no_grad():
            eval_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            analysis_loss = eval_model(inputs, targets, cum_seqlens, bigram_inputs, forward_args, block_masks).mean()
        dist.reduce(analysis_loss, 0, op=dist.ReduceOp.AVG)
        base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        write_engram_analysis(
            base_model.bigram_embed,
            analysis_out,
            run_id=args.run_id,
            checkpoint_path=engram_analyze_ckpt,
            loss=float(analysis_loss.item()),
            step=checkpoint_step,
        )
        wandb_log({"analysis/loss": analysis_loss, "analysis/tokens": analysis_tokens}, checkpoint_step)
        print0(f"Engram analysis loss:{analysis_loss:.4f} tokens:{analysis_tokens} out:{analysis_out}", console=True)
    if wandb_run is not None:
        wandb_run.finish()
    dist.barrier()
    dist.destroy_process_group()
    sys.exit(0)


########################################
#            Warmup kernels            #
########################################
if skip_kernel_warmup:
    print0("Skipping kernel warmup because SKIP_KERNEL_WARMUP=1", console=True)
else:
    print0("Warming up kernels (~7 minutes on first execution)", console=True)
    # Warmup the training kernels, then re-initialize the state so we aren't cheating
    initial_state = dict(model=copy.deepcopy(model.state_dict()),
                         optimizer=training_manager.get_state()) # save the initial state
    train_loader = distributed_data_generator(args.train_files, TRAINING_STAGES[0].batch_size, TRAINING_STAGES[0].train_max_seq_len, grad_accum_steps=grad_accum_steps, data_seed=train_data_seed)
    val_loader = distributed_data_generator(args.val_files, args.val_batch_size, -1, grad_accum_steps=grad_accum_steps, align_to_bos=False)

    transition_steps = training_manager.get_transition_steps()
    # first and last pair of steps in each transition
    warmup_steps = sorted(step for step in ({0, 1} | {s + offset for s in transition_steps for offset in [-2, -1, 0, 1] if s + offset >= 2}) if step <= training_schedule.total_steps)
    print0(f"Sampling steps {warmup_steps} for warmup", console=True)
    for step in warmup_steps:
        training_manager.advance_schedule(step)
        model.eval()
        with torch.no_grad():
            inputs, targets, cum_seqlens, bigram_inputs, _ = next(val_loader)
            forward_args = training_manager.get_forward_args()
            block_masks = build_flex_block_masks(cum_seqlens, forward_args, inputs.numel())
            eval_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            eval_model(inputs, targets, cum_seqlens, bigram_inputs, forward_args, block_masks).mean()
        model.train()
        for idx in range(grad_accum_steps):
            send_args = training_manager.train_loader_send_args
            inputs, targets, cum_seqlens, bigram_inputs, bigram_cpu = train_loader.send(send_args)
            forward_args = training_manager.get_forward_args()
            block_masks = build_flex_block_masks(cum_seqlens, forward_args, inputs.numel())
            training_manager.sparse_index_update(step, bigram_cpu)
            loss = model(inputs, targets, cum_seqlens, bigram_inputs, forward_args, block_masks).sum() * grad_scale
            debug_check_finite(f"warmup_loss_step{step}_accum{idx}", loss)
            training_manager.sparse_index_share(step)
            loss.backward()
            del loss
        training_manager.step_optimizers(step)
    print0("Resetting Model", console=True)
    model.zero_grad(set_to_none=True)
    model.load_state_dict(initial_state["model"])
    training_manager.reset(initial_state["optimizer"])
    if engram_bigram and engram_hit_hist:
        base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        base_model.bigram_embed.reset_hit_hist()
    del val_loader, train_loader, initial_state
    model.train()

########################################
#        Training and validation       #
########################################
train_loader = distributed_data_generator(args.train_files, TRAINING_STAGES[0].batch_size, TRAINING_STAGES[0].train_max_seq_len, grad_accum_steps=grad_accum_steps, data_seed=train_data_seed)

gc.collect()

training_time_ms = 0
active_torch_profiler = None
torch_profiler_trace_path = None

def save_training_checkpoint(step: int):
    save_hit_hist_only = (
        engram_save_hit_hist
        and engram_bigram
        and engram_hit_hist
        and (step == training_schedule.total_steps or (args.save_checkpoint_every > 0 and step % args.save_checkpoint_every == 0))
    )
    if not args.save_checkpoint and not save_hit_hist_only:
        return
    base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    global_hit_hist = None
    if engram_bigram and engram_hit_hist:
        global_hit_hist = base_model.bigram_embed.global_hit_hist()
    if not master_process:
        return
    os.makedirs(f"logs/{run_id}", exist_ok=True)
    if args.save_checkpoint:
        log = dict(step=step, code=code, model=model.state_dict(), optimizer=training_manager.get_state())
        if engram_bigram and engram_offload:
            log["offloaded_engram"] = base_model.bigram_embed.offload_state_dict()
        torch.save(log, f"logs/{run_id}/state_step{step:06d}.pt")
    if engram_bigram and engram_hit_hist:
        base_model.bigram_embed.save_hit_hist(f"logs/{run_id}/engram_hit_hist_step{step:06d}.pt", step=step, hit_hist=global_hit_hist)

# start the clock
torch.cuda.synchronize()
t0 = time.perf_counter()
# begin training
train_steps = training_schedule.total_steps
for step in range(train_steps + 1):
    rom_debug_nan_current_step = step
    if engram_debug_backward and rom_debug_nan and step >= rom_debug_nan_min_step:
        engram_debug_address_records.clear()
    last_step = (step == train_steps)
    training_manager.advance_schedule(step)
    # --------------- VALIDATION SECTION -----------------
    if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
        if last_step:
            training_manager.apply_final_ws_ext()
        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.perf_counter() - t0)
        model.eval()
        eval_hit_hist_active = False
        if engram_bigram and engram_hit_hist:
            base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            base_model.bigram_embed.begin_global_hit_hist_eval()
            eval_hit_hist_active = True
        debug_check_model_params(model, step)
        assert args.val_tokens % args.val_batch_size == 0
        val_steps = grad_accum_steps * args.val_tokens // args.val_batch_size
        val_loader = distributed_data_generator(args.val_files, args.val_batch_size, -1, grad_accum_steps=grad_accum_steps, align_to_bos=False)
        val_loss = 0
        val_cache_recon_loss = None
        with torch.no_grad():
            eval_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            for _ in range(val_steps):
                inputs, targets, cum_seqlens, bigram_inputs, _ = next(val_loader)
                forward_args = training_manager.get_forward_args()
                block_masks = build_flex_block_masks(cum_seqlens, forward_args, inputs.numel())
                batch_val_loss = eval_model(inputs, targets, cum_seqlens, bigram_inputs, forward_args, block_masks).mean()
                debug_check_finite(f"eval_batch_loss_step{step}", batch_val_loss)
                val_loss += batch_val_loss
                if engram_cache_recon:
                    batch_cache_recon_loss = eval_model.last_cache_recon_loss
                    if batch_cache_recon_loss is None:
                        raise RuntimeError("ENGRAM_CACHE_RECON expected a cache recon loss during validation")
                    debug_check_finite(f"eval_cache_recon_loss_step{step}", batch_cache_recon_loss)
                    val_cache_recon_loss = batch_cache_recon_loss if val_cache_recon_loss is None else val_cache_recon_loss + batch_cache_recon_loss
        val_loss /= val_steps
        if val_cache_recon_loss is not None:
            val_cache_recon_loss /= val_steps
        del val_loader
        dist.reduce(val_loss, 0, op=dist.ReduceOp.AVG)
        cache_recon_metrics = ""
        if val_cache_recon_loss is not None:
            dist.reduce(val_cache_recon_loss, 0, op=dist.ReduceOp.AVG)
            cache_recon_metrics = f" cache_recon_loss:{val_cache_recon_loss:.6g}"
        engram_metrics = training_manager.format_engram_update_metrics()
        mhc_metrics = training_manager.format_engram_mhc_metrics()
        val_log = {
            "val/loss": val_loss,
            "time/train_ms": training_time_ms,
            "time/step_avg_ms": training_time_ms / max(step, 1),
            "schedule/step": step,
            "schedule/total_steps": train_steps,
        }
        if val_cache_recon_loss is not None:
            val_log["val/cache_recon_loss"] = val_cache_recon_loss
        val_log.update(_parse_metric_suffix(engram_metrics, "engram/"))
        val_log.update(_parse_metric_suffix(mhc_metrics, "engram/"))
        if wandb_hist_every > 0 and step % wandb_hist_every == 0:
            val_log.update(collect_engram_wandb_histograms(training_manager, step=step, phase="eval"))
        wandb_log(val_log, step)
        print0(f"step:{step}/{train_steps} val_loss:{val_loss:.4f}{cache_recon_metrics} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/max(step, 1):.2f}ms{engram_metrics}{mhc_metrics}", console=True)
        if eval_hit_hist_active:
            base_model.bigram_embed.end_global_hit_hist_eval()
        model.train()
        if step > 0 and args.save_checkpoint_every > 0 and step % args.save_checkpoint_every == 0:
            save_training_checkpoint(step)
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.perf_counter()

    if last_step:
        if args.save_checkpoint_every <= 0 or step % args.save_checkpoint_every != 0:
            save_training_checkpoint(step)
        # the last step only has the validation loop, so break to avoid training
        break

    # --------------- TRAINING SECTION -----------------
    debug_check_model_params(model, step)
    if torch_profiler and step == torch_profiler_start:
        os.makedirs(f"logs/{run_id}", exist_ok=True)
        torch_profiler_trace_path = f"logs/{run_id}/torch_trace_step{step:06d}.json"
        active_torch_profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=torch_profiler_record_shapes,
            profile_memory=torch_profiler_profile_memory,
            with_stack=torch_profiler_with_stack,
        )
        active_torch_profiler.start()
        print0(f"torch_profiler_start step:{step} path:{torch_profiler_trace_path}", console=True)
    step_profile = profile_step_enabled(step)
    step_pairs = []
    step_wall_times: dict[str, float] = {}
    if step_profile:
        torch.cuda.synchronize()
    step_wall_start = time.perf_counter()
    train_loss = None
    train_cache_recon_loss = None
    for idx in range(grad_accum_steps):
        wall_start = time.perf_counter()
        inputs, targets, cum_seqlens, bigram_inputs, bigram_cpu = train_loader.send(training_manager.train_loader_send_args)
        if step_profile:
            profile_add_wall(step_wall_times, "loader", wall_start)

        wall_start = time.perf_counter()
        forward_args = training_manager.get_forward_args()
        if step_profile:
            profile_add_wall(step_wall_times, "forward_args", wall_start)

        wall_start = time.perf_counter()
        event = profile_start_event(step_profile)
        block_masks = build_flex_block_masks(cum_seqlens, forward_args, inputs.numel())
        if step_profile:
            profile_end_event(step_pairs, "block_masks", event)
            profile_add_wall(step_wall_times, "block_masks", wall_start)

        wall_start = time.perf_counter()
        event = profile_start_event(step_profile)
        training_manager.sparse_index_update(step, bigram_cpu)
        if step_profile:
            profile_end_event(step_pairs, "sparse_index_update", event)
            profile_add_wall(step_wall_times, "sparse_index_update", wall_start)

        wall_start = time.perf_counter()
        event = profile_start_event(step_profile)
        loss_per_token = model(inputs, targets, cum_seqlens, bigram_inputs, forward_args, block_masks)
        batch_train_loss = loss_per_token.detach().mean()
        train_loss = batch_train_loss if train_loss is None else train_loss + batch_train_loss
        loss = loss_per_token.sum() * grad_scale
        if step_profile:
            profile_end_event(step_pairs, "forward", event)
            profile_add_wall(step_wall_times, "forward", wall_start)
        debug_check_finite(f"train_loss_step{step}_accum{idx}", loss)
        if engram_cache_recon:
            batch_cache_recon_loss = (model._orig_mod if hasattr(model, "_orig_mod") else model).last_cache_recon_loss
            if batch_cache_recon_loss is None:
                raise RuntimeError("ENGRAM_CACHE_RECON expected a cache recon loss during training")
            debug_check_finite(f"train_cache_recon_loss_step{step}_accum{idx}", batch_cache_recon_loss)
            train_cache_recon_loss = batch_cache_recon_loss if train_cache_recon_loss is None else train_cache_recon_loss + batch_cache_recon_loss

        wall_start = time.perf_counter()
        event = profile_start_event(step_profile)
        training_manager.sparse_index_share(step)
        if step_profile:
            profile_end_event(step_pairs, "sparse_index_share", event)
            profile_add_wall(step_wall_times, "sparse_index_share", wall_start)

        wall_start = time.perf_counter()
        event = profile_start_event(step_profile)
        loss.backward()
        if step_profile:
            profile_end_event(step_pairs, "backward", event)
            profile_add_wall(step_wall_times, "backward", wall_start)
        debug_check_param_grad(model, step, idx, "qk_bank")
        del loss_per_token
        del loss
    wall_start = time.perf_counter()
    event = profile_start_event(step_profile)
    training_manager.step_optimizers(step)
    if step_profile:
        profile_end_event(step_pairs, "optimizer_total", event)
        profile_add_wall(step_wall_times, "optimizer_total", wall_start)
        profile_print("profile_train_step", step, step_pairs, step_wall_times, 1000 * (time.perf_counter() - step_wall_start))
    if active_torch_profiler is not None:
        active_torch_profiler.step()
        if step >= torch_profiler_start + torch_profiler_steps - 1:
            torch.cuda.synchronize()
            active_torch_profiler.stop()
            active_torch_profiler.export_chrome_trace(torch_profiler_trace_path)
            print0(f"torch_profiler_export step:{step} path:{torch_profiler_trace_path}", console=True)
            active_torch_profiler = None
            torch_profiler_trace_path = None
    debug_check_model_params(model, step)

    # logging
    approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
    train_cache_recon_metrics = ""
    if train_loss is None:
        raise RuntimeError("Training loss was not collected")
    train_loss /= grad_accum_steps
    dist.reduce(train_loss, 0, op=dist.ReduceOp.AVG)
    if train_cache_recon_loss is not None:
        train_cache_recon_loss /= grad_accum_steps
        dist.reduce(train_cache_recon_loss, 0, op=dist.ReduceOp.AVG)
        train_cache_recon_metrics = f" train_cache_recon_loss:{train_cache_recon_loss:.6g}"
    train_log = {
        "train/loss": train_loss,
        "time/train_ms": approx_training_time_ms,
        "time/step_avg_ms": approx_training_time_ms / (step + 1),
        "schedule/step": step + 1,
        "schedule/total_steps": train_steps,
    }
    if train_cache_recon_loss is not None:
        train_log["train/cache_recon_loss"] = train_cache_recon_loss
    engram_output_grad_log, engram_output_grad_metrics = consume_engram_output_grad_metrics()
    train_log.update(engram_output_grad_log)
    if wandb_hist_every > 0 and (step + 1) % wandb_hist_every == 0:
        train_log.update(collect_engram_wandb_histograms(training_manager, step=step + 1, phase="train"))
    wandb_log(train_log, step + 1)
    print0(f"step:{step+1}/{train_steps} train_loss:{train_loss:.4f}{train_cache_recon_metrics}{engram_output_grad_metrics} train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms/(step + 1):.2f}ms", console=True)

if args.run_evals:
    model.eval()
    from evals import hellaswag
    hellaswag.evaluate(model=model, 
                       schedule_cfg=training_manager.get_forward_args(), 
                       seq_len=args.val_batch_size // (grad_accum_steps * world_size),
                       get_bigram_hash=get_bigram_hash, 
                       print0=print0)

print0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
       f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB", console=True)
wandb_log({
    "memory/peak_allocated_mib": torch.cuda.max_memory_allocated() // 1024 // 1024,
    "memory/peak_reserved_mib": torch.cuda.max_memory_reserved() // 1024 // 1024,
}, train_steps)
if wandb_run is not None:
    wandb_run.finish()
dist.destroy_process_group()
