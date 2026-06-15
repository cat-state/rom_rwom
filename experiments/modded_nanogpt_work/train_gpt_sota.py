#!/usr/bin/env python3
"""SOTA-only entrypoint for the Engram NanoGPT experiments.

This file intentionally keeps the supported surface small.  It encodes the
current June 2026 SOTA backbone and a single near-SOTA sketch follow-up, then
execs ``train_gpt.py`` with the corresponding environment.  The full trainer
still contains the shared DDP/data/optimizer machinery; this file is the clean
path to run or audit the pathways we actually want to keep.
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path


TRUTHY = {"1", "true", "yes", "y", "on"}


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in TRUTHY


def set_env(values: dict[str, str | int | float]) -> None:
    for key, value in values.items():
        os.environ[key] = str(value)


def set_default_env(values: dict[str, str | int | float]) -> None:
    for key, value in values.items():
        os.environ.setdefault(key, str(value))


def common_sota_env(args: argparse.Namespace) -> dict[str, str | int | float]:
    return {
        "ENGRAM_BIGRAM": 1,
        "BIGRAM_FACTOR": args.bigram_factor,
        "GRAD_ACCUM_STEPS": args.grad_accum_steps,
        "MODEL_SEED": args.model_seed,
        "TRAIN_DATA_SEED": args.train_data_seed,
        "ROM_LAYERS": "2,8",
        "ENGRAM_DIM": 768,
        "ENGRAM_HEADS": 1,
        "ENGRAM_MAX_NGRAM": 3,
        "ENGRAM_NGRAM_ROW_FACTORS": "0.5,1.5",
        "ENGRAM_SHORT_CONV": 1,
        "ENGRAM_HASH_SEED": args.hash_seed,
        "ENGRAM_LAYER_HASHES": 1,
        "ENGRAM_LAYER_PARTITIONS": 1,
        "ENGRAM_LAYER_PARTITION_GROUPS": 1,
        "ENGRAM_LAYER_SIGNS": 0,
        "ENGRAM_LAYER_ROW_SIGNS": 0,
        "ENGRAM_HEAD_MIX": 1,
        "ENGRAM_HEAD_MIX_INIT": "0.5,-0.5",
        "ENGRAM_PER_HEAD": 1,
        "ENGRAM_CANONICALIZE": 1,
        "ENGRAM_NORMALIZE_READOUT": 1,
        "ENGRAM_NORMALIZE_MEMORY_HEADS": 1,
        "ENGRAM_INIT_STD": 0.01,
        "ENGRAM_UNTIED_PROJ": 1,
        "ENGRAM_ATTNRES_MERGE": 1,
        "ENGRAM_ATTNRES_MERGE_GAIN": 1.5,
        "ENGRAM_READ_HIT_SCALE_EXPONENT": 0.25,
        "ENGRAM_READ_HIT_SCALE_OFFSET": 1.0,
        "ENGRAM_READ_HIT_SCALE_MIN": 0.25,
        "ENGRAM_READ_HIT_SCALE_MAX": 4.0,
        "ENGRAM_READ_HIT_SCALE_NORM_MEAN": 1,
        "ENGRAM_LR_MUL": 5.0,
        "ENGRAM_LR_FLOOR": 0,
        "ENGRAM_SPARSE_ADAM": 0,
        "ENGRAM_SPARSE_VECTOR_ADAM": 0,
        "ENGRAM_SPARSE_SCALAR_ADAM": 1,
        "ENGRAM_SPARSE_ROW_ADAGRAD": 0,
        "ENGRAM_SPARSE_ADAM_TAIL_STEPS": 0,
        "ENGRAM_ADAM_EVERY_STEP": 1,
        "ENGRAM_HIT_HIST": 1,
        "ENGRAM_UPDATE_METRICS": 1,
        "ENGRAM_UPDATE_METRICS_EVERY": 250,
        "ENGRAM_LATENT": 0,
        "ENGRAM_CACHE_RECON": 0,
        "ENGRAM_CACHE_READOUT": 0,
        "ENGRAM_MHC": 0,
        "ENGRAM_HOT_SPLIT": 0,
        "ENGRAM_HIT_DROPOUT": 0.0,
        "ENGRAM_OUTPUT_DROPOUT": 0.0,
        "ENGRAM_HEAD_DROPOUT": 0.0,
        "NUM_SCHEDULED_ITERATIONS": args.steps,
        "NUM_EXTENSION_ITERATIONS": 0,
        "VAL_LOSS_EVERY": args.val_every,
        "SAVE_CHECKPOINT": int(args.save_checkpoint),
        "SAVE_CHECKPOINT_EVERY": args.save_checkpoint_every,
        "COMPILE_MODEL": 0,
        "COMPILE_LAYER_MODULES": 0,
        "COMPILE_DENSE_LAYER_BODY": 1,
    }


def sota_k2_env(args: argparse.Namespace) -> dict[str, str | int | float]:
    values = common_sota_env(args)
    values.update(
        {
            "ENGRAM_LAYER_READOUTS": 0,
            "ENGRAM_LAYER_READOUT_DELTA": 1,
            "ENGRAM_SUPERPOSE_K": 2,
            "ENGRAM_SUPERPOSE_INCLUDE_BASE": 1,
            "ENGRAM_SUPERPOSE_AUX_SCALE": 0.5,
            "ENGRAM_SUPERPOSE_AUX_SCALE_FINAL": 0.5,
            "ENGRAM_SUPERPOSE_NORMALIZE": 1,
            "ENGRAM_SKETCH_K": 1,
            "ENGRAM_SKETCH_INCLUDE_BASE": 0,
            "ENGRAM_SKETCH_SLOT_READOUT": 0,
            "ENGRAM_SKETCH_COMBINE_MIX": 0,
            "ENGRAM_SKETCH_AUX_LEARNED_SCALE": 0,
        }
    )
    return values


def sketch_combinemix_env(args: argparse.Namespace) -> dict[str, str | int | float]:
    values = common_sota_env(args)
    values.update(
        {
            "ENGRAM_LAYER_READOUTS": 0,
            "ENGRAM_LAYER_READOUT_DELTA": int(args.layerdelta),
            "ENGRAM_LAYER_READOUT_DELTA_SCALE": args.layerdelta_scale,
            "ENGRAM_LAYER_READOUT_DELTA_SCALE_FINAL": args.layerdelta_scale_final,
            "ENGRAM_LAYER_READOUT_DELTA_SCALE_SCHEDULE_START": args.layerdelta_scale_start,
            "ENGRAM_LAYER_READOUT_DELTA_SCALE_SCHEDULE_STEPS": args.layerdelta_scale_steps,
            "ENGRAM_LAYER_READOUT_DELTA_LEARNED_SCALE": int(args.layerdelta_learned_scale),
            "ENGRAM_LAYER_READOUT_DELTA_LEARNED_SCALE_INIT": args.layerdelta_learned_scale_init,
            "ENGRAM_LAYER_READOUT_DELTA_LEARNED_SCALE_MAX": args.layerdelta_learned_scale_max,
            "ENGRAM_LAYER_READOUT_DELTA_LEARNED_SCALE_LR_MUL": args.layerdelta_learned_scale_lr_mul,
            "ENGRAM_SUPERPOSE_K": 1,
            "ENGRAM_SUPERPOSE_INCLUDE_BASE": 0,
            "ENGRAM_SKETCH_K": args.sketch_k,
            "ENGRAM_SKETCH_DIM_SIGNS": 1,
            "ENGRAM_SKETCH_DIM_SIGN_MODE": "balanced",
            "ENGRAM_SKETCH_SCALAR_SIGN_MODE": "balanced",
            "ENGRAM_SKETCH_INCLUDE_BASE": 1,
            "ENGRAM_SKETCH_COMBINE_MIX": 1,
            "ENGRAM_SKETCH_COMBINE_MIX_MODE": args.mix_mode,
            "ENGRAM_SKETCH_COMBINE_MIX_MAX_DEV": args.mix_max_dev,
            "ENGRAM_SKETCH_HIT_HIST_BASE_ONLY": int(args.sketch_hit_hist_base_only),
            "ENGRAM_SKETCH_AUX_SCALE": args.sketch_aux_scale,
            "ENGRAM_SKETCH_AUX_SCALE_FINAL": args.sketch_aux_scale_final,
            "ENGRAM_SKETCH_AUX_SCALE_SCHEDULE_START": args.sketch_aux_scale_start,
            "ENGRAM_SKETCH_AUX_SCALE_SCHEDULE_STEPS": args.sketch_aux_scale_steps,
            "ENGRAM_SKETCH_AUX_LEARNED_SCALE": int(args.sketch_aux_learned_scale),
            "ENGRAM_SKETCH_AUX_LEARNED_SCALE_INIT": args.sketch_aux_learned_scale_init,
            "ENGRAM_SKETCH_AUX_LEARNED_SCALE_MAX": args.sketch_aux_learned_scale_max,
            "ENGRAM_SKETCH_AUX_LEARNED_SCALE_LR_MUL": args.sketch_aux_learned_scale_lr_mul,
        }
    )
    return values


def default_run_id(args: argparse.Namespace) -> str:
    if args.run_id:
        return args.run_id
    if args.pathway == "sota_k2":
        return (
            f"bf{args.bigram_factor}_sota_k2_headmix_layerdelta_norowsigns"
            f"_hashseed{args.hash_seed}_seed{args.model_seed}_{args.steps}_sotafile"
        )
    aux_tag = str(args.sketch_aux_scale).replace(".", "p")
    learn_tag = "_learnaux" if args.sketch_aux_learned_scale else ""
    basehist_tag = "_basehist" if args.sketch_hit_hist_base_only else ""
    return (
        f"bf{args.bigram_factor}_sketchk{args.sketch_k}_combinemix_{args.mix_mode}"
        f"_aux{aux_tag}{learn_tag}{basehist_tag}_seed{args.model_seed}_{args.steps}_sotafile"
    )


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run a slim, SOTA-only Engram pathway by configuring train_gpt.py."
    )
    parser.add_argument("--pathway", choices=("sota_k2", "sketch_combinemix"), default=os.environ.get("SOTA_PATHWAY", "sota_k2"))
    parser.add_argument("--run_id", default=os.environ.get("RUN_ID", ""))
    parser.add_argument("--bigram-factor", type=int, default=int(os.environ.get("BIGRAM_FACTOR", "300")))
    parser.add_argument("--model-seed", type=int, default=int(os.environ.get("MODEL_SEED", "5")))
    parser.add_argument("--train-data-seed", type=int, default=int(os.environ.get("TRAIN_DATA_SEED", "0")))
    parser.add_argument("--hash-seed", type=int, default=int(os.environ.get("ENGRAM_HASH_SEED", os.environ.get("HASH_SEED", "0"))))
    parser.add_argument("--steps", type=int, default=int(os.environ.get("NUM_SCHEDULED_ITERATIONS", "1500")))
    parser.add_argument("--val-every", type=int, default=int(os.environ.get("VAL_LOSS_EVERY", "250")))
    parser.add_argument("--grad-accum-steps", type=int, default=int(os.environ.get("GRAD_ACCUM_STEPS", "16")))
    parser.add_argument("--save-checkpoint", action="store_true", default=env_flag("SAVE_CHECKPOINT", False))
    parser.add_argument("--save-checkpoint-every", type=int, default=int(os.environ.get("SAVE_CHECKPOINT_EVERY", "0")))
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--sketch-k", type=int, default=int(os.environ.get("ENGRAM_SKETCH_K", os.environ.get("SKETCH_K", "2"))))
    parser.add_argument("--sketch-aux-scale", type=float, default=float(os.environ.get("ENGRAM_SKETCH_AUX_SCALE", "0.5")))
    parser.add_argument("--sketch-aux-scale-final", type=float, default=float(os.environ.get("ENGRAM_SKETCH_AUX_SCALE_FINAL", os.environ.get("ENGRAM_SKETCH_AUX_SCALE", "0.5"))))
    parser.add_argument("--sketch-aux-scale-start", type=int, default=int(os.environ.get("ENGRAM_SKETCH_AUX_SCALE_SCHEDULE_START", "0")))
    parser.add_argument("--sketch-aux-scale-steps", type=int, default=int(os.environ.get("ENGRAM_SKETCH_AUX_SCALE_SCHEDULE_STEPS", "0")))
    parser.add_argument("--sketch-hit-hist-base-only", action="store_true", default=env_flag("ENGRAM_SKETCH_HIT_HIST_BASE_ONLY", False))
    parser.add_argument("--sketch-aux-learned-scale", action="store_true", default=env_flag("ENGRAM_SKETCH_AUX_LEARNED_SCALE", False))
    parser.add_argument("--sketch-aux-learned-scale-init", type=float, default=float(os.environ.get("ENGRAM_SKETCH_AUX_LEARNED_SCALE_INIT", "1.0")))
    parser.add_argument("--sketch-aux-learned-scale-max", type=float, default=float(os.environ.get("ENGRAM_SKETCH_AUX_LEARNED_SCALE_MAX", "2.0")))
    parser.add_argument("--sketch-aux-learned-scale-lr-mul", type=float, default=float(os.environ.get("ENGRAM_SKETCH_AUX_LEARNED_SCALE_LR_MUL", "0.25")))
    parser.add_argument("--mix-mode", choices=("bounded", "softmax"), default=os.environ.get("ENGRAM_SKETCH_COMBINE_MIX_MODE", "bounded"))
    parser.add_argument("--mix-max-dev", type=float, default=float(os.environ.get("ENGRAM_SKETCH_COMBINE_MIX_MAX_DEV", "0.1")))

    parser.add_argument("--layerdelta", action="store_true", default=env_flag("ENGRAM_LAYER_READOUT_DELTA", False))
    parser.add_argument("--layerdelta-scale", type=float, default=float(os.environ.get("ENGRAM_LAYER_READOUT_DELTA_SCALE", "1.0")))
    parser.add_argument("--layerdelta-scale-final", type=float, default=float(os.environ.get("ENGRAM_LAYER_READOUT_DELTA_SCALE_FINAL", os.environ.get("ENGRAM_LAYER_READOUT_DELTA_SCALE", "1.0"))))
    parser.add_argument("--layerdelta-scale-start", type=int, default=int(os.environ.get("ENGRAM_LAYER_READOUT_DELTA_SCALE_SCHEDULE_START", "0")))
    parser.add_argument("--layerdelta-scale-steps", type=int, default=int(os.environ.get("ENGRAM_LAYER_READOUT_DELTA_SCALE_SCHEDULE_STEPS", "0")))
    parser.add_argument("--layerdelta-learned-scale", action="store_true", default=env_flag("ENGRAM_LAYER_READOUT_DELTA_LEARNED_SCALE", False))
    parser.add_argument("--layerdelta-learned-scale-init", type=float, default=float(os.environ.get("ENGRAM_LAYER_READOUT_DELTA_LEARNED_SCALE_INIT", "0.5")))
    parser.add_argument("--layerdelta-learned-scale-max", type=float, default=float(os.environ.get("ENGRAM_LAYER_READOUT_DELTA_LEARNED_SCALE_MAX", "1.0")))
    parser.add_argument("--layerdelta-learned-scale-lr-mul", type=float, default=float(os.environ.get("ENGRAM_LAYER_READOUT_DELTA_LEARNED_SCALE_LR_MUL", "0.25")))
    return parser.parse_known_args()


def main() -> None:
    args, extra_args = parse_args()
    if args.pathway == "sota_k2":
        env = sota_k2_env(args)
    else:
        env = sketch_combinemix_env(args)

    run_id = default_run_id(args)
    env.update({"RUN_ID": run_id, "WANDB_NAME": os.environ.get("WANDB_NAME", run_id)})
    set_default_env(
        {
            "WANDB": 1,
            "WANDB_MODE": "offline",
            "WANDB_PROJECT": "rom-rwom",
            "WANDB_ENTITY": "uwu1",
            "WANDB_GROUP": f"sota-slim-{args.pathway}",
            "WANDB_TAGS": f"engram,sota,slim,{args.pathway}",
            "WANDB_HISTOGRAMS": 1,
            "WANDB_HIST_EVERY": 250,
            "WANDB_HIST_ROWS": 131072,
        }
    )
    set_env(env)

    trainer = Path(__file__).with_name("train_gpt.py")
    if args.dry_run:
        for key in sorted(env):
            print(f"export {key}={shlex.quote(str(env[key]))}")
        print("exec", shlex.join([sys.executable, str(trainer), *extra_args]))
        return

    os.execv(sys.executable, [sys.executable, str(trainer), *extra_args])


if __name__ == "__main__":
    main()
