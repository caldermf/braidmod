#!/usr/bin/env python3
"""Train a B_4 transformer on integer Burau sign-slice tokens."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from interp.b4_transformer import B4AbsoluteTransformer, B4TransformerConfig  # noqa: E402
from interp.b4_interp import zero_except_windows  # noqa: E402
from interp.b4_z_sign import (  # noqa: E402
    SIGN_DIGIT_CONVENTION,
    SIGN_VOCAB_SIZE,
    B4FactorBatchIterable,
    discover_b4_shards,
    factor_ids_to_z_sign_tokens,
    simple_mats_z,
)
from interp.generate_b4_dataset import absolute_depth_for_length  # noqa: E402
from interp.train_b4_transformer import (  # noqa: E402
    atomic_json_dump,
    atomic_torch_save,
    best_from_history,
    finish_metrics,
    init_totals,
    load_shard,
    resolve_device,
    set_seed,
    update_metrics,
)


def infer_data_config(shard_paths: list[Path]) -> dict:
    meta = load_shard(shard_paths[0])["metadata"]
    length = int(meta["length"])
    return {
        "length": length,
        "absolute_depth": int(meta.get("absolute_depth", absolute_depth_for_length(length))),
        "vocab_size": SIGN_VOCAB_SIZE,
        "num_labels": 3,
    }


def make_loader(
    shard_paths: list[Path],
    *,
    split: str,
    batch_size: int,
    seed: int,
    epoch: int,
    shuffle: bool,
    max_examples: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    dataset = B4FactorBatchIterable(
        shard_paths,
        split=split,
        batch_size=batch_size,
        seed=seed,
        epoch=epoch,
        shuffle_shards=shuffle,
        shuffle_rows=shuffle,
        max_examples=max_examples,
    )
    return DataLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )


def checkpoint_payload(
    *,
    model: B4AbsoluteTransformer,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    epoch: int,
    history: list[dict],
    best_val_exact_accuracy: float,
    best_val_loss: float,
) -> dict:
    return {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "model_config": model.config.to_dict(),
        "args": vars(args),
        "epoch": epoch,
        "history": history,
        "best_val_exact_accuracy": best_val_exact_accuracy,
        "best_val_loss": best_val_loss,
        "label_convention": {
            "logit_0": "s_1 in final right descent set",
            "logit_1": "s_2 in final right descent set",
            "logit_2": "s_3 in final right descent set",
        },
        "input_format": "absolute_degree_z_burau_sign_slice_tokens_base3",
        "input_transform": (
            "none"
            if int(args.boundary_only_radius) < 0
            else f"zero_except_leading_and_trailing_windows_radius_{int(args.boundary_only_radius)}"
        ),
        "sign_digit_convention": SIGN_DIGIT_CONVENTION,
    }


def batch_to_tokens(
    batch: dict,
    *,
    device: torch.device,
    length: int,
    absolute_depth: int,
    simple_mats: torch.Tensor,
    boundary_only_radius: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    factor_ids = batch["factor_ids"].to(device, non_blocking=True)
    labels = batch["label_bits"].to(device, non_blocking=True)
    tokens = factor_ids_to_z_sign_tokens(
        factor_ids,
        length=length,
        absolute_depth=absolute_depth,
        simple_mats=simple_mats,
    )
    if boundary_only_radius >= 0:
        tokens = zero_except_windows(tokens, boundary_only_radius, leading=True, trailing=True)
    return tokens, labels


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    grad_clip: float,
    use_amp: bool,
    length: int,
    absolute_depth: int,
    simple_mats: torch.Tensor,
    boundary_only_radius: int,
) -> dict:
    model.train()
    totals = init_totals()
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for batch in loader:
        tokens, labels = batch_to_tokens(
            batch,
            device=device,
            length=length,
            absolute_depth=absolute_depth,
            simple_mats=simple_mats,
            boundary_only_radius=boundary_only_radius,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(tokens)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        update_metrics(totals, logits.detach().cpu(), labels.detach().cpu(), loss.detach().cpu())
    return finish_metrics(totals)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    criterion: nn.Module,
    use_amp: bool,
    length: int,
    absolute_depth: int,
    simple_mats: torch.Tensor,
    boundary_only_radius: int,
) -> dict:
    model.eval()
    totals = init_totals()
    for batch in loader:
        tokens, labels = batch_to_tokens(
            batch,
            device=device,
            length=length,
            absolute_depth=absolute_depth,
            simple_mats=simple_mats,
            boundary_only_radius=boundary_only_radius,
        )
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(tokens)
            loss = criterion(logits, labels)
        update_metrics(totals, logits.detach().cpu(), labels.detach().cpu(), loss.detach().cpu())
    return finish_metrics(totals)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a B_4 transformer on Z[v] sign tokens.")
    parser.add_argument("--data-dir", default="interp/data/generated/b4_l25_p2_n16777216")
    parser.add_argument("--out-dir", default="interp/artifacts/b4_l25_zsign_xfmr3_abs")
    parser.add_argument("--num-shards", type=int, default=64)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--train-examples-per-epoch", type=int, default=1_048_576)
    parser.add_argument("--eval-examples", type=int, default=262_144)
    parser.add_argument("--test-examples", type=int, default=524_288)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--ffn-mult", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--boundary-only-radius",
        type=int,
        default=-1,
        help="If nonnegative, zero all input tokens except leading/trailing support windows of this radius.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_paths = discover_b4_shards(args.data_dir, num_shards=args.num_shards, allow_partial=args.allow_partial)
    data_config = infer_data_config(shard_paths)

    device = resolve_device(args.device)
    use_amp = device.type == "cuda" and not args.no_amp
    pin_memory = device.type == "cuda"
    simple_mats = simple_mats_z(device)
    config = B4TransformerConfig(
        length=data_config["length"],
        absolute_depth=data_config["absolute_depth"],
        vocab_size=data_config["vocab_size"],
        num_labels=data_config["num_labels"],
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ffn_mult=args.ffn_mult,
        dropout=args.dropout,
    )
    model = B4AbsoluteTransformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    num_params = sum(param.numel() for param in model.parameters())

    print(
        json.dumps(
            {
                "data_dir": args.data_dir,
                "num_shards": len(shard_paths),
                "out_dir": str(out_dir),
                "device": str(device),
                "use_amp": use_amp,
                "num_params": num_params,
                "model_config": config.to_dict(),
                "input_format": "absolute_degree_z_burau_sign_slice_tokens_base3",
                "input_transform": (
                    "none"
                    if int(args.boundary_only_radius) < 0
                    else f"zero_except_leading_and_trailing_windows_radius_{int(args.boundary_only_radius)}"
                ),
            },
            indent=2,
        ),
        flush=True,
    )

    history: list[dict] = []
    best_val_exact_accuracy = -math.inf
    best_val_loss = math.inf
    best_path = out_dir / "best_model.pt"
    latest_path = out_dir / "latest_model.pt"
    history_path = out_dir / "history.json"
    start_epoch = 1

    if args.resume and latest_path.exists():
        resume_payload = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(resume_payload["model_state"])
        optimizer.load_state_dict(resume_payload["optimizer_state"])
        history = list(resume_payload.get("history", []))
        start_epoch = int(resume_payload.get("epoch", 0)) + 1
        best_val_exact_accuracy, best_val_loss = best_from_history(history)
        print(
            f"resumed_from={latest_path} "
            f"start_epoch={start_epoch:03d} "
            f"best_val_exact={best_val_exact_accuracy:.5f} "
            f"best_val_loss={best_val_loss:.5f}",
            flush=True,
        )

    for epoch in range(start_epoch, args.epochs + 1):
        train_loader = make_loader(
            shard_paths,
            split="train",
            batch_size=args.batch_size,
            seed=args.seed,
            epoch=epoch,
            shuffle=True,
            max_examples=args.train_examples_per_epoch,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
        val_loader = make_loader(
            shard_paths,
            split="val",
            batch_size=args.batch_size,
            seed=args.seed,
            epoch=0,
            shuffle=False,
            max_examples=args.eval_examples,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
        train_metrics = train_one_epoch(
            model,
            train_loader,
            device=device,
            optimizer=optimizer,
            criterion=criterion,
            grad_clip=args.grad_clip,
            use_amp=use_amp,
            length=config.length,
            absolute_depth=config.absolute_depth,
            simple_mats=simple_mats,
            boundary_only_radius=args.boundary_only_radius,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            device=device,
            criterion=criterion,
            use_amp=use_amp,
            length=config.length,
            absolute_depth=config.absolute_depth,
            simple_mats=simple_mats,
            boundary_only_radius=args.boundary_only_radius,
        )
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics, "lr": args.lr}
        history.append(row)
        atomic_json_dump(history, history_path)

        improved = (val_metrics["exact_set_accuracy"] > best_val_exact_accuracy) or (
            val_metrics["exact_set_accuracy"] == best_val_exact_accuracy and val_metrics["loss"] < best_val_loss
        )
        if improved:
            best_val_exact_accuracy = val_metrics["exact_set_accuracy"]
            best_val_loss = val_metrics["loss"]

        payload = checkpoint_payload(
            model=model,
            optimizer=optimizer,
            args=args,
            epoch=epoch,
            history=history,
            best_val_exact_accuracy=best_val_exact_accuracy,
            best_val_loss=best_val_loss,
        )
        atomic_torch_save(payload, latest_path)
        if improved:
            atomic_torch_save(payload, best_path)

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.5f} "
            f"train_exact={train_metrics['exact_set_accuracy']:.5f} "
            f"train_bit={train_metrics['bit_accuracy']:.5f} "
            f"val_loss={val_metrics['loss']:.5f} "
            f"val_exact={val_metrics['exact_set_accuracy']:.5f} "
            f"val_bit={val_metrics['bit_accuracy']:.5f} "
            f"val_micro_f1={val_metrics['micro_f1']:.5f}",
            flush=True,
        )

    if best_path.exists():
        best_payload = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(best_payload["model_state"])
    test_loader = make_loader(
        shard_paths,
        split="test",
        batch_size=args.batch_size,
        seed=args.seed,
        epoch=0,
        shuffle=False,
        max_examples=args.test_examples,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_metrics = evaluate(
        model,
        test_loader,
        device=device,
        criterion=criterion,
        use_amp=use_amp,
        length=config.length,
        absolute_depth=config.absolute_depth,
        simple_mats=simple_mats,
        boundary_only_radius=args.boundary_only_radius,
    )
    atomic_json_dump({"history": history, "test": test_metrics}, out_dir / "results.json")
    print(
        f"test_loss={test_metrics['loss']:.5f} "
        f"test_exact={test_metrics['exact_set_accuracy']:.5f} "
        f"test_bit={test_metrics['bit_accuracy']:.5f} "
        f"test_micro_f1={test_metrics['micro_f1']:.5f}",
        flush=True,
    )
    print(f"best_checkpoint={best_path}", flush=True)


if __name__ == "__main__":
    main()
