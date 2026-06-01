#!/usr/bin/env python3
"""Train the B_3 absolute-degree Burau transformer."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from interp.b3_data import B3ShardBatchIterable, discover_shards  # noqa: E402
from interp.b3_transformer import B3AbsoluteTransformer, B3TransformerConfig  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def atomic_json_dump(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    tmp_path.replace(path)


def make_loader(
    shard_paths: list[Path],
    *,
    split: str,
    length: int,
    batch_size: int,
    seed: int,
    epoch: int,
    shuffle: bool,
    max_examples: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    dataset = B3ShardBatchIterable(
        shard_paths,
        split=split,
        length=length,
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


def logits_to_preds(logits: torch.Tensor) -> torch.Tensor:
    return (logits >= 0).to(torch.long)


def update_metrics(totals: dict, logits: torch.Tensor, labels: torch.Tensor, min_degree: torch.Tensor) -> None:
    with torch.no_grad():
        labels_long = labels.to(torch.long)
        preds = logits_to_preds(logits.detach())
        n = int(labels.numel())
        totals["n"] += n
        totals["correct"] += int((preds == labels_long).sum().item())
        totals["label_0"] += int((labels_long == 0).sum().item())
        totals["label_1"] += int((labels_long == 1).sum().item())
        parity_pred = (min_degree.to(torch.long) % 2 == 0).to(torch.long)
        totals["min_degree_even_baseline_correct"] += int((parity_pred == labels_long.cpu()).sum().item())


def finish_metrics(totals: dict, loss_sum: float) -> dict:
    n = max(1, totals["n"])
    return {
        "loss": loss_sum / n,
        "accuracy": totals["correct"] / n,
        "n": totals["n"],
        "label_counts": [totals["label_0"], totals["label_1"]],
        "min_degree_even_baseline_accuracy": totals["min_degree_even_baseline_correct"] / n,
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    grad_clip: float,
    use_amp: bool,
) -> dict:
    model.train()
    loss_sum = 0.0
    totals = {
        "n": 0,
        "correct": 0,
        "label_0": 0,
        "label_1": 0,
        "min_degree_even_baseline_correct": 0,
    }
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for batch in loader:
        tokens = batch["tokens"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
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

        n = int(labels.numel())
        loss_sum += float(loss.detach().item()) * n
        update_metrics(totals, logits.detach().cpu(), labels.detach().cpu(), batch["min_degree"])
    return finish_metrics(totals, loss_sum)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    criterion: nn.Module,
    use_amp: bool,
) -> dict:
    model.eval()
    loss_sum = 0.0
    totals = {
        "n": 0,
        "correct": 0,
        "label_0": 0,
        "label_1": 0,
        "min_degree_even_baseline_correct": 0,
    }
    for batch in loader:
        tokens = batch["tokens"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(tokens)
            loss = criterion(logits, labels)
        n = int(labels.numel())
        loss_sum += float(loss.detach().item()) * n
        update_metrics(totals, logits.detach().cpu(), labels.detach().cpu(), batch["min_degree"])
    return finish_metrics(totals, loss_sum)


def checkpoint_payload(
    *,
    model: B3AbsoluteTransformer,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    epoch: int,
    history: list[dict],
    best_val_accuracy: float,
    best_val_loss: float,
) -> dict:
    return {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "model_config": model.config.to_dict(),
        "args": vars(args),
        "epoch": epoch,
        "history": history,
        "best_val_accuracy": best_val_accuracy,
        "best_val_loss": best_val_loss,
        "label_convention": {"0": "{s_1}", "1": "{s_2}"},
        "input_format": "absolute_degree_slice_tokens_0_to_15",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small B_3 Burau transformer.")
    parser.add_argument("--data-dir", default="interp/data/generated/b3_l25_p2_full")
    parser.add_argument("--out-dir", default="interp/artifacts/b3_l25_p2_xfmr2_abs")
    parser.add_argument("--length", type=int, default=25)
    parser.add_argument("--num-shards", type=int, default=64)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--train-examples-per-epoch", type=int, default=16_777_216)
    parser.add_argument("--eval-examples", type=int, default=1_048_576)
    parser.add_argument("--test-examples", type=int, default=1_048_576)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--ffn-mult", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_paths = discover_shards(args.data_dir, num_shards=args.num_shards, allow_partial=args.allow_partial)

    device = resolve_device(args.device)
    use_amp = device.type == "cuda" and not args.no_amp
    pin_memory = device.type == "cuda"
    config = B3TransformerConfig(
        length=args.length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ffn_mult=args.ffn_mult,
        dropout=args.dropout,
    )
    model = B3AbsoluteTransformer(config).to(device)
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
            },
            indent=2,
        )
    )

    history: list[dict] = []
    best_val_accuracy = -math.inf
    best_val_loss = math.inf
    best_path = out_dir / "best_model.pt"
    latest_path = out_dir / "latest_model.pt"
    history_path = out_dir / "history.json"

    for epoch in range(1, args.epochs + 1):
        train_loader = make_loader(
            shard_paths,
            split="train",
            length=args.length,
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
            length=args.length,
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
        )
        val_metrics = evaluate(model, val_loader, device=device, criterion=criterion, use_amp=use_amp)
        row = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "lr": args.lr,
        }
        history.append(row)
        atomic_json_dump(history, history_path)

        payload = checkpoint_payload(
            model=model,
            optimizer=optimizer,
            args=args,
            epoch=epoch,
            history=history,
            best_val_accuracy=best_val_accuracy,
            best_val_loss=best_val_loss,
        )
        atomic_torch_save(payload, latest_path)

        improved = (val_metrics["accuracy"] > best_val_accuracy) or (
            val_metrics["accuracy"] == best_val_accuracy and val_metrics["loss"] < best_val_loss
        )
        if improved:
            best_val_accuracy = val_metrics["accuracy"]
            best_val_loss = val_metrics["loss"]
            payload["best_val_accuracy"] = best_val_accuracy
            payload["best_val_loss"] = best_val_loss
            atomic_torch_save(payload, best_path)

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.5f} train_acc={train_metrics['accuracy']:.5f} "
            f"val_loss={val_metrics['loss']:.5f} val_acc={val_metrics['accuracy']:.5f} "
            f"val_min_deg_even_baseline={val_metrics['min_degree_even_baseline_accuracy']:.5f}"
        )

    if best_path.exists():
        best_payload = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(best_payload["model_state"])
    test_loader = make_loader(
        shard_paths,
        split="test",
        length=args.length,
        batch_size=args.batch_size,
        seed=args.seed,
        epoch=0,
        shuffle=False,
        max_examples=args.test_examples,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_metrics = evaluate(model, test_loader, device=device, criterion=criterion, use_amp=use_amp)
    atomic_json_dump({"history": history, "test": test_metrics}, out_dir / "results.json")
    print(
        f"test_loss={test_metrics['loss']:.5f} test_acc={test_metrics['accuracy']:.5f} "
        f"test_min_deg_even_baseline={test_metrics['min_degree_even_baseline_accuracy']:.5f}"
    )
    print(f"best_checkpoint={best_path}")


if __name__ == "__main__":
    main()
