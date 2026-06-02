#!/usr/bin/env python3
"""Train the B_4 absolute-degree Burau transformer."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from interp.b4_data import B4ShardBatchIterable, discover_shards, load_shard  # noqa: E402
from interp.b4_transformer import B4AbsoluteTransformer, B4TransformerConfig  # noqa: E402


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


def infer_data_config(shard_paths: list[Path]) -> dict:
    meta = load_shard(shard_paths[0])["metadata"]
    return {
        "length": int(meta["length"]),
        "absolute_depth": int(meta["absolute_depth"]),
        "vocab_size": 1 << (int(meta["matrix_size"]) * int(meta["matrix_size"])),
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
    dataset = B4ShardBatchIterable(
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


def mask_from_bits(bits: torch.Tensor) -> torch.Tensor:
    bits = bits.to(torch.long)
    weights = torch.tensor([1, 2, 4], device=bits.device, dtype=torch.long).view(1, 3)
    return (bits * weights).sum(dim=1)


def init_totals() -> dict:
    return {
        "n": 0,
        "loss_sum": 0.0,
        "exact_correct": 0,
        "bit_correct": 0,
        "per_label_correct": [0, 0, 0],
        "per_label_positive_true": [0, 0, 0],
        "per_label_positive_pred": [0, 0, 0],
        "tp": [0, 0, 0],
        "fp": [0, 0, 0],
        "fn": [0, 0, 0],
        "true_mask_counts": [0] * 8,
        "pred_mask_counts": [0] * 8,
    }


def update_metrics(totals: dict, logits: torch.Tensor, labels: torch.Tensor, loss: torch.Tensor) -> None:
    with torch.no_grad():
        labels = labels.to(torch.long)
        preds = (logits.detach() >= 0).to(torch.long)
        n = int(labels.shape[0])
        totals["n"] += n
        totals["loss_sum"] += float(loss.detach().item()) * n
        exact = preds.eq(labels).all(dim=1)
        totals["exact_correct"] += int(exact.sum().item())
        totals["bit_correct"] += int(preds.eq(labels).sum().item())
        for idx in range(3):
            pred_i = preds[:, idx]
            label_i = labels[:, idx]
            totals["per_label_correct"][idx] += int(pred_i.eq(label_i).sum().item())
            totals["per_label_positive_true"][idx] += int(label_i.sum().item())
            totals["per_label_positive_pred"][idx] += int(pred_i.sum().item())
            totals["tp"][idx] += int(((pred_i == 1) & (label_i == 1)).sum().item())
            totals["fp"][idx] += int(((pred_i == 1) & (label_i == 0)).sum().item())
            totals["fn"][idx] += int(((pred_i == 0) & (label_i == 1)).sum().item())
        true_masks = torch.bincount(mask_from_bits(labels).cpu(), minlength=8)
        pred_masks = torch.bincount(mask_from_bits(preds).cpu(), minlength=8)
        for idx in range(8):
            totals["true_mask_counts"][idx] += int(true_masks[idx].item())
            totals["pred_mask_counts"][idx] += int(pred_masks[idx].item())


def finish_metrics(totals: dict) -> dict:
    n = max(1, int(totals["n"]))
    bit_total = max(1, 3 * n)
    f1 = []
    for idx in range(3):
        tp = totals["tp"][idx]
        fp = totals["fp"][idx]
        fn = totals["fn"][idx]
        denom = 2 * tp + fp + fn
        f1.append(0.0 if denom == 0 else (2 * tp) / denom)
    micro_tp = sum(totals["tp"])
    micro_fp = sum(totals["fp"])
    micro_fn = sum(totals["fn"])
    micro_denom = 2 * micro_tp + micro_fp + micro_fn
    return {
        "loss": totals["loss_sum"] / n,
        "exact_set_accuracy": totals["exact_correct"] / n,
        "bit_accuracy": totals["bit_correct"] / bit_total,
        "per_label_accuracy": [x / n for x in totals["per_label_correct"]],
        "per_label_true_positive_rate": [x / n for x in totals["per_label_positive_true"]],
        "per_label_pred_positive_rate": [x / n for x in totals["per_label_positive_pred"]],
        "per_label_f1": f1,
        "micro_f1": 0.0 if micro_denom == 0 else (2 * micro_tp) / micro_denom,
        "true_mask_counts": totals["true_mask_counts"],
        "pred_mask_counts": totals["pred_mask_counts"],
        "n": n,
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
    totals = init_totals()
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for batch in loader:
        tokens = batch["tokens"].to(device, non_blocking=True)
        labels = batch["label_bits"].to(device, non_blocking=True)
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
) -> dict:
    model.eval()
    totals = init_totals()
    for batch in loader:
        tokens = batch["tokens"].to(device, non_blocking=True)
        labels = batch["label_bits"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(tokens)
            loss = criterion(logits, labels)
        update_metrics(totals, logits.detach().cpu(), labels.detach().cpu(), loss.detach().cpu())
    return finish_metrics(totals)


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
        "input_format": "absolute_degree_slice_tokens_0_to_511",
    }


def best_from_history(history: list[dict]) -> tuple[float, float]:
    best_exact = -math.inf
    best_loss = math.inf
    for row in history:
        val_metrics = row.get("val", {})
        exact = float(val_metrics.get("exact_set_accuracy", -math.inf))
        loss = float(val_metrics.get("loss", math.inf))
        if exact > best_exact or (exact == best_exact and loss < best_loss):
            best_exact = exact
            best_loss = loss
    return best_exact, best_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a B_4 Burau descent-set transformer.")
    parser.add_argument("--data-dir", default="interp/data/generated/b4_l25_p2_n16777216")
    parser.add_argument("--out-dir", default="interp/artifacts/b4_l25_p2_xfmr3_abs")
    parser.add_argument("--num-shards", type=int, default=64)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--train-examples-per-epoch", type=int, default=8_388_608)
    parser.add_argument("--eval-examples", type=int, default=524_288)
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_paths = discover_shards(args.data_dir, num_shards=args.num_shards, allow_partial=args.allow_partial)
    data_config = infer_data_config(shard_paths)

    device = resolve_device(args.device)
    use_amp = device.type == "cuda" and not args.no_amp
    pin_memory = device.type == "cuda"
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
            },
            indent=2,
        )
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
        )
        val_metrics = evaluate(model, val_loader, device=device, criterion=criterion, use_amp=use_amp)
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
    test_metrics = evaluate(model, test_loader, device=device, criterion=criterion, use_amp=use_amp)
    atomic_json_dump({"history": history, "test": test_metrics}, out_dir / "results.json")
    print(
        f"test_loss={test_metrics['loss']:.5f} "
        f"test_exact={test_metrics['exact_set_accuracy']:.5f} "
        f"test_bit={test_metrics['bit_accuracy']:.5f} "
        f"test_micro_f1={test_metrics['micro_f1']:.5f}"
    )
    print(f"best_checkpoint={best_path}")


if __name__ == "__main__":
    main()
