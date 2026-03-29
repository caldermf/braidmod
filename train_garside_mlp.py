import argparse
import copy
import json
import math
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from garside_models import PERMUTATIONS_S4, build_model_from_config


PERM_TO_ID = {perm: idx for idx, perm in enumerate(PERMUTATIONS_S4)}


@dataclass
class TrainConfig:
    data_path: str
    p: int
    model_type: str = "mlp"
    batch_size: int = 256
    epochs: int = 40
    lr: float = 3e-4
    weight_decay: float = 1e-2
    val_fraction: float = 0.1
    seed: int = 42
    embed_dim: int = 32
    hidden_dim: int = 1024
    blocks: int = 3
    dropout: float = 0.1
    aux_weight: float = 0.2
    task: str = "multitask"  # one of: final_factor, right_descent, multitask
    num_workers: int = 0
    grad_clip: float = 1.0
    out_dir: str = "artifacts"
    device: str = "auto"
    use_min_degree: bool = True
    use_garside_length: bool = False
    matrix_size: int = 3
    d_model: int = 256
    ffn_mult: float = 4.0
    num_local_blocks: int = 2
    num_local_heads: int = 4
    num_global_blocks: int = 6
    num_global_heads: int = 8
    label_smoothing: float = 0.0
    ema_decay: float = 0.0
    selection_objective: str = "metric"  # one of: metric, loss


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class BurauDataset(Dataset):
    """
    Expects JSON list records with:
      - burau_tensor: D x 3 x 3 ints in [0, p-1]
      - burau_min_degree: optional integer projective shift (defaults to 0)
      - final_factor_perm: length-4 permutation
      - final_factor_right_descent: subset of [0,1,2]
    """

    def __init__(self, records, p: int):
        if not records:
            raise ValueError("Dataset is empty")
        self.p = p

        x = []
        min_degrees = []
        garside_lengths = []
        y_perm = []
        y_desc = []
        for rec in records:
            tensor = rec["burau_tensor"]
            min_degrees.append(int(rec.get("burau_min_degree", 0)))
            if "gnf_factors" in rec:
                garside_lengths.append(len(rec["gnf_factors"]))
            elif "garside_length" in rec:
                garside_lengths.append(int(rec["garside_length"]))
            elif "gnf_length" in rec:
                garside_lengths.append(int(rec["gnf_length"]))
            else:
                garside_lengths.append(0)
            perm = tuple(rec["final_factor_perm"])
            rdesc = rec.get("final_factor_right_descent", [])

            if perm not in PERM_TO_ID:
                raise ValueError(f"Permutation not in S4: {perm}")
            y_perm.append(PERM_TO_ID[perm])

            desc_vec = [0.0, 0.0, 0.0]
            for d in rdesc:
                if d not in (0, 1, 2):
                    raise ValueError(f"Invalid descent index {d}; expected 0,1,2")
                desc_vec[d] = 1.0
            y_desc.append(desc_vec)

            x.append(tensor)

        x_tensor = torch.tensor(x, dtype=torch.long)
        if x_tensor.ndim != 4 or x_tensor.shape[2:] != (3, 3):
            raise ValueError("Expected burau_tensor with shape [N, D, 3, 3]")
        if x_tensor.min().item() < 0 or x_tensor.max().item() >= p:
            raise ValueError(f"Input tensor values must be in [0, {p - 1}]")

        self.x = x_tensor
        self.min_degrees = torch.tensor(min_degrees, dtype=torch.float32)
        self.garside_lengths = torch.tensor(garside_lengths, dtype=torch.float32)
        self.y_perm = torch.tensor(y_perm, dtype=torch.long)
        self.y_desc = torch.tensor(y_desc, dtype=torch.float32)
        self.D = self.x.shape[1]
        self.matrix_size = self.x.shape[2]

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return (
            self.x[idx],
            self.min_degrees[idx],
            self.garside_lengths[idx],
            self.y_perm[idx],
            self.y_desc[idx],
        )


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return (preds == targets).float().mean().item()


def descent_exact_match_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = (torch.sigmoid(logits) >= 0.5).float()
    exact = (preds == targets).all(dim=-1).float()
    return exact.mean().item()


def make_loaders(dataset: BurauDataset, batch_size: int, val_fraction: float, seed: int, num_workers: int):
    if not (0.0 < val_fraction < 1.0):
        raise ValueError("val_fraction must be in (0,1)")
    val_size = max(1, int(len(dataset) * val_fraction))
    train_size = len(dataset) - val_size
    if train_size < 1:
        raise ValueError("Not enough data after split; increase dataset size")

    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )
    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )
    return train_loader, val_loader


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)


def cosine_with_warmup(step: int, total_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return float(step + 1) / float(max(1, warmup_steps))
    progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def compute_task_loss_and_metric(task, factor_logits, desc_logits, y_perm, y_desc, ce_loss, bce_loss, aux_weight):
    if task == "right_descent":
        if desc_logits is None:
            raise ValueError("desc_logits is None for right_descent task")
        loss = bce_loss(desc_logits, y_desc)
        metric = descent_exact_match_from_logits(desc_logits, y_desc)
        metric_name = "desc_exact"
        return loss, metric, metric_name

    if task == "final_factor":
        loss = ce_loss(factor_logits, y_perm)
        metric = accuracy_from_logits(factor_logits, y_perm)
        metric_name = "factor_acc"
        return loss, metric, metric_name

    if task == "multitask":
        if desc_logits is None:
            raise ValueError("desc_logits is None for multitask")
        loss = ce_loss(factor_logits, y_perm) + aux_weight * bce_loss(desc_logits, y_desc)
        metric = accuracy_from_logits(factor_logits, y_perm)
        metric_name = "factor_acc"
        return loss, metric, metric_name

    raise ValueError(f"Unknown task: {task}")


def evaluate(model, loader, device, ce_loss, bce_loss, aux_weight, task):
    model.eval()
    total_loss = 0.0
    total_metric = 0.0
    total_n = 0
    metric_name = None
    with torch.no_grad():
        for x, min_degree, garside_length, y_perm, y_desc in loader:
            x = x.to(device, non_blocking=True)
            min_degree = min_degree.to(device, non_blocking=True)
            garside_length = garside_length.to(device, non_blocking=True)
            y_perm = y_perm.to(device, non_blocking=True)
            y_desc = y_desc.to(device, non_blocking=True)

            factor_logits, desc_logits = model(
                x,
                min_degree=min_degree,
                garside_length=garside_length,
            )
            loss, metric, metric_name = compute_task_loss_and_metric(
                task=task,
                factor_logits=factor_logits,
                desc_logits=desc_logits,
                y_perm=y_perm,
                y_desc=y_desc,
                ce_loss=ce_loss,
                bce_loss=bce_loss,
                aux_weight=aux_weight,
            )

            n = x.size(0)
            total_n += n
            total_loss += loss.item() * n
            total_metric += metric * n
    return total_loss / total_n, total_metric / total_n, metric_name


@torch.no_grad()
def update_ema_model(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    ema_state = ema_model.state_dict()
    model_state = model.state_dict()
    for name, ema_value in ema_state.items():
        model_value = model_state[name].detach()
        if torch.is_floating_point(ema_value):
            ema_value.mul_(decay).add_(model_value, alpha=1.0 - decay)
        else:
            ema_value.copy_(model_value)


def train(config: TrainConfig):
    set_seed(config.seed)
    os.makedirs(config.out_dir, exist_ok=True)

    with open(config.data_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    dataset = BurauDataset(records, p=config.p)
    if config.use_garside_length and torch.any(dataset.garside_lengths <= 0):
        raise ValueError(
            "Dataset is missing positive Garside lengths for some records; "
            "expected 'gnf_factors', 'garside_length', or 'gnf_length'."
        )
    train_loader, val_loader = make_loaders(
        dataset=dataset,
        batch_size=config.batch_size,
        val_fraction=config.val_fraction,
        seed=config.seed,
        num_workers=config.num_workers,
    )

    device = resolve_device(config.device)
    config.matrix_size = dataset.matrix_size
    model = build_model_from_config(
        vars(config),
        p=config.p,
        D=dataset.D,
        matrix_size=dataset.matrix_size,
    ).to(device)
    num_params = sum(param.numel() for param in model.parameters())
    print(f"model_type={config.model_type} num_parameters={num_params}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    ce_loss_train = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    ce_loss_eval = nn.CrossEntropyLoss()
    bce_loss = nn.BCEWithLogitsLoss()

    total_steps = config.epochs * len(train_loader)
    warmup_steps = max(10, int(0.05 * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: cosine_with_warmup(s, total_steps=total_steps, warmup_steps=warmup_steps),
    )

    use_amp = device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    ema_model = None
    if config.ema_decay > 0.0:
        if not (0.0 < config.ema_decay < 1.0):
            raise ValueError("ema_decay must be in (0,1)")
        ema_model = copy.deepcopy(model)
        ema_model.requires_grad_(False)
        ema_model.eval()

    best_val_metric = -1.0
    best_val_loss = float("inf")
    best_path = os.path.join(config.out_dir, "best_model.pt")
    best_metric_path = os.path.join(config.out_dir, "best_metric_model.pt")
    best_loss_path = os.path.join(config.out_dir, "best_loss_model.pt")
    history_path = os.path.join(config.out_dir, "history.json")
    history = []

    def make_checkpoint_payload(best_metric_value: float, best_loss_value: float, eval_model_name: str):
        source_model = ema_model if ema_model is not None else model
        return {
            "model_state": source_model.state_dict(),
            "config": vars(config),
            "D": dataset.D,
            "p": config.p,
            "matrix_size": dataset.matrix_size,
            "perm_classes": [list(p_) for p_ in PERMUTATIONS_S4],
            "best_val_metric": best_metric_value,
            "best_val_loss": best_loss_value,
            "best_metric_name": metric_name,
            "selection_objective": config.selection_objective,
            "eval_model": eval_model_name,
        }

    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        running_metric = 0.0
        running_n = 0
        metric_name = None

        for x, min_degree, garside_length, y_perm, y_desc in train_loader:
            x = x.to(device, non_blocking=True)
            min_degree = min_degree.to(device, non_blocking=True)
            garside_length = garside_length.to(device, non_blocking=True)
            y_perm = y_perm.to(device, non_blocking=True)
            y_desc = y_desc.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                factor_logits, desc_logits = model(
                    x,
                    min_degree=min_degree,
                    garside_length=garside_length,
                )
                opt_loss, _, _ = compute_task_loss_and_metric(
                    task=config.task,
                    factor_logits=factor_logits,
                    desc_logits=desc_logits,
                    y_perm=y_perm,
                    y_desc=y_desc,
                    ce_loss=ce_loss_train,
                    bce_loss=bce_loss,
                    aux_weight=config.aux_weight,
                )
                log_loss, metric, metric_name = compute_task_loss_and_metric(
                    task=config.task,
                    factor_logits=factor_logits.detach(),
                    desc_logits=None if desc_logits is None else desc_logits.detach(),
                    y_perm=y_perm,
                    y_desc=y_desc,
                    ce_loss=ce_loss_eval,
                    bce_loss=bce_loss,
                    aux_weight=config.aux_weight,
                )

            scaler.scale(opt_loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            if ema_model is not None:
                update_ema_model(ema_model, model, decay=config.ema_decay)

            n = x.size(0)
            running_n += n
            running_loss += log_loss.item() * n
            running_metric += metric * n
        train_loss = running_loss / running_n
        train_metric = running_metric / running_n
        eval_model = ema_model if ema_model is not None else model
        val_loss, val_metric, metric_name = evaluate(
            model=eval_model,
            loader=val_loader,
            device=device,
            ce_loss=ce_loss_eval,
            bce_loss=bce_loss,
            aux_weight=config.aux_weight,
            task=config.task,
        )

        eval_model_name = "ema" if ema_model is not None else "raw"
        if val_metric > best_val_metric:
            best_val_metric = val_metric
            payload = make_checkpoint_payload(best_val_metric, best_val_loss, eval_model_name)
            torch.save(payload, best_metric_path)
            if config.selection_objective == "metric":
                torch.save(payload, best_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            payload = make_checkpoint_payload(best_val_metric, best_val_loss, eval_model_name)
            torch.save(payload, best_loss_path)
            if config.selection_objective == "loss":
                torch.save(payload, best_path)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "metric_name": metric_name,
                "train_metric": train_metric,
                "val_metric": val_metric,
                "lr": scheduler.get_last_lr()[0],
                "eval_model": eval_model_name,
            }
        )
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.4f} train_{metric_name}={train_metric:.4f} "
            f"val_loss={val_loss:.4f} val_{metric_name}={val_metric:.4f} "
            f"lr={scheduler.get_last_lr()[0]:.2e} "
            f"eval_model={eval_model_name}"
        )

    if config.selection_objective == "metric" and not os.path.exists(best_path):
        torch.save(make_checkpoint_payload(best_val_metric, best_val_loss, eval_model_name), best_path)
    if config.selection_objective == "loss" and not os.path.exists(best_path):
        torch.save(make_checkpoint_payload(best_val_metric, best_val_loss, eval_model_name), best_path)

    print(f"best_val_loss={best_val_loss:.4f}")
    print(f"best_val_{metric_name}={best_val_metric:.4f}")
    print(f"best_loss_checkpoint={best_loss_path}")
    print(f"best_metric_checkpoint={best_metric_path}")
    print(f"best_checkpoint={best_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a Burau-tensor model to predict the final Garside factor."
    )
    parser.add_argument("--data-path", type=str, required=True, help="Path to JSON dataset (list of records)")
    parser.add_argument("--p", type=int, required=True, help="Prime modulus for input tokens")
    parser.add_argument(
        "--model-type",
        type=str,
        default="mlp",
        choices=["mlp", "transformer"],
        help="Architecture to train.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--ffn-mult", type=float, default=4.0)
    parser.add_argument("--num-local-blocks", type=int, default=2)
    parser.add_argument("--num-local-heads", type=int, default=4)
    parser.add_argument("--num-global-blocks", type=int, default=6)
    parser.add_argument("--num-global-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--ema-decay", type=float, default=0.0)
    parser.add_argument(
        "--selection-objective",
        type=str,
        default="metric",
        choices=["metric", "loss"],
        help="Which validation quantity best_model.pt should optimize.",
    )
    parser.add_argument("--aux-weight", type=float, default=0.2)
    parser.add_argument(
        "--task",
        type=str,
        default="multitask",
        choices=["final_factor", "right_descent", "multitask"],
        help="Training objective.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--out-dir", type=str, default="artifacts")
    parser.add_argument("--device", type=str, default="auto", help="auto, cuda, cpu, cuda:0, ...")
    parser.add_argument(
        "--no-min-degree",
        action="store_true",
        help="Disable the extra projective min-degree feature for compatibility experiments.",
    )
    parser.add_argument(
        "--use-garside-length",
        action="store_true",
        help="Condition the model on the true Garside length as an extra scalar input.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = TrainConfig(
        data_path=args.data_path,
        p=args.p,
        model_type=args.model_type,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_fraction=args.val_fraction,
        seed=args.seed,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        blocks=args.blocks,
        d_model=args.d_model,
        ffn_mult=args.ffn_mult,
        num_local_blocks=args.num_local_blocks,
        num_local_heads=args.num_local_heads,
        num_global_blocks=args.num_global_blocks,
        num_global_heads=args.num_global_heads,
        dropout=args.dropout,
        label_smoothing=args.label_smoothing,
        ema_decay=args.ema_decay,
        selection_objective=args.selection_objective,
        aux_weight=args.aux_weight,
        task=args.task,
        num_workers=args.num_workers,
        grad_clip=args.grad_clip,
        out_dir=args.out_dir,
        device=args.device,
        use_min_degree=not args.no_min_degree,
        use_garside_length=args.use_garside_length,
    )
    train(cfg)
