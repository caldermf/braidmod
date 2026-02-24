import argparse
import json
import math
import os
from dataclasses import dataclass
from itertools import permutations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split


PERMUTATIONS_S4 = list(permutations(range(4)))
PERM_TO_ID = {perm: idx for idx, perm in enumerate(PERMUTATIONS_S4)}


@dataclass
class TrainConfig:
    data_path: str
    p: int
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
    use_aux_head: bool = True
    num_workers: int = 0
    grad_clip: float = 1.0
    out_dir: str = "artifacts"
    device: str = "auto"


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class BurauDataset(Dataset):
    """
    Expects JSON list records with:
      - burau_tensor: D x 3 x 3 ints in [0, p-1]
      - final_factor_perm: length-4 permutation
      - final_factor_right_descent: subset of [0,1,2]
    """

    def __init__(self, records, p: int):
        if not records:
            raise ValueError("Dataset is empty")
        self.p = p

        x = []
        y_perm = []
        y_desc = []
        for rec in records:
            tensor = rec["burau_tensor"]
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
        self.y_perm = torch.tensor(y_perm, dtype=torch.long)
        self.y_desc = torch.tensor(y_desc, dtype=torch.float32)
        self.D = self.x.shape[1]

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y_perm[idx], self.y_desc[idx]


class ResidualMLPBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class BurauEmbeddingMLP(nn.Module):
    """
    Embeds categorical mod-p entries + learned (depth,row,col) position embeddings,
    then applies a residual MLP trunk.
    """

    def __init__(
        self,
        p: int,
        D: int,
        embed_dim: int = 32,
        hidden_dim: int = 1024,
        blocks: int = 3,
        dropout: float = 0.1,
        use_aux_head: bool = True,
    ):
        super().__init__()
        self.p = p
        self.D = D
        self.use_aux_head = use_aux_head

        self.value_emb = nn.Embedding(p, embed_dim)
        self.depth_emb = nn.Embedding(D, embed_dim)
        self.row_emb = nn.Embedding(3, embed_dim)
        self.col_emb = nn.Embedding(3, embed_dim)

        flat_dim = D * 3 * 3 * embed_dim
        self.input_proj = nn.Sequential(
            nn.LayerNorm(flat_dim),
            nn.Linear(flat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.Sequential(
            *[ResidualMLPBlock(hidden_dim=hidden_dim, dropout=dropout) for _ in range(blocks)]
        )
        self.trunk_norm = nn.LayerNorm(hidden_dim)
        self.factor_head = nn.Linear(hidden_dim, len(PERMUTATIONS_S4))
        self.desc_head = nn.Linear(hidden_dim, 3)

        depth_idx = torch.arange(D).view(D, 1, 1)
        row_idx = torch.arange(3).view(1, 3, 1)
        col_idx = torch.arange(3).view(1, 1, 3)
        self.register_buffer("depth_idx", depth_idx, persistent=False)
        self.register_buffer("row_idx", row_idx, persistent=False)
        self.register_buffer("col_idx", col_idx, persistent=False)

    def forward(self, x):
        # x: [B, D, 3, 3] with integer values in [0, p-1]
        if x.shape[1] != self.D:
            raise ValueError(f"Model D={self.D}, got input depth {x.shape[1]}")

        v = self.value_emb(x)  # [B, D, 3, 3, E]
        pos = (
            self.depth_emb(self.depth_idx)
            + self.row_emb(self.row_idx)
            + self.col_emb(self.col_idx)
        )  # [D, 3, 3, E]
        h = v + pos.unsqueeze(0)
        h = h.flatten(start_dim=1)
        h = self.input_proj(h)
        h = self.blocks(h)
        h = self.trunk_norm(h)

        factor_logits = self.factor_head(h)
        desc_logits = self.desc_head(h) if self.use_aux_head else None
        return factor_logits, desc_logits


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return (preds == targets).float().mean().item()


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


def evaluate(model, loader, device, ce_loss, bce_loss, aux_weight):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_n = 0
    with torch.no_grad():
        for x, y_perm, y_desc in loader:
            x = x.to(device, non_blocking=True)
            y_perm = y_perm.to(device, non_blocking=True)
            y_desc = y_desc.to(device, non_blocking=True)

            factor_logits, desc_logits = model(x)
            loss = ce_loss(factor_logits, y_perm)
            if desc_logits is not None and aux_weight > 0:
                loss = loss + aux_weight * bce_loss(desc_logits, y_desc)

            n = x.size(0)
            total_n += n
            total_loss += loss.item() * n
            total_acc += accuracy_from_logits(factor_logits, y_perm) * n
    return total_loss / total_n, total_acc / total_n


def train(config: TrainConfig):
    set_seed(config.seed)
    os.makedirs(config.out_dir, exist_ok=True)

    with open(config.data_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    dataset = BurauDataset(records, p=config.p)
    train_loader, val_loader = make_loaders(
        dataset=dataset,
        batch_size=config.batch_size,
        val_fraction=config.val_fraction,
        seed=config.seed,
        num_workers=config.num_workers,
    )

    device = resolve_device(config.device)
    model = BurauEmbeddingMLP(
        p=config.p,
        D=dataset.D,
        embed_dim=config.embed_dim,
        hidden_dim=config.hidden_dim,
        blocks=config.blocks,
        dropout=config.dropout,
        use_aux_head=config.use_aux_head,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    ce_loss = nn.CrossEntropyLoss()
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

    best_val_acc = -1.0
    best_path = os.path.join(config.out_dir, "best_model.pt")

    global_step = 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        running_n = 0

        for x, y_perm, y_desc in train_loader:
            x = x.to(device, non_blocking=True)
            y_perm = y_perm.to(device, non_blocking=True)
            y_desc = y_desc.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                factor_logits, desc_logits = model(x)
                loss = ce_loss(factor_logits, y_perm)
                if desc_logits is not None and config.aux_weight > 0:
                    loss = loss + config.aux_weight * bce_loss(desc_logits, y_desc)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            n = x.size(0)
            running_n += n
            running_loss += loss.item() * n
            running_acc += accuracy_from_logits(factor_logits, y_perm) * n
            global_step += 1

        train_loss = running_loss / running_n
        train_acc = running_acc / running_n
        val_loss, val_acc = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            ce_loss=ce_loss,
            bce_loss=bce_loss,
            aux_weight=config.aux_weight if config.use_aux_head else 0.0,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": vars(config),
                    "D": dataset.D,
                    "p": config.p,
                    "perm_classes": [list(p_) for p_ in PERMUTATIONS_S4],
                    "best_val_acc": best_val_acc,
                },
                best_path,
            )

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )

    print(f"best_val_acc={best_val_acc:.4f}")
    print(f"best_checkpoint={best_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train GPU MLP on Burau tensors to predict final Garside factor.")
    parser.add_argument("--data-path", type=str, required=True, help="Path to JSON dataset (list of records)")
    parser.add_argument("--p", type=int, required=True, help="Prime modulus for input tokens")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--aux-weight", type=float, default=0.2)
    parser.add_argument("--no-aux-head", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--out-dir", type=str, default="artifacts")
    parser.add_argument("--device", type=str, default="auto", help="auto, cuda, cpu, cuda:0, ...")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = TrainConfig(
        data_path=args.data_path,
        p=args.p,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_fraction=args.val_fraction,
        seed=args.seed,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        blocks=args.blocks,
        dropout=args.dropout,
        aux_weight=args.aux_weight,
        use_aux_head=not args.no_aux_head,
        num_workers=args.num_workers,
        grad_clip=args.grad_clip,
        out_dir=args.out_dir,
        device=args.device,
    )
    train(cfg)
