"""TransformerLens-style tools for the B_4 absolute-degree transformer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import torch
import torch.nn.functional as F

from interp.b4_transformer import B4AbsoluteTransformer, B4TransformerConfig


HookFn = Callable[[torch.Tensor, str], torch.Tensor]


@dataclass
class LoadedB4Transformer:
    model: B4AbsoluteTransformer
    checkpoint: dict


def load_transformer_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> LoadedB4Transformer:
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    cfg = checkpoint["model_config"]
    config = B4TransformerConfig(
        length=int(cfg["length"]),
        absolute_depth=int(cfg["absolute_depth"]),
        vocab_size=int(cfg["vocab_size"]),
        num_labels=int(cfg["num_labels"]),
        d_model=int(cfg["d_model"]),
        num_layers=int(cfg["num_layers"]),
        num_heads=int(cfg["num_heads"]),
        ffn_mult=int(cfg["ffn_mult"]),
        dropout=float(cfg.get("dropout", 0.0)),
    )
    model = B4AbsoluteTransformer(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return LoadedB4Transformer(model=model, checkpoint=checkpoint)


def multilabel_logit_score(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    labels = labels.to(device=logits.device, dtype=logits.dtype)
    direction = labels.mul(2).sub(1)
    return logits * direction


def mask_from_bits(bits: torch.Tensor) -> torch.Tensor:
    bits = bits.to(torch.long)
    weights = torch.tensor([1, 2, 4], device=bits.device, dtype=torch.long).view(1, 3)
    return (bits * weights).sum(dim=1)


def metrics_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    labels = labels.to(device=logits.device, dtype=torch.float32)
    preds = (logits >= 0).to(torch.float32)
    loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="mean")
    exact = preds.eq(labels).all(dim=1)
    score = multilabel_logit_score(logits, labels)
    out = {
        "loss": float(loss.item()),
        "exact_set_accuracy": float(exact.float().mean().item()),
        "bit_accuracy": float(preds.eq(labels).float().mean().item()),
        "mean_logit_score": float(score.mean().item()),
        "per_label_accuracy": [float(x) for x in preds.eq(labels).float().mean(dim=0).tolist()],
        "per_label_pred_positive_rate": [float(x) for x in preds.mean(dim=0).tolist()],
        "per_label_true_positive_rate": [float(x) for x in labels.mean(dim=0).tolist()],
        "n": int(labels.shape[0]),
    }
    tp = ((preds == 1) & (labels == 1)).sum(dim=0)
    fp = ((preds == 1) & (labels == 0)).sum(dim=0)
    fn = ((preds == 0) & (labels == 1)).sum(dim=0)
    denom = 2 * tp + fp + fn
    f1 = torch.where(denom > 0, 2 * tp / denom.clamp_min(1), torch.zeros_like(denom, dtype=torch.float32))
    micro_denom = 2 * tp.sum() + fp.sum() + fn.sum()
    out["per_label_f1"] = [float(x) for x in f1.tolist()]
    out["micro_f1"] = float((2 * tp.sum() / micro_denom.clamp_min(1)).item())
    out["true_mask_counts"] = torch.bincount(mask_from_bits(labels), minlength=8).cpu().tolist()
    out["pred_mask_counts"] = torch.bincount(mask_from_bits(preds), minlength=8).cpu().tolist()
    return out


def apply_hooks(
    name: str,
    value: torch.Tensor,
    hooks: Mapping[str, HookFn | torch.Tensor] | None,
) -> torch.Tensor:
    if hooks is None or name not in hooks:
        return value
    hook = hooks[name]
    if callable(hook):
        return hook(value, name)
    return hook.to(device=value.device, dtype=value.dtype)


def _split_qkv(block, attn_in: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    qkv = F.linear(attn_in, block.attn.in_proj_weight, block.attn.in_proj_bias)
    return qkv.chunk(3, dim=-1)


def _shape_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    batch, seq_len, d_model = x.shape
    head_dim = d_model // num_heads
    return x.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)


def _project_heads(block, z: torch.Tensor) -> torch.Tensor:
    d_model = block.attn.embed_dim
    num_heads = block.attn.num_heads
    head_dim = d_model // num_heads
    weight = block.attn.out_proj.weight.view(d_model, num_heads, head_dim)
    return torch.einsum("bhsd,mhd->bhsm", z, weight)


def run_with_cache(
    model: B4AbsoluteTransformer,
    tokens: torch.Tensor,
    *,
    hooks: Mapping[str, HookFn | torch.Tensor] | None = None,
    names_filter: set[str] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Run the B_4 transformer with named activation cache and optional hooks."""
    if tokens.ndim != 2 or tokens.shape[1] != model.absolute_depth:
        raise ValueError(f"Expected tokens with shape [B, {model.absolute_depth}], got {tuple(tokens.shape)}")

    cache: dict[str, torch.Tensor] = {}

    def save(name: str, value: torch.Tensor) -> torch.Tensor:
        value = apply_hooks(name, value, hooks)
        if names_filter is None or name in names_filter:
            cache[name] = value.detach()
        return value

    batch_size = tokens.shape[0]
    hidden = model.slice_emb(tokens)
    hidden = save("hook_token_embed", hidden)
    hidden = hidden + model.degree_emb(model.degree_idx).unsqueeze(0)
    hidden = save("hook_embed", hidden)
    hidden = torch.cat([model.cls.expand(batch_size, -1, -1), hidden], dim=1)
    hidden = save("hook_resid_embed", hidden)

    for layer, block in enumerate(model.blocks):
        prefix = f"blocks.{layer}"
        hidden = save(f"{prefix}.hook_resid_pre", hidden)
        attn_in = save(f"{prefix}.hook_attn_norm", block.attn_norm(hidden))
        q_raw, k_raw, v_raw = _split_qkv(block, attn_in)
        q = save(f"{prefix}.hook_q", _shape_heads(q_raw, block.attn.num_heads))
        k = save(f"{prefix}.hook_k", _shape_heads(k_raw, block.attn.num_heads))
        v = save(f"{prefix}.hook_v", _shape_heads(v_raw, block.attn.num_heads))
        scores = torch.matmul(q, k.transpose(-1, -2)) / (q.shape[-1] ** 0.5)
        pattern = save(f"{prefix}.hook_pattern", scores.softmax(dim=-1))
        z = save(f"{prefix}.hook_z", torch.matmul(pattern, v))
        head_out = save(f"{prefix}.hook_attn_head_out", _project_heads(block, z))
        attn_out = head_out.sum(dim=1)
        if block.attn.out_proj.bias is not None:
            attn_out = attn_out + block.attn.out_proj.bias.view(1, 1, -1)
        attn_out = save(f"{prefix}.hook_attn_out", block.attn_dropout(attn_out))
        hidden = hidden + attn_out
        hidden = save(f"{prefix}.hook_resid_mid", hidden)
        mlp_in = save(f"{prefix}.hook_mlp_norm", block.ffn_norm(hidden))
        mlp_out = save(f"{prefix}.hook_mlp_out", block.ffn(mlp_in))
        hidden = hidden + mlp_out
        hidden = save(f"{prefix}.hook_resid_post", hidden)

    final_hidden = save("hook_final_hidden", model.final_norm(hidden))
    logits = save("hook_logits", model.head(final_hidden[:, 0]))
    return logits, cache


def patch_hook_from_cache(
    source_cache: Mapping[str, torch.Tensor],
    name: str,
    *,
    token_idx: int | None = None,
    batch_idx: torch.Tensor | None = None,
) -> HookFn:
    source = source_cache[name]

    def hook(value: torch.Tensor, _: str) -> torch.Tensor:
        patched = value.clone()
        src = source.to(device=value.device, dtype=value.dtype)
        rows = slice(None) if batch_idx is None else batch_idx.to(value.device)
        if token_idx is None:
            patched[rows] = src[rows]
        else:
            patched[rows, token_idx] = src[rows, token_idx]
        return patched

    return hook


def patch_positions_hook_from_cache(
    source_cache: Mapping[str, torch.Tensor],
    name: str,
    token_indices: list[int],
    *,
    batch_idx: torch.Tensor | None = None,
) -> HookFn:
    source = source_cache[name]

    def hook(value: torch.Tensor, _: str) -> torch.Tensor:
        patched = value.clone()
        src = source.to(device=value.device, dtype=value.dtype)
        rows = slice(None) if batch_idx is None else batch_idx.to(value.device)
        for token_idx in token_indices:
            patched[rows, token_idx] = src[rows, token_idx]
        return patched

    return hook


def patch_head_positions_hook_from_cache(
    source_cache: Mapping[str, torch.Tensor],
    name: str,
    *,
    head_idx: int,
    token_indices: list[int] | None = None,
    batch_idx: torch.Tensor | None = None,
) -> HookFn:
    source = source_cache[name]

    def hook(value: torch.Tensor, _: str) -> torch.Tensor:
        patched = value.clone()
        src = source.to(device=value.device, dtype=value.dtype)
        rows = slice(None) if batch_idx is None else batch_idx.to(value.device)
        if token_indices is None:
            patched[rows, head_idx] = src[rows, head_idx]
        else:
            for token_idx in token_indices:
                patched[rows, head_idx, token_idx] = src[rows, head_idx, token_idx]
        return patched

    return hook


def zero_hook(value: torch.Tensor, _: str) -> torch.Tensor:
    return torch.zeros_like(value)


def zero_head_positions_hook(
    *,
    head_idx: int,
    token_indices: list[int] | None = None,
) -> HookFn:
    def hook(value: torch.Tensor, _: str) -> torch.Tensor:
        patched = value.clone()
        if token_indices is None:
            patched[:, head_idx] = 0
        else:
            for token_idx in token_indices:
                patched[:, head_idx, token_idx] = 0
        return patched

    return hook


def keep_heads_hook(keep: list[int]) -> HookFn:
    def hook(value: torch.Tensor, _: str) -> torch.Tensor:
        out = torch.zeros_like(value)
        out[:, keep] = value[:, keep]
        return out

    return hook


def support_features(tokens: torch.Tensor) -> dict[str, torch.Tensor]:
    support = tokens.ne(0)
    any_support = support.any(dim=1)
    first = support.to(torch.long).argmax(dim=1)
    last = tokens.shape[1] - 1 - support.flip(dims=[1]).to(torch.long).argmax(dim=1)
    first = torch.where(any_support, first, torch.zeros_like(first))
    last = torch.where(any_support, last, torch.zeros_like(last))
    rows = torch.arange(tokens.shape[0], device=tokens.device)
    return {
        "first": first,
        "last": last,
        "width": last - first + 1,
        "leading_token": tokens[rows, first],
        "trailing_token": tokens[rows, last],
        "support": support,
    }


def gather_relative_window(tokens: torch.Tensor, centers: torch.Tensor, radius: int) -> torch.Tensor:
    offsets = torch.arange(-radius, radius + 1, device=tokens.device)
    idx = centers.unsqueeze(1) + offsets.unsqueeze(0)
    valid = (idx >= 0) & (idx < tokens.shape[1])
    idx = idx.clamp(0, tokens.shape[1] - 1)
    gathered = tokens.gather(1, idx)
    return torch.where(valid, gathered, torch.zeros_like(gathered))


def zero_except_windows(tokens: torch.Tensor, radius: int, *, leading: bool = True, trailing: bool = True) -> torch.Tensor:
    feats = support_features(tokens)
    keep = torch.zeros_like(tokens, dtype=torch.bool)
    offsets = torch.arange(-radius, radius + 1, device=tokens.device)
    centers = []
    if leading:
        centers.append(feats["first"])
    if trailing:
        centers.append(feats["last"])
    for center in centers:
        idx = center.unsqueeze(1) + offsets.unsqueeze(0)
        valid = (idx >= 0) & (idx < tokens.shape[1])
        idx = idx.clamp(0, tokens.shape[1] - 1)
        keep.scatter_(1, idx, valid)
    return torch.where(keep, tokens, torch.zeros_like(tokens))


def zero_windows(tokens: torch.Tensor, radius: int, *, leading: bool = True, trailing: bool = True) -> torch.Tensor:
    feats = support_features(tokens)
    drop = torch.zeros_like(tokens, dtype=torch.bool)
    offsets = torch.arange(-radius, radius + 1, device=tokens.device)
    centers = []
    if leading:
        centers.append(feats["first"])
    if trailing:
        centers.append(feats["last"])
    for center in centers:
        idx = center.unsqueeze(1) + offsets.unsqueeze(0)
        valid = (idx >= 0) & (idx < tokens.shape[1])
        idx = idx.clamp(0, tokens.shape[1] - 1)
        drop.scatter_(1, idx, valid)
    return torch.where(drop, torch.zeros_like(tokens), tokens)
