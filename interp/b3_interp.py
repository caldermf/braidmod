"""TransformerLens-style tools for the B_3 absolute-degree transformer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import torch
import torch.nn.functional as F

from interp.b3_data import absolute_depth_for_length
from interp.b3_transformer import B3AbsoluteTransformer, B3TransformerConfig


HookFn = Callable[[torch.Tensor, str], torch.Tensor]


@dataclass
class LoadedTransformer:
    model: B3AbsoluteTransformer
    checkpoint: dict


def load_transformer_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> LoadedTransformer:
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    cfg = checkpoint["model_config"]
    config = B3TransformerConfig(
        length=int(cfg["length"]),
        d_model=int(cfg["d_model"]),
        num_layers=int(cfg["num_layers"]),
        num_heads=int(cfg["num_heads"]),
        ffn_mult=int(cfg["ffn_mult"]),
        dropout=float(cfg.get("dropout", 0.0)),
    )
    model = B3AbsoluteTransformer(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return LoadedTransformer(model=model, checkpoint=checkpoint)


def binary_logit_score(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    direction = labels.to(device=logits.device, dtype=logits.dtype).mul(2).sub(1)
    return logits * direction


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


def _merge_heads(x: torch.Tensor) -> torch.Tensor:
    batch, num_heads, seq_len, head_dim = x.shape
    return x.transpose(1, 2).contiguous().view(batch, seq_len, num_heads * head_dim)


def _project_heads(block, z: torch.Tensor) -> torch.Tensor:
    """Apply W_O separately to each attention head.

    Returns per-head residual-stream contributions with shape [B, H, S, D].
    The module-level out_proj bias is intentionally not included; it is added
    once after summing heads.
    """
    d_model = block.attn.embed_dim
    num_heads = block.attn.num_heads
    head_dim = d_model // num_heads
    weight = block.attn.out_proj.weight.view(d_model, num_heads, head_dim)
    return torch.einsum("bhsd,mhd->bhsm", z, weight)


def run_with_cache(
    model: B3AbsoluteTransformer,
    tokens: torch.Tensor,
    *,
    hooks: Mapping[str, HookFn | torch.Tensor] | None = None,
    names_filter: set[str] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Run the transformer with named activation cache and optional forward hooks.

    Hook/cache names intentionally mirror TransformerLens conventions where
    practical: `blocks.{layer}.hook_resid_pre`, `hook_q`, `hook_pattern`,
    `hook_attn_out`, `hook_mlp_out`, and so on.
    """
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
    logits = save("hook_logits", model.head(final_hidden[:, 0]).squeeze(-1))
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
    """Patch one attention head contribution from a source cache.

    Expected activation shape is [batch, head, token, d_model] for
    `hook_attn_head_out` or [batch, head, token, d_head] for `hook_z`.
    """
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


def patch_head_set_positions_hook_from_cache(
    source_cache: Mapping[str, torch.Tensor],
    name: str,
    *,
    head_indices: list[int],
    token_indices: list[int] | None = None,
    batch_idx: torch.Tensor | None = None,
) -> HookFn:
    """Patch a set of attention heads from a source cache."""
    source = source_cache[name]

    def hook(value: torch.Tensor, _: str) -> torch.Tensor:
        patched = value.clone()
        src = source.to(device=value.device, dtype=value.dtype)
        rows = slice(None) if batch_idx is None else batch_idx.to(value.device)
        for head_idx in head_indices:
            if token_indices is None:
                patched[rows, head_idx] = src[rows, head_idx]
            else:
                for token_idx in token_indices:
                    patched[rows, head_idx, token_idx] = src[rows, head_idx, token_idx]
        return patched

    return hook


def zero_head_positions_hook(
    *,
    head_idx: int,
    token_indices: list[int] | None = None,
) -> HookFn:
    """Zero one attention head contribution at all or selected destinations."""

    def hook(value: torch.Tensor, _: str) -> torch.Tensor:
        patched = value.clone()
        if token_indices is None:
            patched[:, head_idx] = 0
        else:
            for token_idx in token_indices:
                patched[:, head_idx, token_idx] = 0
        return patched

    return hook


def zero_head_set_positions_hook(
    *,
    head_indices: list[int],
    token_indices: list[int] | None = None,
) -> HookFn:
    """Zero a set of attention head contributions."""

    def hook(value: torch.Tensor, _: str) -> torch.Tensor:
        patched = value.clone()
        for head_idx in head_indices:
            if token_indices is None:
                patched[:, head_idx] = 0
            else:
                for token_idx in token_indices:
                    patched[:, head_idx, token_idx] = 0
        return patched

    return hook


def activation_patch_position_set(
    model: B3AbsoluteTransformer,
    clean_tokens: torch.Tensor,
    corrupt_tokens: torch.Tensor,
    clean_labels: torch.Tensor,
    *,
    name: str,
    token_indices: list[int],
) -> dict[str, float]:
    with torch.no_grad():
        clean_logits, clean_cache = run_with_cache(model, clean_tokens)
        corrupt_logits, _ = run_with_cache(model, corrupt_tokens)
        patched_logits, _ = run_with_cache(
            model,
            corrupt_tokens,
            hooks={name: patch_positions_hook_from_cache(clean_cache, name, token_indices)},
            names_filter=set(),
        )
        clean_score = binary_logit_score(clean_logits, clean_labels)
        corrupt_score = binary_logit_score(corrupt_logits, clean_labels)
        patched_score = binary_logit_score(patched_logits, clean_labels)
        denom = (clean_score - corrupt_score).mean().clamp_min(1e-6)
        recovery = (patched_score.mean() - corrupt_score.mean()) / denom
    return {
        "clean_score": float(clean_score.mean().item()),
        "corrupt_score": float(corrupt_score.mean().item()),
        "patched_score": float(patched_score.mean().item()),
        "recovery": float(recovery.item()),
    }


def activation_patch_token_sweep(
    model: B3AbsoluteTransformer,
    clean_tokens: torch.Tensor,
    corrupt_tokens: torch.Tensor,
    clean_labels: torch.Tensor,
    *,
    site: str,
    layers: list[int] | None = None,
) -> dict[str, torch.Tensor]:
    """Patch one token position at a time from clean into corrupt examples."""
    if clean_tokens.shape != corrupt_tokens.shape:
        raise ValueError("clean_tokens and corrupt_tokens must have the same shape")
    if layers is None:
        layers = list(range(len(model.blocks)))

    with torch.no_grad():
        clean_logits, clean_cache = run_with_cache(model, clean_tokens)
        corrupt_logits, _ = run_with_cache(model, corrupt_tokens)
        clean_score = binary_logit_score(clean_logits, clean_labels)
        corrupt_score = binary_logit_score(corrupt_logits, clean_labels)
        denom = (clean_score - corrupt_score).mean().clamp_min(1e-6)
        seq_len = clean_tokens.shape[1] + 1
        recovery = torch.empty(len(layers), seq_len, device=clean_tokens.device)
        patched_score = torch.empty(len(layers), seq_len, device=clean_tokens.device)

        for layer_i, layer in enumerate(layers):
            name = f"blocks.{layer}.{site}"
            if name not in clean_cache:
                raise KeyError(f"{name} not found in clean cache")
            for token_idx in range(seq_len):
                logits, _ = run_with_cache(
                    model,
                    corrupt_tokens,
                    hooks={name: patch_hook_from_cache(clean_cache, name, token_idx=token_idx)},
                    names_filter=set(),
                )
                score = binary_logit_score(logits, clean_labels).mean()
                patched_score[layer_i, token_idx] = score
                recovery[layer_i, token_idx] = (score - corrupt_score.mean()) / denom

    return {
        "layers": torch.tensor(layers),
        "patched_score": patched_score.detach().cpu(),
        "recovery": recovery.detach().cpu(),
        "clean_score": clean_score.detach().cpu(),
        "corrupt_score": corrupt_score.detach().cpu(),
    }


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


def encode_base16_window(window: torch.Tensor) -> torch.Tensor:
    powers = (16 ** torch.arange(window.shape[1], device=window.device, dtype=torch.long)).view(1, -1)
    return (window.to(torch.long) * powers).sum(dim=1)


def boundary_feature_keys(tokens: torch.Tensor, radius: int) -> dict[str, torch.Tensor]:
    feats = support_features(tokens)
    lead_window = gather_relative_window(tokens, feats["first"], radius)
    trail_window = gather_relative_window(tokens, feats["last"], radius)
    lead_code = encode_base16_window(lead_window)
    trail_code = encode_base16_window(trail_window)
    depth = tokens.shape[1]
    window_base = 16 ** (2 * radius + 1)
    keys = {
        "first": feats["first"],
        "last": feats["last"],
        "first_last": feats["first"] * depth + feats["last"],
        "leading_token": feats["leading_token"],
        "trailing_token": feats["trailing_token"],
        "first_leading": feats["first"] * 16 + feats["leading_token"],
        "last_trailing": feats["last"] * 16 + feats["trailing_token"],
        "boundary_tokens": (((feats["first"] * depth + feats["last"]) * 16 + feats["leading_token"]) * 16)
        + feats["trailing_token"],
        f"leading_window_r{radius}": feats["first"] * window_base + lead_code,
        f"trailing_window_r{radius}": feats["last"] * window_base + trail_code,
        f"both_windows_r{radius}": (((feats["first"] * depth + feats["last"]) * window_base + lead_code) * window_base)
        + trail_code,
    }
    return keys


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


def flip_unit_token_column(tokens: torch.Tensor) -> torch.Tensor:
    flipped = tokens.clone()
    flipped = torch.where(tokens == 1, torch.full_like(tokens, 2), flipped)
    flipped = torch.where(tokens == 2, torch.full_like(tokens, 1), flipped)
    flipped = torch.where(tokens == 4, torch.full_like(tokens, 8), flipped)
    flipped = torch.where(tokens == 8, torch.full_like(tokens, 4), flipped)
    return flipped


def flip_boundary_columns(tokens: torch.Tensor, *, leading: bool = True, trailing: bool = True) -> torch.Tensor:
    feats = support_features(tokens)
    out = tokens.clone()
    rows = torch.arange(tokens.shape[0], device=tokens.device)
    if leading:
        idx = feats["first"]
        out[rows, idx] = flip_unit_token_column(out[rows, idx])
    if trailing:
        idx = feats["last"]
        out[rows, idx] = flip_unit_token_column(out[rows, idx])
    return out


def absolute_depth_from_checkpoint(checkpoint: Mapping) -> int:
    cfg = checkpoint["model_config"]
    return int(cfg.get("absolute_depth", absolute_depth_for_length(int(cfg["length"]))))
