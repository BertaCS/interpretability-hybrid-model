"""Cross-architecture activation patching for the Hybrid Mamba-Transformer model.

Implements causal tracing over a heterogeneous layer stack:
  - Mamba layers: patch the SSM hidden state h_t at specific positions
  - Attention layers: patch K or V tensors at specific positions

Metric (corrected from spec):
    logit_diff = logit(B) - logit(B')
    recovery(l, t, c) = (patched_LD - corrupted_LD) / (clean_LD - corrupted_LD)

Corruption protocol:
    Replace the B token at first_B_pos with B' (a specific foil token, B' ≠ B).
    Only one token is changed so that recovery can be attributed to specific positions.

Patchable components:
    Mamba layers: 'h'         — SSM hidden state post-step
    Attn layers : 'K'         — key projection
                  'V'         — value projection
                  'attn_out'  — full attention output (for comparison baseline)

Output:
    PatchGrid — (n_layers, seq_len) recovery tensor per component.
    Components from different layer types are stored in the same grid but are
    only meaningful at positions corresponding to the appropriate layer type.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import torch
import torch.nn.functional as F

from model import HybridLM, SimpleMambaBlock, StandardAttentionBlock


# ---------------------------------------------------------------------------
# Logit difference helpers
# ---------------------------------------------------------------------------

ComponentName = Literal["h", "K", "V", "attn_out"]

MAMBA_COMPONENTS: tuple[str, ...] = ("h",)
ATTN_COMPONENTS: tuple[str, ...] = ("K", "V", "attn_out")


def logit_diff(
    logits: torch.Tensor,                    # (B, V)
    correct: int | torch.Tensor,             # scalar or (B,) per-example B
    foil: int | torch.Tensor,                # scalar or (B,) per-example B'
) -> float:
    """Mean logit(correct) - logit(foil) over batch (per-example tokens supported)."""
    if isinstance(correct, int):
        correct = torch.full((logits.shape[0],), correct, device=logits.device)
    if isinstance(foil, int):
        foil = torch.full((logits.shape[0],), foil, device=logits.device)
    c = logits.gather(1, correct.view(-1, 1)).squeeze(1)   # (B,)
    f = logits.gather(1, foil.view(-1, 1)).squeeze(1)      # (B,)
    return (c - f).mean().item()


def _last_logits(logits: torch.Tensor) -> torch.Tensor:
    """(B, L, V) → (B, V) at the final position."""
    return logits[:, -1, :]


# ---------------------------------------------------------------------------
# Single-example patched forward pass
# ---------------------------------------------------------------------------

def _patched_forward(
    model: HybridLM,
    corrupted_ids: torch.Tensor,    # (B, L)
    clean_caches: list[dict],
    patch_layer: int,
    patch_pos: int,
    patch_component: ComponentName,
) -> torch.Tensor:
    """
    Run HybridLM on `corrupted_ids`, replacing one activation component at
    (patch_layer, patch_pos) with the corresponding value from `clean_caches`.

    Returns logits (B, L, V).
    """
    B, L = corrupted_ids.shape
    x = model._embed(corrupted_ids)             # (B, L, D)

    for layer_idx, layer in enumerate(model.layers):

        if layer_idx != patch_layer:
            # Standard forward, no patching
            x, _ = layer(x)
            continue

        # ── Patching this layer ──────────────────────────────────────────
        if isinstance(layer, SimpleMambaBlock):
            # Only 'h' is patchable in Mamba layers
            if patch_component == "h":
                mask = torch.zeros(L, dtype=torch.bool, device=x.device)
                mask[patch_pos] = True
                h_override = clean_caches[layer_idx]["h"]   # (B, L, Di, N)
                x, _ = layer(x, h_override=h_override, h_override_mask=mask)
            else:
                x, _ = layer(x)

        elif isinstance(layer, StandardAttentionBlock):
            clean_K = clean_caches[layer_idx]["K"]          # (B, L, H, d_h)
            clean_V = clean_caches[layer_idx]["V"]          # (B, L, H, d_h)
            mask = torch.zeros(L, dtype=torch.bool, device=x.device)
            mask[patch_pos] = True

            if patch_component == "K":
                x, _ = layer(x, K_override=clean_K, KV_mask=mask)
            elif patch_component == "V":
                x, _ = layer(x, V_override=clean_V, KV_mask=mask)
            elif patch_component == "attn_out":
                # Run the full block normally on corrupted input
                x, cache = layer(x)
                # block output = residual + attn_out + mlp_out
                # Swap only the attn_out contribution at patch_pos:
                #   x_patched[patch_pos] = x[patch_pos]
                #                         - corrupted_attn_out[patch_pos]
                #                         + clean_attn_out[patch_pos]
                clean_ao = clean_caches[layer_idx]["attn_out"]   # (B, L, D)
                corrupted_ao = cache["attn_out"]                  # (B, L, D)
                x = x.clone()
                x[:, patch_pos, :] = (
                    x[:, patch_pos, :]
                    - corrupted_ao[:, patch_pos, :]
                    + clean_ao[:, patch_pos, :]
                )
            else:
                x, _ = layer(x)
        else:
            x, _ = layer(x)

    x = model.norm_f(x)
    return model.lm_head(x)                     # (B, L, V)


# ---------------------------------------------------------------------------
# Full patching scan
# ---------------------------------------------------------------------------

@dataclass
class PatchGrid:
    """Recovery grids for one model run."""
    n_layers: int
    seq_len: int
    layer_types: list[str]           # 'mamba' or 'attn' per layer
    # component → (n_layers, seq_len) recovery tensor
    grids: dict[str, torch.Tensor] = field(default_factory=dict)
    clean_ld: float = 0.0
    corrupted_ld: float = 0.0

    def to_dict(self) -> dict:
        return {
            "n_layers": self.n_layers,
            "seq_len": self.seq_len,
            "layer_types": self.layer_types,
            "clean_ld": self.clean_ld,
            "corrupted_ld": self.corrupted_ld,
            "grids": {k: v.tolist() for k, v in self.grids.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PatchGrid":
        pg = cls(
            n_layers=d["n_layers"],
            seq_len=d["seq_len"],
            layer_types=d["layer_types"],
            clean_ld=d["clean_ld"],
            corrupted_ld=d["corrupted_ld"],
        )
        pg.grids = {k: torch.tensor(v) for k, v in d["grids"].items()}
        return pg


@torch.no_grad()
def patch_scan(
    model: HybridLM,
    clean_ids: torch.Tensor,          # (B, L)
    corrupted_ids: torch.Tensor,      # (B, L)  — B at first_B_pos replaced by B'
    correct_token: int | torch.Tensor,   # B (scalar or per-example (B,))
    foil_token: int | torch.Tensor,      # B' (scalar or per-example (B,))
    components: Optional[list[str]] = None,
    positions: Optional[list[int]] = None,   # None → all positions
) -> PatchGrid:
    """
    Sweep all (layer, position, component) combinations and compute
    logit-difference recovery.

    Only sweeps components that are valid for each layer type:
      Mamba layers: 'h'
      Attn layers : 'K', 'V', 'attn_out'
    """
    if components is None:
        components = list(MAMBA_COMPONENTS) + list(ATTN_COMPONENTS)

    device = next(model.parameters()).device
    clean_ids = clean_ids.to(device)
    corrupted_ids = corrupted_ids.to(device)
    if isinstance(correct_token, torch.Tensor):
        correct_token = correct_token.to(device)
    if isinstance(foil_token, torch.Tensor):
        foil_token = foil_token.to(device)

    # Baseline runs
    clean_logits, clean_caches = model.run_with_cache(clean_ids)
    corrupted_logits, _ = model.run_with_cache(corrupted_ids)

    clean_ld = logit_diff(_last_logits(clean_logits), correct_token, foil_token)
    corr_ld = logit_diff(_last_logits(corrupted_logits), correct_token, foil_token)
    denom = max(abs(clean_ld - corr_ld), 1e-6)

    n_layers = len(model.layers)
    L = clean_ids.shape[1]
    layer_types = [model.config.layer_type(i) for i in range(n_layers)]

    # NaN marks (layer, pos, component) cells that were never evaluated
    # (invalid component for that layer type, or position not scanned).
    grids: dict[str, torch.Tensor] = {
        c: torch.full((n_layers, L), float("nan")) for c in components
    }
    scan_positions = list(range(L)) if positions is None else positions

    for layer_idx in range(n_layers):
        ltype = layer_types[layer_idx]
        valid_comps = [c for c in components if (
            (ltype == "mamba" and c in MAMBA_COMPONENTS) or
            (ltype == "attn" and c in ATTN_COMPONENTS)
        )]

        for comp in valid_comps:
            for pos in scan_positions:
                patched_logits = _patched_forward(
                    model, corrupted_ids, clean_caches,
                    patch_layer=layer_idx,
                    patch_pos=pos,
                    patch_component=comp,
                )
                patched_ld = logit_diff(_last_logits(patched_logits), correct_token, foil_token)
                recovery = (patched_ld - corr_ld) / denom
                grids[comp][layer_idx, pos] = recovery

    pg = PatchGrid(
        n_layers=n_layers,
        seq_len=L,
        layer_types=layer_types,
        clean_ld=clean_ld,
        corrupted_ld=corr_ld,
    )
    pg.grids = grids
    return pg


# ---------------------------------------------------------------------------
# Batch experiment: average over many induction examples
# ---------------------------------------------------------------------------

def make_corrupted(
    seq: torch.Tensor,      # (L,) or (B, L)
    first_B_pos: int,
    foil_token: int,
) -> torch.Tensor:
    """Replace the B token at first_B_pos with foil_token."""
    corrupted = seq.clone()
    if corrupted.dim() == 1:
        corrupted[first_B_pos] = foil_token
    else:
        corrupted[:, first_B_pos] = foil_token
    return corrupted


def run_patch_experiment(
    model: HybridLM,
    dataset,                      # InductionDataset with min_dist == max_dist
    n_examples: int = 100,
    batch_size: int = 32,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    verbose: bool = True,
    positions: Optional[list[int]] = None,
) -> PatchGrid:
    """
    Batched patch scan over a FIXED-DISTANCE dataset.

    Requiring min_dist == max_dist means first_A_pos, first_B_pos and
    second_A_pos are identical across examples, so patching a given position
    means the same thing for every example and the scan can run in batch
    (one forward per (layer, pos, component) instead of one per example).

    The foil token B' is per-example (dataset.foils), guaranteeing
    B' ≠ B ≠ A. Corruption replaces only B at first_B_pos with B'.
    The per-chunk recovery grids are averaged weighted by chunk size.
    """
    assert getattr(dataset, "min_dist", None) == getattr(dataset, "max_dist", None), (
        "run_patch_experiment requires a fixed-distance dataset "
        "(InductionDataset(min_dist=d, max_dist=d)) so that critical positions "
        "align across examples and the scan can be batched."
    )
    model = model.to(device)
    model.eval()

    n = min(n_examples, len(dataset))
    seqs = dataset.data[:n]                        # (n, L)
    targets = dataset.targets[:n]                  # (n,)
    foils = dataset.foils[:n]                      # (n,)
    first_A_pos = int(dataset.first_positions[0])
    first_B_pos = first_A_pos + 1

    n_layers = len(model.layers)
    seq_len = seqs.shape[1]
    layer_types = [model.config.layer_type(i) for i in range(n_layers)]
    components = list(MAMBA_COMPONENTS) + list(ATTN_COMPONENTS)

    accum: dict[str, torch.Tensor] = {}
    sum_clean_ld = sum_corr_ld = 0.0
    total = 0

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        chunk = end - start
        clean_ids = seqs[start:end]
        # per-example foil corruption
        corrupted_ids = clean_ids.clone()
        corrupted_ids[:, first_B_pos] = foils[start:end]

        pg = patch_scan(
            model, clean_ids, corrupted_ids,
            correct_token=targets[start:end],
            foil_token=foils[start:end],
            components=components,
            positions=positions,
        )
        for c in components:
            if c not in accum:
                accum[c] = pg.grids[c] * chunk
            else:
                accum[c] += pg.grids[c] * chunk
        sum_clean_ld += pg.clean_ld * chunk
        sum_corr_ld += pg.corrupted_ld * chunk
        total += chunk
        if verbose:
            print(f"  patched {total}/{n} examples (batched)...")

    result = PatchGrid(
        n_layers=n_layers,
        seq_len=seq_len,
        layer_types=layer_types,
        clean_ld=sum_clean_ld / max(total, 1),
        corrupted_ld=sum_corr_ld / max(total, 1),
    )
    result.grids = {c: g / max(total, 1) for c, g in accum.items()}
    if abs(result.clean_ld - result.corrupted_ld) < 0.5:
        print("  WARNING: |clean_LD - corrupted_LD| "
              f"= {abs(result.clean_ld - result.corrupted_ld):.3f} < 0.5 — the "
              "recovery denominator is tiny, so recovery values are noise "
              "(can wildly exceed 1). Do not interpret this scan.")
    if verbose:
        print(f"  Done. {total} examples | clean_LD={result.clean_ld:.3f} "
              f"corrupted_LD={result.corrupted_ld:.3f} "
              f"| A1={first_A_pos} B={first_B_pos} A2={seq_len-1}")
    return result


def save_patch_results(pg: PatchGrid, path: str | Path) -> None:
    Path(path).write_text(json.dumps(pg.to_dict(), indent=2))


def load_patch_results(path: str | Path) -> PatchGrid:
    return PatchGrid.from_dict(json.loads(Path(path).read_text()))
