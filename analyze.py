"""Mechanistic analysis tools for the trained HybridLM.

Three modules:
1. A-matrix eigendecomposition  — memory timescales per Mamba layer
2. Linear probing on h_t        — which state dims encode 'saw token A at pos p'?
3. Ablation                     — zero targeted dims, measure accuracy drop
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

from model import HybridLM, HybridConfig, SimpleMambaBlock


# ---------------------------------------------------------------------------
# 1. A-matrix eigendecomposition
# ---------------------------------------------------------------------------

@dataclass
class EigenResult:
    layer: int                       # index in HybridLM.layers (Mamba layers only)
    eigenvalue_magnitudes: list[float]  # |λ| sorted descending; diagonal SSM → real
    decay_timescales: list[float]    # τ = -1/log|λ| in steps
    A_log_mean: list[float]          # mean A_log over d_inner per state dim
    A_log_std: list[float]


@torch.no_grad()
def empirical_mean_delta(model: HybridLM, dataset, n_examples: int = 64,
                         device: str = "cpu") -> dict[int, float]:
    """Mean Δ per Mamba layer measured on real data.

    Δ is input-dependent (selective) — using an arbitrary fixed Δ (e.g. 0.1)
    can distort the implied timescales τ = -1/log|exp(Δ·A)| by an order of
    magnitude. This measures what the trained model actually uses.
    """
    model = model.to(device).eval()
    seqs = dataset.data[:n_examples].to(device)
    _, caches = model.run_with_cache(seqs)
    return {
        li: float(caches[li]["delta"].mean())
        for li, layer in enumerate(model.layers)
        if isinstance(layer, SimpleMambaBlock)
    }


def analyze_A_matrices(
    model: HybridLM,
    delta_ref: float = 0.1,
    delta_per_layer: dict[int, float] | None = None,
) -> list[EigenResult]:
    """
    For every SimpleMambaBlock in the hybrid model, compute the eigendecomposition
    of the discretised A matrix. If delta_per_layer (from empirical_mean_delta)
    is given, it overrides delta_ref — strongly recommended.

    Returns one EigenResult per Mamba layer (skips Attention layers).
    """
    results = []
    for layer_idx, layer in enumerate(model.layers):
        if not isinstance(layer, SimpleMambaBlock):
            continue

        A = -torch.exp(layer.A_log.detach().float().cpu())   # (D_inner, N), negative
        A_mean = A.mean(dim=0)                          # (N,) — average over channels
        A_std  = A.std(dim=0)                           # (N,)

        dref = (delta_per_layer.get(layer_idx, delta_ref)
                if delta_per_layer else delta_ref)

        # Diagonal SSM: eigenvalues of Ā = exp(Δ * A) are the diagonal entries
        A_bar_diag = torch.exp(dref * A_mean).numpy()   # (N,)
        magnitudes = np.abs(A_bar_diag)
        order = np.argsort(-magnitudes)
        magnitudes = magnitudes[order].tolist()

        timescales = []
        for mag in magnitudes:
            if mag < 1e-8:
                timescales.append(0.0)
            elif mag >= 1.0:
                timescales.append(float("inf"))
            else:
                timescales.append(float(-1.0 / np.log(mag)))

        results.append(EigenResult(
            layer=layer_idx,
            eigenvalue_magnitudes=magnitudes,
            decay_timescales=timescales,
            A_log_mean=A_mean.tolist(),
            A_log_std=A_std.tolist(),
        ))

    return results


# ---------------------------------------------------------------------------
# 2. Linear probing on h_t
# ---------------------------------------------------------------------------

@dataclass
class ProbeResult:
    """Multiclass decoding of the *content* carried by the SSM state.

    Old design (per-dim AUROC on 'token 0 appeared at position p') was
    statistically broken: with vocab 64, only ~1-2% of examples are positive,
    so AUROC estimates from ~5 positives are noise. New design: decode the
    IDENTITY of B (the token to be recalled) from the state — a balanced
    64-way problem where chance is 1/64 and every example is usable.
    """
    layer: int              # index in HybridLM.layers (Mamba layers only)
    position: int           # sequence position where h is read
    accuracy: float         # cross-validated multiclass decode accuracy
    chance: float           # 1 / vocab_size
    per_dim_score: list[float]  # ANOVA F-score per state dim (for ranking/ablation)
    n_examples: int


def probe_state_dimensions(
    model: HybridLM,
    dataset,
    probe_positions: list[int],
    n_examples: int = 500,
    device: str = "cpu",
    n_splits: int = 5,
    batch_size: int = 64,
    decode: str = "target",   # 'target' → decode B; 'trigger' → decode A
) -> list[ProbeResult]:
    """
    For each Mamba layer and each position, fit a cross-validated multinomial
    logistic regression on the channel-averaged state h̄_t ∈ R^N to decode the
    identity of B (dataset.targets). Reading positions *before* the [A,B]
    bigram acts as a built-in negative control (decoding must be at chance).

    NOTE: channel-averaging h over D_inner is lossy; decoding accuracy is a
    lower bound on the information in the full state.

    dataset must be an InductionDataset (uses .data / .targets).
    """
    from sklearn.feature_selection import f_classif

    model = model.to(device)
    model.eval()
    n = min(n_examples, len(dataset))
    vocab_size = dataset.vocab_size

    mamba_layer_indices = [
        i for i, layer in enumerate(model.layers)
        if isinstance(layer, SimpleMambaBlock)
    ]

    labels_all = (dataset.targets[:n] if decode == "target"
                  else dataset.data[:n, -1]).numpy()

    # Collect features in batches: h̄ at every (layer, pos) requested
    feats: dict[tuple[int, int], list[np.ndarray]] = {
        (li, p): [] for li in mamba_layer_indices for p in probe_positions
    }
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            seqs = dataset.data[start:end].to(device)
            _, caches = model.run_with_cache(seqs)
            for li in mamba_layer_indices:
                h = caches[li]["h"]                     # (B, L, Di, N)
                for p in probe_positions:
                    feats[(li, p)].append(h[:, p].mean(dim=1).cpu().numpy())  # (B, N)

    results = []
    for (li, p), chunks in feats.items():
        X = np.concatenate(chunks, axis=0)              # (n, N)
        y = labels_all
        X_scaled = StandardScaler().fit_transform(X)

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        try:
            preds = cross_val_predict(
                LogisticRegression(max_iter=500, C=1.0),
                X_scaled, y, cv=cv,
            )
            acc = float((preds == y).mean())
        except ValueError:
            # some class has < n_splits members; fall back to 2 folds
            preds = cross_val_predict(
                LogisticRegression(max_iter=500, C=1.0),
                X_scaled, y, cv=2,
            )
            acc = float((preds == y).mean())

        f_scores, _ = f_classif(X_scaled, y)
        f_scores = np.nan_to_num(f_scores, nan=0.0)

        results.append(ProbeResult(
            layer=li,
            position=p,
            accuracy=acc,
            chance=1.0 / vocab_size,
            per_dim_score=f_scores.tolist(),
            n_examples=int(n),
        ))
    return results


# ---------------------------------------------------------------------------
# 3. Ablation
# ---------------------------------------------------------------------------

@dataclass
class AblationResult:
    layer: int
    ablated_dims: list[int]
    baseline_accuracy: float
    val_accuracy: float
    accuracy_drop: float


def ablate_state_dims(
    model: HybridLM,
    dataset,
    layer: int,
    dims: list[int],
    n_examples: int = 200,
    device: str = "cpu",
) -> AblationResult:
    """
    Measure induction accuracy when SSM state dimensions `dims` in Mamba
    layer `layer` are silenced.

    Silencing is implemented by:
      1. Setting A_log[:, dims] = -50 → Ā ≈ 0 (no persistent memory in those dims)
      2. Zeroing the C_t read weights for those dims in x_proj
         (so those dims never contribute to the SSM output y_t = C_t · h_t)

    A deep copy of the model is made so the original is not modified.
    dataset must return (seq, target, foil) tuples.
    """
    model = model.to(device)
    model.eval()

    target_layer = model.layers[layer]
    assert isinstance(target_layer, SimpleMambaBlock), (
        f"Layer {layer} is not a SimpleMambaBlock; cannot ablate SSM dims"
    )

    def _eval_acc(m: HybridLM) -> float:
        m.eval()
        correct = total = 0
        with torch.no_grad():
            for idx in range(min(n_examples, len(dataset))):
                seq, target, _ = dataset[idx]
                logits = m(seq.unsqueeze(0).to(device))
                pred = logits[0, -1, :].argmax(-1).item()
                correct += int(pred == int(target))
                total += 1
        return correct / total if total > 0 else 0.0

    baseline_acc = _eval_acc(model)

    # Deep-copy and silence dims
    ablated = copy.deepcopy(model)
    tgt = ablated.layers[layer]          # SimpleMambaBlock

    with torch.no_grad():
        # Kill persistent memory in ablated dims: Ā[:, dims] → exp(-50) ≈ 0
        tgt.A_log[:, dims] = -50.0

        # Zero the C_t readout weights for ablated dims
        # x_proj output layout: [dt_rank | d_state (B_t) | d_state (C_t)]
        c_start = tgt.dt_rank + tgt.d_state   # start of C_t rows in x_proj.weight
        tgt.x_proj.weight[c_start + torch.tensor(dims), :] = 0.0

    ablated_acc = _eval_acc(ablated)

    return AblationResult(
        layer=layer,
        ablated_dims=dims,
        baseline_accuracy=baseline_acc,
        val_accuracy=ablated_acc,
        accuracy_drop=baseline_acc - ablated_acc,
    )


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_analysis(
    eigen: list[EigenResult],
    probes: list[ProbeResult],
    ablations: list[AblationResult],
    path: str | Path,
) -> None:
    import dataclasses
    out = {
        "eigen":    [dataclasses.asdict(r) for r in eigen],
        "probe":    [dataclasses.asdict(r) for r in probes],
        "ablation": [dataclasses.asdict(r) for r in ablations],
    }
    Path(path).write_text(json.dumps(out, indent=2))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from train import default_hybrid_config
    from data import InductionDataset

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--vocab_size", type=int, default=64)
    parser.add_argument("--seq_len",   type=int, default=64)
    parser.add_argument("--n_examples", type=int, default=500)
    parser.add_argument("--out", default="results/analysis.json")
    args = parser.parse_args()

    cfg = default_hybrid_config(vocab_size=args.vocab_size)
    model = HybridLM(cfg)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.eval()

    val_ds = InductionDataset(args.n_examples,
                              seq_len=args.seq_len,
                              vocab_size=args.vocab_size,
                              seed=99)

    print("Eigendecomposition (empirical Δ)...")
    deltas = empirical_mean_delta(model, val_ds)
    eigen = analyze_A_matrices(model, delta_per_layer=deltas)

    # Fixed-distance dataset → known critical positions for probing
    d_probe = args.seq_len - 16
    probe_ds = InductionDataset(args.n_examples, seq_len=args.seq_len,
                                vocab_size=args.vocab_size, seed=99,
                                min_dist=d_probe, max_dist=d_probe)
    first_B = int(probe_ds.first_positions[0]) + 1
    positions = [first_B - 2, first_B, args.seq_len // 2, args.seq_len - 1]
    print(f"Decoding B identity from h at positions {positions} "
          f"(pos {first_B - 2} is a pre-bigram negative control)...")
    probes = probe_state_dimensions(model, probe_ds,
                                    probe_positions=positions,
                                    n_examples=args.n_examples)

    mamba_indices = [i for i, l in enumerate(model.layers)
                     if isinstance(l, SimpleMambaBlock)]
    ablations = []
    for li in mamba_indices:
        pr = [p for p in probes if p.layer == li and p.position == args.seq_len - 1]
        if not pr:
            continue
        scores = np.array(pr[0].per_dim_score)
        top2 = np.argsort(-scores)[:2].tolist()
        print(f"Ablating dims {top2} in Mamba layer L{li}...")
        abl = ablate_state_dims(model, probe_ds, layer=li, dims=top2,
                                n_examples=200)
        ablations.append(abl)
        print(f"  drop: {abl.accuracy_drop:.3f}")

    save_analysis(eigen, probes, ablations, args.out)
    print(f"Saved to {args.out}")
