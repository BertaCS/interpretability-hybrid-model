"""Publication-ready figures for the Hybrid Mamba-Transformer interpretability paper.

Figure 1 (figure_patching_heatmap.png):
    Cross-architecture activation patching heatmap.
    Sub-panels per component (h, K, V, attn_out).
    Layer types annotated; Mamba rows show h recovery, Attn rows show K/V recovery.
    Vertical markers for first_A_pos, first_B_pos, second_A_pos.

Figure 2 (figure_phase_transition.png):
    Generated directly by train.py. Loss + accuracy + logit-diff curves.

Figure 3 (figure_eigenspectrum.png):
    SSM A-matrix memory timescales per Mamba layer.

Figure 4 (figure_probe_auroc.png):
    Linear probe AUROC on h_t state dimensions.

Figure 5 (figure_ablation.png):
    Accuracy drop when top state dimensions are ablated.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Colours per component
COMP_COLOR = {
    "h":        "#1a9641",
    "K":        "#2166ac",
    "V":        "#756bb1",
    "attn_out": "#d6604d",
}

LAYER_HATCH = {
    "mamba": "",
    "attn":  "//",
}


# ---------------------------------------------------------------------------
# Figure 1: Cross-architecture patching heatmap
# ---------------------------------------------------------------------------

def plot_patching_heatmap(
    grids: dict[str, np.ndarray],       # component → (n_layers, seq_len)
    layer_types: list[str],             # 'mamba' or 'attn' per layer
    first_A_pos: Optional[int] = None,
    first_B_pos: Optional[int] = None,
    second_A_pos: Optional[int] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    One sub-panel per component.  Mamba/Attn rows are annotated with band colours.
    Black vertical lines mark the critical sequence positions.
    """
    components = list(grids.keys())
    n_layers, seq_len = next(iter(grids.values())).shape
    n_cols = len(components)

    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 3.8), squeeze=False)
    axes = axes[0]

    for ax, comp in zip(axes, components):
        g = np.array(grids[comp], dtype=float)
        g_masked = np.ma.masked_invalid(g)   # NaN = not evaluated (wrong layer type)
        cmap = matplotlib.colormaps["RdYlGn"].copy()
        cmap.set_bad("#e0e0e0")

        im = ax.imshow(
            g_masked, aspect="auto", vmin=0, vmax=1,
            cmap=cmap, origin="lower",
            extent=[-0.5, seq_len - 0.5, -0.5, n_layers - 0.5],
        )

        # Shade rows by layer type
        for li, ltype in enumerate(layer_types):
            color = "#d1e5f0" if ltype == "mamba" else "#fee0d2"
            ax.axhspan(li - 0.5, li + 0.5, color=color, alpha=0.20, zorder=0)

        # Mark critical positions
        for pos, ls, label in [
            (first_A_pos,  ":",  "A₁"),
            (first_B_pos,  "--", "B"),
            (second_A_pos, "-",  "A₂"),
        ]:
            if pos is not None:
                ax.axvline(pos, color="black", lw=1.4, ls=ls, alpha=0.85)
                ax.text(pos + 0.3, n_layers - 0.4, label,
                        fontsize=8, color="black", va="top")

        ax.set_title(f"Component: {comp}", pad=6,
                     color=COMP_COLOR.get(comp, "black"), fontweight="bold")
        ax.set_xlabel("Sequence position")
        ax.set_ylabel("Layer" if comp == components[0] else "")
        ax.set_yticks(range(n_layers))
        ax.set_yticklabels(
            [f"L{i} ({t[0].upper()})" for i, t in enumerate(layer_types)]
        )

    # Shared colorbar
    cbar = fig.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04)
    cbar.set_label("Recovery fraction (LD patched − LD corrupted) / (LD clean − LD corrupted)")

    # Legend for layer types
    patches = [
        mpatches.Patch(facecolor="#d1e5f0", alpha=0.5, label="Mamba layer"),
        mpatches.Patch(facecolor="#fee0d2", alpha=0.5, label="Attention layer"),
    ]
    fig.legend(handles=patches, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))

    fig.suptitle(
        "Cross-Architecture Activation Patching: Logit-Difference Recovery\n"
        "Hybrid Mamba-Transformer (L0:Mamba → L1:Attn → L2:Mamba → L3:Attn)",
        fontweight="bold", y=1.06,
    )
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Figure 3: A-matrix eigenspectrum
# ---------------------------------------------------------------------------

def plot_eigenspectrum(
    eigen_results: list,    # list of EigenResult-like objects
    save_path: Optional[str] = None,
) -> plt.Figure:
    mamba_layers = [er for er in eigen_results]
    n = len(mamba_layers)
    if n == 0:
        return plt.figure()

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.8), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, er in zip(axes, mamba_layers):
        ts = np.array(er.decay_timescales)
        max_display = 200
        ts_clipped = np.where(np.isinf(ts), max_display, ts)
        dims = np.arange(len(ts_clipped))

        sc = ax.scatter(dims, ts_clipped, c=ts_clipped, cmap="plasma",
                        s=60, vmin=0, vmax=max_display, zorder=3)
        ax.axhline(1, color="gray", ls=":", lw=0.8)
        ax.set_xlabel("State dimension (sorted by eigenvalue magnitude)")
        ax.set_ylabel("Decay timescale τ (steps)" if er == mamba_layers[0] else "")
        ax.set_title(f"Mamba Layer {er.layer}", fontweight="bold")
        ax.set_ylim(-5, max_display + 15)

    plt.colorbar(sc, ax=axes[-1], label="τ (steps)", fraction=0.046, pad=0.04)
    fig.suptitle(
        "SSM State Dimension Memory Timescales\n"
        "(τ = −1/log|λ|; longer τ = longer memory)",
        fontweight="bold",
    )
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Figure 4: Probe AUROC heatmap
# ---------------------------------------------------------------------------

def plot_probe_decoding(
    probe_results: list,     # ProbeResult-like: layer, position, accuracy, chance, per_dim_score
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Left : decoding accuracy of B's identity vs read position, one line per
           Mamba layer (positions before the bigram = negative control).
    Right: per-dim F-score at the trigger position → which state dims carry B.
    """
    layers = sorted({pr.layer for pr in probe_results})
    positions = sorted({pr.position for pr in probe_results})
    chance = probe_results[0].chance if probe_results else 0.0
    trigger_pos = max(positions)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))

    colors = plt.cm.viridis(np.linspace(0.15, 0.8, max(len(layers), 2)))
    for c, li in zip(colors, layers):
        pts = sorted(
            [(pr.position, pr.accuracy) for pr in probe_results if pr.layer == li]
        )
        ax1.plot([p for p, _ in pts], [a for _, a in pts],
                 "o-", color=c, label=f"Mamba L{li}")
    ax1.axhline(chance, color="gray", ls=":", lw=1, label=f"chance = {chance:.3f}")
    ax1.set_xlabel("Read position of h_t")
    ax1.set_ylabel("Decode accuracy of B (64-way)")
    ax1.set_title("Is B's identity linearly decodable\nfrom the SSM state?")
    ax1.legend(fontsize=8)
    ax1.set_ylim(-0.03, 1.05)

    width = 0.8 / max(len(layers), 1)
    for k, (c, li) in enumerate(zip(colors, layers)):
        pr = next((p for p in probe_results
                   if p.layer == li and p.position == trigger_pos), None)
        if pr is None:
            continue
        scores = np.array(pr.per_dim_score)
        dims = np.arange(len(scores))
        ax2.bar(dims + k * width, scores, width, color=c, label=f"L{li}")
    ax2.set_xlabel("SSM state dimension")
    ax2.set_ylabel("ANOVA F-score")
    ax2.set_title(f"Which dims carry B at the trigger (pos {trigger_pos})?\n"
                  "(top dims are the ablation targets)")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Figure: 4-architecture comparison (Q1)
# ---------------------------------------------------------------------------

def plot_arch_comparison(
    patch_grids: dict[str, "object"],     # arch label → PatchGrid
    first_B_pos: int,                      # position of B in the fixed-distance set
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    For each architecture, the recovery per (layer, component) at the
    first_B position — the single most informative slice: 'which layer, of
    which type, carries the [A,B] binding forward?'
    """
    archs = list(patch_grids.keys())
    fig, axes = plt.subplots(1, len(archs), figsize=(3.6 * len(archs), 3.6),
                             sharey=True, squeeze=False)
    axes = axes[0]

    for ax, arch in zip(axes, archs):
        pg = patch_grids[arch]
        first_B = first_B_pos
        n_layers = pg.n_layers

        bottoms = np.arange(n_layers)
        for comp, grid in pg.grids.items():
            g = np.array(grid, dtype=float)
            vals = g[:, first_B]
            valid = ~np.isnan(vals)
            if not valid.any():
                continue
            ax.barh(bottoms[valid], vals[valid], height=0.6,
                    color=COMP_COLOR.get(comp, "gray"), alpha=0.85,
                    label=comp)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_yticks(range(n_layers))
        ax.set_yticklabels([f"L{i} ({t[0].upper()})"
                            for i, t in enumerate(pg.layer_types)])
        ax.set_xlabel("Recovery @ first_B_pos")
        ax.set_title(arch, fontweight="bold")
        ax.set_xlim(-0.2, 1.05)

    handles = [mpatches.Patch(color=c, label=k) for k, c in COMP_COLOR.items()]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.06))
    fig.suptitle("Where does the [A,B] binding live? — per architecture",
                 fontweight="bold", y=1.14)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Figure: circuit formation timeline (Q2)
# ---------------------------------------------------------------------------

def plot_formation(
    formation: dict,          # output of run_formation_analysis
    save_path: Optional[str] = None,
    top_k: int = 4,
) -> plt.Figure:
    recs = formation["records"]
    steps = [r["step"] for r in recs]
    accs = [r["val_acc"] for r in recs]

    # Rank (comp, layer, keypos) traces by final recovery; plot top_k
    last = recs[-1]["recovery"]
    traces = []
    for comp, layers in last.items():
        for li, poss in layers.items():
            for pname, v in poss.items():
                if v is not None:
                    traces.append((v, comp, li, pname))
    traces = sorted(traces, reverse=True)[:top_k]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax2 = ax.twinx()
    ax2.plot(steps, accs, color="black", lw=2.2, alpha=0.85, label="val accuracy")
    ax2.set_ylabel("Validation accuracy", color="black")
    ax2.set_ylim(-0.05, 1.05)

    ltypes = formation["layer_types"]
    for _, comp, li, pname in traces:
        ys = [r["recovery"][comp][li][pname] for r in recs]
        ys = [np.nan if y is None else y for y in ys]
        ax.plot(steps, ys, "o-", ms=3.5, lw=1.6,
                color=COMP_COLOR.get(comp, "gray"),
                alpha=0.9,
                label=f"{comp} L{li}({ltypes[int(li)][0].upper()}) @ {pname}")

    ax.set_xlabel("Training step")
    ax.set_ylabel("Patching recovery")
    ax.set_ylim(-0.15, 1.1)
    ax.set_title(f"Circuit formation — {formation['run_name']}\n"
                 "Does the mechanism (recovery) precede the behaviour (accuracy)?",
                 fontweight="bold")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Figure: distance sweep + eigenspectrum prediction (Q3)
# ---------------------------------------------------------------------------

def plot_distance(
    sweeps: dict[str, dict],       # arch label → run_distance_sweep output
    save_path: Optional[str] = None,
) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2))

    colors = plt.cm.tab10(np.linspace(0, 0.5, max(len(sweeps), 2)))
    chance = None
    for c, (arch, sw) in zip(colors, sweeps.items()):
        recs = sw["records"]
        ds = [r["distance"] for r in recs]
        accs = [r["accuracy"] for r in recs]
        ax1.plot(ds, accs, "o-", color=c, lw=1.8, label=arch)
        # mark out-of-distribution points
        ood = [(r["distance"], r["accuracy"]) for r in recs if not r["in_distribution"]]
        if ood:
            ax1.scatter([d for d, _ in ood], [a for _, a in ood],
                        facecolors="none", edgecolors=c, s=110, lw=1.6, zorder=5)
        if sw.get("trained_seq_len"):
            ax1.axvline(sw["trained_seq_len"], color="gray", ls="--", lw=0.9, alpha=0.5)
        chance = sw.get("chance_accuracy", chance)

    if chance:
        ax1.axhline(chance, color="gray", ls=":", lw=1)
        ax1.text(ax1.get_xlim()[0], chance + 0.02, "chance", fontsize=8, color="gray")
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("Induction distance d = A₂ − A₁ (tokens)")
    ax1.set_ylabel("Recall accuracy")
    ax1.set_title("Behaviour: recall vs distance\n(hollow markers = beyond training length)")
    ax1.legend(fontsize=8)
    ax1.set_ylim(-0.05, 1.08)

    # Right: accuracy vs predicted state survival, for architectures with SSM
    for c, (arch, sw) in zip(colors, sweeps.items()):
        recs = [r for r in sw["records"] if r.get("survival")]
        if not recs:
            continue
        ds = [r["distance"] for r in recs]
        best_surv = [
            max(max(v["survival_empirical"]) for v in r["survival"].values())
            for r in recs
        ]
        accs = [r["accuracy"] for r in recs]
        ax2.plot(ds, best_surv, "s--", color=c, lw=1.4, alpha=0.8,
                 label=f"{arch}: best-dim survival (empirical Δ)")
        ax2.plot(ds, accs, "o-", color=c, lw=1.8, alpha=0.9,
                 label=f"{arch}: accuracy")
        fixed = [
            max(max(v["survival_fixed_delta"]) for v in r["survival"].values())
            for r in recs
        ]
        ax2.plot(ds, fixed, ":", color=c, lw=1.2, alpha=0.6,
                 label=f"{arch}: survival (fixed Δ)")

    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("Induction distance d (tokens)")
    ax2.set_ylabel("Accuracy / survival factor")
    ax2.set_title("Prediction: does SSM state survival\n(exp(A·ΣΔ)) track the recall cliff?")
    ax2.legend(fontsize=7)
    ax2.set_ylim(-0.05, 1.08)

    fig.suptitle("Q3 — Memory limits of in-context recall", fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Figure 5: Ablation bar chart
# ---------------------------------------------------------------------------

def plot_ablation(
    ablation_results: list,
    save_path: Optional[str] = None,
) -> plt.Figure:
    if not ablation_results:
        return plt.figure()

    fig, ax = plt.subplots(figsize=(6, 4))
    labels = [f"L{r.layer}\ndims {r.ablated_dims}" for r in ablation_results]
    baselines = [r.baseline_accuracy for r in ablation_results]
    ablated = [r.val_accuracy for r in ablation_results]
    x = np.arange(len(labels))
    w = 0.35

    ax.bar(x - w/2, baselines, w, label="Baseline", color="#2166ac", alpha=0.8)
    ax.bar(x + w/2, ablated,   w, label="Ablated",  color="#fc8d59", alpha=0.9)

    for xi, (b, a) in enumerate(zip(baselines, ablated)):
        ax.text(xi + w/2, a + 0.01, f"−{b-a:.2f}", ha="center", fontsize=9, color="#d6604d")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Induction accuracy")
    ax.set_ylim(0, 1.15)
    ax.set_title("Accuracy drop when induction-critical SSM dims are ablated")
    ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Convenience: generate all figures from saved results
# ---------------------------------------------------------------------------

class _Obj:
    pass


def _to(d):
    o = _Obj()
    o.__dict__.update(d)
    return o


def make_all_figures(results_dir: str = "results", figures_dir: str = "figures") -> None:
    rdir = Path(results_dir)
    fdir = Path(figures_dir)
    fdir.mkdir(parents=True, exist_ok=True)
    from patch import load_patch_results

    # ── Q1: per-architecture patching heatmaps + cross-arch comparison ──
    arch_grids = {}
    meta_path = rdir / "patch_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    for pth in sorted(rdir.glob("patch_results_*.json")):
        arch = pth.stem.replace("patch_results_", "")
        pg = load_patch_results(pth)
        arch_grids[arch] = pg
        grids_np = {k: np.array(v, dtype=float) for k, v in pg.grids.items()}
        first_A = meta.get("first_A_pos", pg.seq_len - 1 - meta.get("patch_dist", pg.seq_len - 16))
        fig = plot_patching_heatmap(
            grids_np,
            layer_types=pg.layer_types,
            first_A_pos=first_A,
            first_B_pos=first_A + 1,
            second_A_pos=pg.seq_len - 1,
            save_path=str(fdir / f"figure_patching_heatmap_{arch}.png"),
        )
        plt.close(fig)
        print(f"Saved figure_patching_heatmap_{arch}.png")

    if len(arch_grids) >= 2 and meta:
        fig = plot_arch_comparison(
            arch_grids, first_B_pos=meta["first_A_pos"] + 1,
            save_path=str(fdir / "figure_arch_comparison.png"),
        )
        plt.close(fig)
        print("Saved figure_arch_comparison.png")

    # ── Q2: formation timelines ──
    for pth in sorted(rdir.glob("formation_*.json")):
        arch = pth.stem.replace("formation_", "")
        formation = json.loads(pth.read_text())
        fig = plot_formation(formation,
                             save_path=str(fdir / f"figure_formation_{arch}.png"))
        plt.close(fig)
        print(f"Saved figure_formation_{arch}.png")

    # ── Q3: distance sweep ──
    sweeps = {}
    for pth in sorted(rdir.glob("distance_*.json")):
        arch = pth.stem.replace("distance_", "")
        sweeps[arch] = json.loads(pth.read_text())
    if sweeps:
        fig = plot_distance(sweeps, save_path=str(fdir / "figure_distance.png"))
        plt.close(fig)
        print("Saved figure_distance.png")

    # ── Eigenspectrum / probes / ablation from analysis.json ──
    analysis_path = rdir / "analysis.json"
    if analysis_path.exists():
        data = json.loads(analysis_path.read_text())
        eigen = [_to(r) for r in data.get("eigen", [])]
        probes = [_to(r) for r in data.get("probe", [])]
        ablations = [_to(r) for r in data.get("ablation", [])]

        if eigen:
            fig = plot_eigenspectrum(eigen, save_path=str(fdir / "figure_eigenspectrum.png"))
            plt.close(fig)
            print("Saved figure_eigenspectrum.png")

        if probes:
            fig = plot_probe_decoding(probes,
                                      save_path=str(fdir / "figure_probe_decoding.png"))
            plt.close(fig)
            print("Saved figure_probe_decoding.png")

        if ablations:
            fig = plot_ablation(ablations, save_path=str(fdir / "figure_ablation.png"))
            plt.close(fig)
            print("Saved figure_ablation.png")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--figures_dir", default="figures")
    args = parser.parse_args()
    make_all_figures(args.results_dir, args.figures_dir)
