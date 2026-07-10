"""
Pipeline orchestrator — "How do SSM and attention layers divide the labour of
in-context recall in a hybrid?"

Architectures (same d_model, d_state, n_heads; params reported, not matched):
    hybrid_ma : M A M A   (Mamba first — the 'canonical' hybrid)
    hybrid_am : A M A M   (order control: does layer order change the circuit?)
    mamba     : M M M M   (pure-SSM baseline)
    attn      : A A A A   (pure-attention baseline; learned pos-emb, see model.py)

Phases:
    python run_all.py --phase train      # Q0: train all archs, save formation ckpts
    python run_all.py --phase patch      # Q1: localisation (batched patch scan)
    python run_all.py --phase formation  # Q2: circuit crystallisation over training
    python run_all.py --phase distance   # Q3: memory limits + eigenspectrum prediction
    python run_all.py --phase analyze    # eigen / decoding probes / ablation (main hybrid)
    python run_all.py --phase figures
    python run_all.py --phase all

Repeat with --seed 43 --seed_tag s43 (etc.) for error bars; every output file
is tagged so seeds do not overwrite each other.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

# ── Global experiment config ────────────────────────────────────────────────
VOCAB_SIZE  = 64
SEQ_LEN     = 64
N_STEPS     = 5_000
BATCH_SIZE  = 64
N_EXAMPLES  = 96          # patching examples (batched)
PATCH_DIST  = SEQ_LEN - 16   # fixed induction distance for patching (48)
CKPT_EVERY  = 250            # formation checkpoint cadence
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

ARCHS = {
    "hybrid_ma": "MAMA",
    "hybrid_am": "AMAM",
    "mamba":     "MMMM",
    "attn":      "AAAA",
}

RESULTS = Path("results")
FIGURES = Path("figures")
CKPTS   = RESULTS / "checkpoints"

for d in (RESULTS, FIGURES, CKPTS):
    d.mkdir(parents=True, exist_ok=True)


def _run_name(arch: str, seed_tag: str) -> str:
    return f"{arch}_{seed_tag}"


def _load_model(arch: str, seed_tag: str, which: str = "best"):
    from model import HybridLM
    from train import default_hybrid_config
    cfg = default_hybrid_config(vocab_size=VOCAB_SIZE, layer_pattern=ARCHS[arch])
    ckpt = CKPTS / f"{_run_name(arch, seed_tag)}_{which}.pt"
    if not ckpt.exists():
        sys.exit(f"Checkpoint not found: {ckpt}. Run --phase train first.")
    model = HybridLM(cfg)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    return model, cfg


# ---------------------------------------------------------------------------
# Phase 1: train all four architectures
# ---------------------------------------------------------------------------

def phase_train(seed: int, seed_tag: str) -> None:
    from model import HybridLM
    from train import default_hybrid_config, train, detect_phase_transition

    for arch, pattern in ARCHS.items():
        run_name = _run_name(arch, seed_tag)
        print("=" * 60)
        print(f"PHASE 1 — Training {arch} ({pattern}) | seed {seed} | {DEVICE}")
        print("=" * 60)

        cfg = default_hybrid_config(vocab_size=VOCAB_SIZE, layer_pattern=pattern)
        model = HybridLM(cfg)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{arch}: {n_params:,} params | pos_emb={cfg.use_pos_emb}")

        results = train(
            model,
            vocab_size=VOCAB_SIZE,
            seq_len=SEQ_LEN,
            n_steps=N_STEPS,
            batch_size=BATCH_SIZE,
            seed=seed,
            device=DEVICE,
            results_dir=str(RESULTS),
            figures_dir=str(FIGURES),
            run_name=run_name,
            ckpt_every=CKPT_EVERY,
            loss_mode="dense",     # LM loss on repeated bigrams — see data.py
            n_pairs=4,
        )
        pt = detect_phase_transition(results["log"])
        print(f"{arch}: phase transition ≈ step {pt} | "
              f"best val acc {results['best_val_acc']:.3f}\n")


# ---------------------------------------------------------------------------
# Phase 2 (Q1): localisation via batched activation patching
# ---------------------------------------------------------------------------

def phase_patch(seed: int, seed_tag: str) -> None:
    from data import InductionDataset
    from patch import run_patch_experiment, save_patch_results

    meta = None
    for arch in ARCHS:
        run_name = _run_name(arch, seed_tag)
        print("=" * 60)
        print(f"PHASE 2 — Patching {arch}")
        print("=" * 60)
        model, _ = _load_model(arch, seed_tag)

        # Fixed-distance eval set → aligned critical positions → batched scan
        ds = InductionDataset(N_EXAMPLES, seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE,
                              seed=99, min_dist=PATCH_DIST, max_dist=PATCH_DIST)

        # Sanity gate: don't interpret a circuit that doesn't exist
        with torch.no_grad():
            logits = model.to(DEVICE)(ds.data[:64].to(DEVICE))[:, -1, :]
            acc = (logits.argmax(-1).cpu() == ds.targets[:64]).float().mean().item()
        print(f"quick val acc @ d={PATCH_DIST}: {acc:.2f}")
        if acc < 0.5:
            print(f"WARNING: {arch} has not learned induction at this distance; "
                  "its heatmap is not interpretable as a circuit.")

        pg = run_patch_experiment(model, ds, n_examples=N_EXAMPLES,
                                  device=DEVICE, verbose=True)
        # un-tagged name is what figures.py consumes; tagged copy per seed
        save_patch_results(pg, RESULTS / f"patch_results_{arch}.json")
        seeds_dir = RESULTS / "seeds"
        seeds_dir.mkdir(exist_ok=True)
        save_patch_results(pg, seeds_dir / f"patch_results_{arch}_{seed_tag}.json")

        if meta is None:
            meta = {"first_A_pos": int(ds.first_positions[0]),
                    "patch_dist": PATCH_DIST, "seq_len": SEQ_LEN}
            (RESULTS / "patch_meta.json").write_text(json.dumps(meta, indent=2))

        top = []
        for comp, grid in pg.grids.items():
            mask = torch.isfinite(grid)
            for li, pos in mask.nonzero(as_tuple=False).tolist():
                top.append((float(grid[li, pos]), li, pos, comp))
        print("Top patching locations:")
        for val, li, pos, comp in sorted(top, reverse=True)[:5]:
            print(f"  L{li}({pg.layer_types[li]}) pos={pos:3d} {comp:8s} rec={val:.3f}")
        print()


# ---------------------------------------------------------------------------
# Phase 3 (Q2): circuit formation over training checkpoints
# ---------------------------------------------------------------------------

def phase_formation(seed: int, seed_tag: str) -> None:
    from train import default_hybrid_config
    from formation import run_formation_analysis

    for arch in ARCHS:
        run_name = _run_name(arch, seed_tag)
        print("=" * 60)
        print(f"PHASE 3 — Formation {arch}")
        print("=" * 60)
        cfg = default_hybrid_config(vocab_size=VOCAB_SIZE, layer_pattern=ARCHS[arch])
        try:
            run_formation_analysis(
                cfg, CKPTS, run_name,
                seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE,
                patch_dist=PATCH_DIST,
                device=DEVICE,
                out_path=RESULTS / f"formation_{arch}.json",
                max_step=N_STEPS,
            )
        except FileNotFoundError as e:
            print(f"skipping {arch}: {e}")


# ---------------------------------------------------------------------------
# Phase 4 (Q3): distance sweep + eigenspectrum prediction
# ---------------------------------------------------------------------------

def phase_distance(seed: int, seed_tag: str) -> None:
    from distance import run_distance_sweep

    for arch in ARCHS:
        print("=" * 60)
        print(f"PHASE 4 — Distance sweep {arch}")
        print("=" * 60)
        model, _ = _load_model(arch, seed_tag)
        run_distance_sweep(
            model,
            vocab_size=VOCAB_SIZE,
            trained_seq_len=SEQ_LEN,
            device=DEVICE,
            run_name=_run_name(arch, seed_tag),
            out_path=RESULTS / f"distance_{arch}.json",
        )


# ---------------------------------------------------------------------------
# Phase 5: eigen / probes / ablation on the main hybrid
# ---------------------------------------------------------------------------

def phase_analyze(seed: int, seed_tag: str) -> None:
    import numpy as np
    from data import InductionDataset
    from model import SimpleMambaBlock
    from analyze import (
        analyze_A_matrices, empirical_mean_delta, probe_state_dimensions,
        ablate_state_dims, save_analysis,
    )

    arch = "hybrid_ma"
    print("=" * 60)
    print(f"PHASE 5 — Mechanistic analysis ({arch})")
    print("=" * 60)
    model, cfg = _load_model(arch, seed_tag)

    ds = InductionDataset(500, seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE, seed=99,
                          min_dist=PATCH_DIST, max_dist=PATCH_DIST)
    first_B = int(ds.first_positions[0]) + 1

    print("1/3  Eigendecomposition with empirical Δ...")
    deltas = empirical_mean_delta(model, ds, device=DEVICE)
    print(f"     mean Δ per Mamba layer: { {k: round(v,4) for k,v in deltas.items()} }")
    eigen = analyze_A_matrices(model, delta_per_layer=deltas)

    positions = [first_B - 2, first_B, (first_B + SEQ_LEN - 1) // 2, SEQ_LEN - 1]
    print(f"2/3  Decoding B from h at positions {positions} "
          f"(pos {first_B - 2} = pre-bigram negative control)...")
    probes = probe_state_dimensions(model, ds, probe_positions=positions,
                                    n_examples=400, device=DEVICE)
    for pr in sorted(probes, key=lambda p: (p.layer, p.position)):
        print(f"     L{pr.layer} pos {pr.position:3d}: decode acc {pr.accuracy:.3f} "
              f"(chance {pr.chance:.3f})")

    print("3/3  Ablation of top SSM dims (+ random-dims control)...")
    rng = np.random.default_rng(seed)
    ablations = []
    for li, layer in enumerate(model.layers):
        if not isinstance(layer, SimpleMambaBlock):
            continue
        pr = next((p for p in probes if p.layer == li and p.position == SEQ_LEN - 1), None)
        if pr is None:
            continue
        scores = np.array(pr.per_dim_score)
        top2 = np.argsort(-scores)[:2].tolist()
        abl = ablate_state_dims(model, ds, layer=li, dims=top2, n_examples=200,
                                device=DEVICE)
        ablations.append(abl)
        print(f"     L{li} top dims {top2}: drop {abl.accuracy_drop:.3f}")

        # Control: same number of RANDOM dims. If this drops accuracy just as
        # much, the 'top dims' result is not specific.
        rand2 = rng.choice(cfg.d_state, size=2, replace=False).tolist()
        abl_r = ablate_state_dims(model, ds, layer=li, dims=rand2, n_examples=200,
                                  device=DEVICE)
        ablations.append(abl_r)
        print(f"     L{li} random dims {rand2}: drop {abl_r.accuracy_drop:.3f} (control)")

    save_analysis(eigen, probes, ablations, RESULTS / "analysis.json")
    print(f"Saved {RESULTS / 'analysis.json'}")


# ---------------------------------------------------------------------------

def phase_figures(seed: int, seed_tag: str) -> None:
    from figures import make_all_figures
    print("=" * 60)
    print("PHASE 6 — Figures")
    print("=" * 60)
    make_all_figures(results_dir=str(RESULTS), figures_dir=str(FIGURES))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True,
                        choices=["train", "patch", "formation", "distance",
                                 "analyze", "figures", "all"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed_tag", default="s42",
                        help="suffix for run names so multiple seeds coexist")
    parser.add_argument("--arch", default="all",
                        choices=["all"] + list(ARCHS.keys()),
                        help="run a single architecture (cheap iteration: "
                             "'attn' trains in ~1-2 min on GPU)")
    parser.add_argument("--n_steps", type=int, default=None,
                        help="override N_STEPS for this invocation")
    args = parser.parse_args()

    if args.arch != "all":
        ARCHS = {args.arch: ARCHS[args.arch]}
    if args.n_steps is not None:
        N_STEPS = args.n_steps

    phases = {
        "train": phase_train, "patch": phase_patch, "formation": phase_formation,
        "distance": phase_distance, "analyze": phase_analyze, "figures": phase_figures,
    }
    order = (["train", "patch", "formation", "distance", "analyze", "figures"]
             if args.phase == "all" else [args.phase])
    for name in order:
        phases[name](args.seed, args.seed_tag)
