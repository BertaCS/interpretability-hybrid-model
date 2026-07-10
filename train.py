"""Training loop for the Hybrid Mamba-Transformer model on the induction task.

Metric (from spec, corrected):
    Logit Difference = logit(B) - logit(B')
    where B is the correct token and B' is a specific foil token.
    This is evaluated strictly at the final position (the induction trigger).

    Note: the spec originally proposed logit(B) - max(incorrect_logits).
    We use the paired-foil version because it is stable across steps and
    directly tracks the patching metric used in Phase 2.

Outputs:
    - results/<run_name>_train_log.json   — per-step metrics
    - results/checkpoints/<run_name>_best.pt
    - figures/figure_phase_transition.png  — phase transition curve
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from data import (
    InductionDataset, make_loader, sample_induction_batch,
    sample_dense_induction_batch,
)
from model import HybridLM, HybridConfig


# ---------------------------------------------------------------------------
# Default config
# ---------------------------------------------------------------------------

def default_hybrid_config(
    vocab_size: int = 64,
    layer_pattern: str = "MAMA",
    use_pos_emb: bool | None = None,
    max_seq_len: int = 512,
) -> HybridConfig:
    # Pure-attention models need explicit position info to form a
    # previous-token head; Mamba-containing models get it from the conv/scan.
    if use_pos_emb is None:
        use_pos_emb = ("M" not in layer_pattern.upper())
    return HybridConfig(
        vocab_size=vocab_size,
        d_model=128,
        d_state=16,
        d_conv=4,
        expand=2,
        n_heads=4,
        layer_pattern=layer_pattern,
        use_pos_emb=use_pos_emb,
        max_seq_len=max_seq_len,
    )


# ---------------------------------------------------------------------------
# Logit difference metric
# ---------------------------------------------------------------------------

def _logit_diff_batch(
    logits: torch.Tensor,   # (B, V)
    targets: torch.Tensor,  # (B,) — correct token B
    foils: torch.Tensor,    # (B,) — per-example foil token B'
) -> float:
    """Mean logit(B) - logit(B') over the batch. Both targets and foils are per-example."""
    correct_logits = logits.gather(1, targets.unsqueeze(1)).squeeze(1)  # (B,)
    foil_logits = logits.gather(1, foils.unsqueeze(1)).squeeze(1)       # (B,)
    return (correct_logits - foil_logits).mean().item()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    model: HybridLM,
    vocab_size: int = 64,
    seq_len: int = 64,
    n_train: int = 10_000,
    n_val: int = 1_000,
    batch_size: int = 64,
    n_steps: int = 5_000,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    results_dir: str = "results",
    figures_dir: str = "figures",
    run_name: str = "hybrid_mamba_sota",
    log_every: int = 50,
    seed: int = 42,
    ckpt_every: int = 0,          # >0 → save formation checkpoints every N steps
    min_dist: int | None = None,  # induction distance range for training data
    max_dist: int | None = None,
    online_data: bool = True,     # fresh batch per step (prevents memorisation)
    loss_mode: str = "dense",     # 'dense' = LM loss at every repeated-bigram
                                  # position (n_pairs per seq — required for the
                                  # circuit to emerge); 'last' = single-query loss
    n_pairs: int = 4,             # bigram pairs per sequence in dense mode
) -> dict:
    """
    Train HybridLM on the induction task.

    Returns a results dict with:
        log      : list of per-step dicts with loss, logit_diff, val_acc
        best_val_acc : float
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    rdir = Path(results_dir)
    ckpt_dir = rdir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    Path(figures_dir).mkdir(parents=True, exist_ok=True)

    train_ds = InductionDataset(n_train, seq_len=seq_len, vocab_size=vocab_size,
                                seed=seed, min_dist=min_dist, max_dist=max_dist)
    val_ds = InductionDataset(n_val, seq_len=seq_len, vocab_size=vocab_size,
                              seed=seed + 1, min_dist=min_dist, max_dist=max_dist)
    train_loader = make_loader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = make_loader(val_ds, batch_size=batch_size, shuffle=False)

    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_steps, eta_min=lr * 0.1)
    criterion = nn.CrossEntropyLoss()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: HybridLM | Params: {n_params:,} | Device: {device}")
    print("Logit-diff metric: per-example logit(B) - logit(B') with B' ≠ B ≠ A")

    def _val_metrics() -> tuple[float, float]:
        """Returns (accuracy, mean_logit_diff) on validation set."""
        model.eval()
        correct = total = 0
        total_ld = 0.0
        n_batches = 0
        with torch.no_grad():
            for seqs, targets, foils in val_loader:
                seqs, targets, foils = seqs.to(device), targets.to(device), foils.to(device)
                logits = model(seqs)[:, -1, :]   # (B, V) at last position
                pred = logits.argmax(-1)
                correct += (pred == targets).sum().item()
                total += targets.numel()
                total_ld += _logit_diff_batch(logits, targets, foils)
                n_batches += 1
        model.train()
        return correct / total, total_ld / max(n_batches, 1)

    log: list[dict] = []
    best_val_acc = 0.0
    t0 = time.time()
    model.train()

    # ONLINE data: a fresh batch every step (see sample_induction_batch's
    # docstring — a fixed dataset gets memorised: train LD → +12 while val
    # stays at chance). train_ds/train_loader are kept only for online=False.
    data_rng = torch.Generator().manual_seed(seed + 12345)
    _min_d = min_dist if min_dist is not None else 3
    _max_d = max_dist if max_dist is not None else seq_len - 2
    train_iter = iter(train_loader)
    if loss_mode == "dense":
        print(f"Training: DENSE LM loss, {n_pairs} repeated bigrams/seq, "
              f"online fresh batches | eval on sparse single-pair probe")
    else:
        print(f"Training data: "
              f"{'ONLINE (fresh batch per step)' if online_data else 'fixed dataset'}"
              f" | distance range [{_min_d}, {_max_d}] | loss at last position")

    for step in range(1, n_steps + 1):
        if loss_mode == "dense":
            seqs, loss_mask, foils = sample_dense_induction_batch(
                batch_size, seq_len=seq_len, vocab_size=vocab_size,
                n_pairs=n_pairs, generator=data_rng,
            )
            seqs = seqs.to(device)
            loss_mask = loss_mask.to(device)
            foils = foils.to(device)

            logits_all = model(seqs)                       # (B, L, V)
            m = loss_mask[:, :-1]
            loss = criterion(logits_all[:, :-1][m], seqs[:, 1:][m])

            # LD logged at each row's LAST supervised position (longest gap)
            rows = torch.arange(seqs.shape[0], device=device)
            pos_idx = torch.arange(seq_len, device=device)
            last_q = (loss_mask * pos_idx).amax(dim=1)     # (B,)
            q_logits = logits_all[rows, last_q]            # (B, V)
            q_targets = seqs[rows, last_q + 1]
            ld_train = _logit_diff_batch(q_logits, q_targets, foils)
        else:
            if online_data:
                seqs, targets, foils = sample_induction_batch(
                    batch_size, seq_len=seq_len, vocab_size=vocab_size,
                    min_dist=_min_d, max_dist=_max_d, generator=data_rng,
                )
            else:
                try:
                    seqs, targets, foils = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    seqs, targets, foils = next(train_iter)

            seqs, targets, foils = seqs.to(device), targets.to(device), foils.to(device)

            # Predict at last position only
            logits = model(seqs)[:, -1, :]            # (B, V)
            loss = criterion(logits, targets)
            ld_train = _logit_diff_batch(logits, targets, foils)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % log_every == 0 or step == 1:
            val_acc, val_ld = _val_metrics()
            entry = {
                "step": step,
                "loss": float(loss.item()),
                "logit_diff_train": float(ld_train),
                "val_acc": float(val_acc),
                "val_logit_diff": float(val_ld),
                "lr": float(scheduler.get_last_lr()[0]),
                "elapsed_s": time.time() - t0,
            }
            log.append(entry)
            print(
                f"step {step:5d} | loss {loss.item():.4f} | "
                f"LD {ld_train:+.3f} | val_acc {val_acc:.3f} | "
                f"val_LD {val_ld:+.3f} | {entry['elapsed_s']:.1f}s"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), ckpt_dir / f"{run_name}_best.pt")

        # Formation checkpoints (Q2): dense early, sparser later is handled
        # by the caller choosing ckpt_every; step 1 always saved.
        if ckpt_every > 0 and (step % ckpt_every == 0 or step == 1):
            fdir = ckpt_dir / "formation"
            fdir.mkdir(exist_ok=True)
            torch.save(model.state_dict(), fdir / f"{run_name}_step{step:06d}.pt")

    torch.save(model.state_dict(), ckpt_dir / f"{run_name}_final.pt")

    results = {
        "run_name": run_name,
        "n_steps": n_steps,
        "best_val_acc": best_val_acc,
        "log": log,
    }
    (rdir / f"{run_name}_train_log.json").write_text(json.dumps(results, indent=2))
    print(f"\nDone. Best val acc: {best_val_acc:.3f}")

    # Generate phase transition figure immediately
    _plot_phase_transition(log, figures_dir, run_name)

    return results


def _plot_phase_transition(log: list[dict], figures_dir: str, run_name: str) -> None:
    """Save figure_phase_transition_<run_name>.png to figures_dir."""
    try:
        import matplotlib.pyplot as plt
        steps = [e["step"] for e in log]
        losses = [e["loss"] for e in log]
        accs = [e["val_acc"] for e in log]
        lds = [e["val_logit_diff"] for e in log]

        fig, axes = plt.subplots(1, 3, figsize=(13, 4))

        axes[0].plot(steps, losses, color="#2166ac", lw=1.8)
        axes[0].set_xlabel("Training step")
        axes[0].set_ylabel("Cross-entropy loss")
        axes[0].set_title("Loss curve")

        axes[1].plot(steps, accs, color="#1a9641", lw=1.8)
        axes[1].axhline(1.0, color="gray", ls=":", lw=0.8)
        axes[1].set_xlabel("Training step")
        axes[1].set_ylabel("Validation accuracy")
        axes[1].set_title("Induction accuracy (phase transition)")
        axes[1].set_ylim(-0.05, 1.1)

        axes[2].plot(steps, lds, color="#d6604d", lw=1.8)
        axes[2].axhline(0, color="gray", ls=":", lw=0.8)
        axes[2].set_xlabel("Training step")
        axes[2].set_ylabel("Logit diff: logit(B) − logit(B′)")
        axes[2].set_title("Logit difference (paired foil metric)")

        fig.suptitle(
            "Hybrid Mamba-Transformer — Induction Circuit Crystallization",
            fontweight="bold",
        )
        plt.tight_layout()
        out = Path(figures_dir) / f"figure_phase_transition_{run_name}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out}")
    except Exception as e:
        print(f"Could not save phase transition figure: {e}")


def detect_phase_transition(log: list[dict], metric: str = "val_acc", window: int = 5) -> int | None:
    """Return the step of steepest increase in `metric`."""
    if len(log) < 2 * window:
        return None
    vals = [(e["step"], e[metric]) for e in log if metric in e]
    max_jump, jump_step = 0.0, None
    for i in range(window, len(vals) - window):
        before = sum(v for _, v in vals[i - window:i]) / window
        after = sum(v for _, v in vals[i:i + window]) / window
        if (jump := after - before) > max_jump:
            max_jump, jump_step = jump, vals[i][0]
    return jump_step if max_jump > 0.05 else None


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from model import HybridLM, HybridConfig

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--vocab_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--figures_dir", default="figures")
    args = parser.parse_args()

    cfg = default_hybrid_config(vocab_size=args.vocab_size)
    model = HybridLM(cfg)

    results = train(
        model,
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        results_dir=args.results_dir,
        figures_dir=args.figures_dir,
    )

    pt = detect_phase_transition(results["log"])
    print(f"Phase transition detected at step: {pt}")
