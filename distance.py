"""Q3 — Memory limits: at what A₁…A₂ distance does recall fail, and does the
SSM state survival predict it?

Behavioural side:
    For each distance d, build a fixed-distance induction set (seq_len = d+8)
    and measure accuracy + logit-diff at the trigger.
    Models trained without positional embeddings (anything containing Mamba)
    can be evaluated beyond the training length; pure-attention models with
    learned positions cannot — those points are flagged out-of-distribution.

Mechanistic side (the prediction):
    Information written into SSM dim n at position p survives to the trigger
    with factor  ∏_{t=p+1}^{L-1} exp(Δ_t · A_n)  =  exp(A_n · Σ_t Δ_t).
    Crucially Δ is *selective* (input-dependent), so we measure survival with
    the EMPIRICAL Δ on real sequences — not a fixed reference Δ. Comparing
    empirical vs fixed-Δ survival quantifies how much the model uses
    selectivity to hold memory. If accuracy collapses where the survival of
    the longest-lived dims collapses, the eigenspectrum + measured Δ jointly
    predict the behavioural memory limit.

Output: results/distance_<run_name>.json
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from data import InductionDataset
from model import HybridLM, SimpleMambaBlock


DEFAULT_DISTANCES = [4, 8, 12, 16, 24, 32, 48, 62, 96, 128, 192, 256]


@torch.no_grad()
def _eval_at_distance(
    model: HybridLM,
    d: int,
    seq_len: int,
    vocab_size: int,
    n_examples: int,
    device: str,
    seed: int,
    batch_size: int = 64,
) -> tuple[float, float]:
    ds = InductionDataset(n_examples, seq_len=seq_len, vocab_size=vocab_size,
                          seed=seed, min_dist=d, max_dist=d)
    correct = total = 0
    ld_sum = 0.0
    for start in range(0, len(ds), batch_size):
        seqs = ds.data[start:start + batch_size].to(device)
        targets = ds.targets[start:start + batch_size].to(device)
        foils = ds.foils[start:start + batch_size].to(device)
        logits = model(seqs)[:, -1, :]
        correct += (logits.argmax(-1) == targets).sum().item()
        total += targets.numel()
        c = logits.gather(1, targets.view(-1, 1)).squeeze(1)
        f = logits.gather(1, foils.view(-1, 1)).squeeze(1)
        ld_sum += (c - f).sum().item()
    return correct / total, ld_sum / total


@torch.no_grad()
def _state_survival(
    model: HybridLM,
    d: int,
    seq_len: int,
    vocab_size: int,
    device: str,
    seed: int,
    n_examples: int = 16,
) -> dict[int, dict]:
    """
    Per Mamba layer: survival factor exp(A_n · Σ_{t>p} Δ_t) from the write
    position p = first_B_pos to the trigger, per state dim n, averaged over
    examples and channels. Also the fixed-Δ counterpart using the mean Δ of
    the whole sequence.
    """
    ds = InductionDataset(n_examples, seq_len=seq_len, vocab_size=vocab_size,
                          seed=seed + 1, min_dist=d, max_dist=d)
    seqs = ds.data.to(device)
    first_B_pos = int(ds.first_positions[0]) + 1

    _, caches = model.run_with_cache(seqs)
    out: dict[int, dict] = {}
    for li, layer in enumerate(model.layers):
        if not isinstance(layer, SimpleMambaBlock):
            continue
        A = -torch.exp(layer.A_log.detach().float())          # (Di, N), negative
        delta = caches[li]["delta"].float()                    # (B, L, Di)

        # Empirical: Σ Δ_t over the retention window (write → trigger)
        S_emp = delta[:, first_B_pos + 1:, :].sum(dim=1)       # (B, Di)
        log_surv = torch.einsum("bd,dn->bdn", S_emp, A)        # (B, Di, N)
        surv_emp = log_surv.exp().mean(dim=(0, 1))             # (N,)

        # Fixed-Δ: same number of steps at the sequence-wide mean Δ
        n_steps = delta.shape[1] - (first_B_pos + 1)
        mean_dt = delta.mean(dim=(0, 1))                       # (Di,)
        log_surv_fix = torch.einsum("d,dn->dn", n_steps * mean_dt, A)
        surv_fix = log_surv_fix.exp().mean(dim=0)              # (N,)

        out[li] = {
            "survival_empirical": surv_emp.cpu().tolist(),     # per state dim
            "survival_fixed_delta": surv_fix.cpu().tolist(),
            "mean_delta": float(mean_dt.mean()),
            "retention_steps": int(n_steps),
        }
    return out


def run_distance_sweep(
    model: HybridLM,
    vocab_size: int = 64,
    trained_seq_len: int = 64,
    distances: list[int] | None = None,
    n_examples: int = 300,
    device: str = "cpu",
    seed: int = 777,
    out_path: str | Path | None = None,
    run_name: str = "model",
) -> dict:
    model = model.to(device)
    model.eval()
    distances = distances or DEFAULT_DISTANCES

    has_pos_emb = model.pos_emb is not None

    records = []
    for d in distances:
        if d <= trained_seq_len - 2:
            # In-range: evaluate at the TRAINED sequence length, varying only
            # A₁'s position — exactly the training/validation distribution.
            # (Evaluating short distances in short sequences broke the
            # pos-emb model: APE transformers trained on one fixed length
            # misfire systematically on other lengths, even shorter ones.)
            seq_len = trained_seq_len
            in_dist = True
        else:
            # Beyond trained length: only meaningful for models with no
            # positional embeddings (recurrence/conv generalise over length).
            if has_pos_emb:
                print(f"  d={d}: skipped (beyond trained length — learned "
                      "pos-emb model cannot extrapolate)")
                continue
            seq_len = d + 8
            in_dist = False

        acc, ld = _eval_at_distance(model, d, seq_len, vocab_size,
                                    n_examples, device, seed)
        surv = _state_survival(model, d, seq_len, vocab_size, device, seed)
        records.append({
            "distance": d,
            "seq_len": seq_len,
            "in_distribution": in_dist,
            "accuracy": acc,
            "logit_diff": ld,
            "survival": {str(k): v for k, v in surv.items()},
        })
        best = max(
            (max(v["survival_empirical"]) for v in surv.values()),
            default=float("nan"),
        )
        print(f"  d={d:4d} (L={seq_len:3d}) | acc {acc:.3f} | LD {ld:+.3f} | "
              f"best-dim survival {best:.3f}")

    result = {
        "run_name": run_name,
        "layer_pattern": model.config.layer_pattern,
        "trained_seq_len": trained_seq_len,
        "chance_accuracy": 1.0 / vocab_size,
        "records": records,
    }
    if out_path is not None:
        Path(out_path).write_text(json.dumps(result, indent=2))
        print(f"Saved distance sweep to {out_path}")
    return result
