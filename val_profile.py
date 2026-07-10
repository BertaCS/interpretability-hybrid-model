"""Per-distance accuracy profile for a trained checkpoint — both eval paths.

Usage:
    python val_profile.py --arch attn --seed_tag s42

Prints, for every distance d:
  MIXED: accuracy on the d-subset of one mixed-distance dataset
         (exactly what training's val_acc averages over)
  FIXED: accuracy on a dedicated fixed-distance dataset
         (exactly what the distance sweep measures)

If the two columns agree (they should — verified on a reproduction), the
mixed val_acc is just the average of the FIXED column over d ∈ [3, L-2],
and any apparent val-vs-sweep contradiction dissolves into 'the sweep grid
sampled distances outside the circuit's current working region'.

Also prints the implied average of the FIXED column so it can be compared
directly against the training val_acc.
"""

from __future__ import annotations

import argparse

import torch

from data import InductionDataset
from model import HybridLM
from train import default_hybrid_config

ARCHS = {"hybrid_ma": "MAMA", "hybrid_am": "AMAM", "mamba": "MMMM", "attn": "AAAA"}


@torch.no_grad()
def _hits(model, data, targets, device, batch_size=128):
    out = []
    for s in range(0, len(data), batch_size):
        logits = model(data[s:s + batch_size].to(device))[:, -1, :]
        out.append((logits.argmax(-1).cpu() == targets[s:s + batch_size]).float())
    return torch.cat(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", required=True, choices=list(ARCHS.keys()))
    p.add_argument("--seed_tag", default="s42")
    p.add_argument("--ckpt", default=None, help="override checkpoint path")
    p.add_argument("--vocab_size", type=int, default=64)
    p.add_argument("--seq_len", type=int, default=64)
    p.add_argument("--n_mixed", type=int, default=6000)
    p.add_argument("--n_fixed", type=int, default=300)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = default_hybrid_config(vocab_size=args.vocab_size,
                                layer_pattern=ARCHS[args.arch])
    model = HybridLM(cfg)
    ckpt = args.ckpt or f"results/checkpoints/{args.arch}_{args.seed_tag}_best.pt"
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.to(device).eval()
    print(f"Loaded {ckpt} | device {device}\n")

    L = args.seq_len
    mixed = InductionDataset(args.n_mixed, seq_len=L, vocab_size=args.vocab_size,
                             seed=43)                      # same seed family as val
    mhits = _hits(model, mixed.data, mixed.targets, device)
    mdist = (L - 1) - mixed.first_positions

    print(f"{'d':>4} | {'MIXED acc':>9} (n) | {'FIXED acc':>9}")
    print("-" * 40)
    fixed_accs = []
    for d in range(3, L - 1):
        sel = mdist == d
        m_acc = float(mhits[sel].mean()) if sel.sum() > 0 else float("nan")
        fx = InductionDataset(args.n_fixed, seq_len=L, vocab_size=args.vocab_size,
                              seed=777, min_dist=d, max_dist=d)
        f_acc = float(_hits(model, fx.data, fx.targets, device).mean())
        fixed_accs.append(f_acc)
        marker = "  <-- disagreement!" if abs(m_acc - f_acc) > 0.15 else ""
        print(f"{d:>4} | {m_acc:>9.3f} ({int(sel.sum()):>3}) | {f_acc:>9.3f}{marker}")

    print("-" * 40)
    print(f"mean of FIXED column (= expected mixed val_acc): "
          f"{sum(fixed_accs) / len(fixed_accs):.3f}")
    print(f"overall MIXED accuracy:                          "
          f"{float(mhits.mean()):.3f}")


if __name__ == "__main__":
    main()
