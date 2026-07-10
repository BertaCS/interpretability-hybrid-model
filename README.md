# Division of Labour in Hybrid Mamba–Transformer In-Context Recall

How do SSM and attention layers divide the labour of
in-context recall (induction) in a hybrid model?

| Q | Question | Phase | Output |
|---|----------|-------|--------|
| Q1 | **Localisation** — where does the induction circuit live? | `patch` | `figure_patching_heatmap_*.png`, `figure_arch_comparison.png` |
| Q2 | **Formation** — when/how does it crystallise during training? | `formation` | `figure_formation_*.png` |
| Q3 | **Memory limits** — at what A₁…A₂ distance does recall fail, and does SSM state survival predict it? | `distance` | `figure_distance.png` |

Four architectures trained identically (same d_model=128, 4 layers, same data):
`MAMA` (hybrid, Mamba-first), `AMAM` (order control), `MMMM` (pure SSM),
`AAAA` (pure attention, learned positional embeddings — see caveat in `model.py`).

## Run

```bash
pip install -r requirements.txt
python run_all.py --phase all --seed 42 --seed_tag s42
# repeat with --seed 43 --seed_tag s43 (≥3 seeds for error bars)
```

Phases can be run separately: `train → patch → formation → distance → analyze → figures`.

## Files

- `model.py`: HybridLM with arbitrary M/A layer patterns; explicit SSM scan so h_t is patchable.
- `data.py`: induction task with controllable A₁…A₂ distance.
- `train.py`: training + formation checkpoints + phase-transition detection.
- `patch.py`: batched cross-architecture activation patching (logit-diff recovery).
- `formation.py`: Q2: recovery at critical positions across checkpoints.
- `distance.py`: Q3: distance sweep + empirical state-survival prediction.
- `analyze.py`: eigenspectrum (empirical Δ), multiclass decoding probes, ablation.
- `figures.py`: all publication figures.
