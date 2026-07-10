# Division of Labour in Hybrid Mamba–Transformer In-Context Recall

**Research question:** How do SSM and attention layers divide the labour of
in-context recall (induction) in a hybrid model?

Three sub-questions, one shared infrastructure:

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

**Smoke test first** (the full pipeline is hours on CPU): temporarily set
`N_STEPS=200`, `CKPT_EVERY=50`, `N_EXAMPLES=16` in `run_all.py` and run
`--phase all`. Everything should complete without errors (accuracy will be
near chance — that's fine, you're testing plumbing, not science).

## How to read the results

**Patching heatmaps (Q1).** Recovery ≈ 1 (green) at a (layer, pos, component)
cell means restoring that clean activation into the corrupted run restores the
correct answer — the information "B follows A" flows through there. Grey =
not evaluated (component invalid for that layer type). Key asymmetry when
reading: patching Mamba `h` at position t propagates forward through the
recurrence (it fixes the state from t onward), whereas patching attention K/V
at t is local to that position. So a green stripe from `first_B_pos` to the
end in an `h` row means the state *carries* the binding; a single green cell
at `first_B_pos` in a K/V row means attention *reads* it there.

Expected hypotheses to test, not assume:
- `MAMA`: Mamba L0/L2 `h` recovery from `first_B_pos` onward (storage), attention
  K or V recovery concentrated at `first_B_pos` (retrieval) → a genuinely
  cross-architecture circuit.
- `AMAM`: does the circuit re-arrange, or does the model route around the order?
- `MMMM` vs `AAAA`: the known single-architecture solutions as anchors.

**Formation (Q2).** If mechanism (recovery) rises *before* accuracy, the
circuit assembles silently then "clicks"; if simultaneously, the phase
transition *is* circuit formation; compare which component moves first —
does the SSM storage or the attention retrieval crystallise first, and does
the order differ between `MAMA` and `AMAM`?

**Distance sweep (Q3).** Left panel: accuracy vs distance per architecture
(hollow markers = beyond training length; only pos-emb-free models are
evaluated there). Right panel: overlay of accuracy with the *predicted*
survival of the longest-lived SSM dims, `exp(A_n · ΣΔ_t)` measured with the
empirical (selective) Δ on real sequences, plus the fixed-Δ counterpart.
If the accuracy cliff co-locates with the empirical-survival cliff — and the
fixed-Δ curve misses it — you've shown that (a) the eigenspectrum + measured
Δ quantitatively predict the behavioural memory limit and (b) selectivity is
doing real work (the model modulates Δ to hold memory).

**Probes.** Decoding accuracy of B's identity from the channel-averaged state:
above chance after `first_B_pos` and at the trigger = the state linearly
carries the answer. The pre-bigram position is a built-in negative control
and must sit at chance (if it doesn't, you have leakage — check the dataset).

**Ablation.** Top-2 dims (by F-score) vs random-2 dims control. The result is
only meaningful if top-dims drop ≫ random-dims drop.

## Statistical hygiene

- ≥3 seeds; report mean ± std for every headline number.
- The `attn` baseline uses learned positional embeddings (a NoPE 4-layer
  attention model struggles to form a previous-token head); this is a
  confound to disclose, and why its distance sweep is capped at trained length.
- Parameter counts differ across architectures (Mamba blocks ≠ attention
  blocks); report counts and treat cross-architecture *accuracy* comparisons
  as secondary to *mechanism* comparisons.

## Files

- `model.py` — HybridLM with arbitrary M/A layer patterns; explicit SSM scan so h_t is patchable.
- `data.py` — induction task with controllable A₁…A₂ distance.
- `train.py` — training + formation checkpoints + phase-transition detection.
- `patch.py` — batched cross-architecture activation patching (logit-diff recovery).
- `formation.py` — Q2: recovery at critical positions across checkpoints.
- `distance.py` — Q3: distance sweep + empirical state-survival prediction.
- `analyze.py` — eigenspectrum (empirical Δ), multiclass decoding probes, ablation.
- `figures.py` — all publication figures.
