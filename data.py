"""Synthetic task generators for mechanistic interpretability experiments.

Tasks:
  InductionDataset  — [rand...][A][B][rand...][A] → predict B
  CopyDataset       — [A][B][C]... → predict [A][B][C]... (simpler baseline)
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset, DataLoader


class InductionDataset(Dataset):
    """
    Each sequence has the form:

        [r_0, ..., r_{k-1}, A, B, r_{k+1}, ..., r_{m-1}, A]

    The model receives the full sequence as input and must predict B
    at the final position (position seq_len - 1, which holds the second A).

    For causal LM training we compute loss only at the last position.
    For the induction task to be non-trivial we require:
      - A ≠ B
      - A and B do not appear in the random filler (no false positives)
      - The first [A, B] pair is placed uniformly in [1, seq_len // 2 - 1]
      - The second A is always at position seq_len - 1

    Returns (input_ids, target_id, foil_id) where:
      target_id : B  — the correct token to predict
      foil_id   : B' — a per-example foil token, B' ≠ B ≠ A, used for the
                       logit-diff metric and the corruption protocol.
                       Storing it here guarantees the metric is never degenerate
                       (the old global FOIL_TOKEN=2 could equal B on some examples).
    """

    def __init__(
        self,
        n_sequences: int,
        seq_len: int = 64,
        vocab_size: int = 64,
        seed: int = 42,
        min_dist: int | None = None,
        max_dist: int | None = None,
    ):
        """
        min_dist / max_dist control the induction distance
            d = second_A_pos - first_A_pos = (seq_len - 1) - first_A_pos.
        The second A is always the last token; A₁ is placed at seq_len-1-d
        with d ~ Uniform[min_dist, max_dist].

        Defaults give broad coverage: d ∈ [3, seq_len-2]  (A₁ ∈ [1, L-4]).
        Setting min_dist == max_dist yields a fixed-distance dataset, which
        (a) enables batched activation patching (all critical positions align
        across examples) and (b) is used for the distance sweep in Q3.

        NOTE: broad distance coverage in *training* matters for Q3 — if the
        model only ever sees d ∈ [33, 62] (the old behaviour), a failure at
        d=100 confounds "memory decay" with "out-of-distribution distance".
        """
        assert seq_len >= 8, "seq_len must be at least 8"
        assert vocab_size >= 4, "vocab_size must be at least 4 (need A, B, B', plus filler)"
        self.seq_len = seq_len
        self.vocab_size = vocab_size

        if max_dist is None:
            max_dist = seq_len - 2          # A₁ at position 1
        if min_dist is None:
            min_dist = 3                    # A₁ at position seq_len - 4
        assert 2 <= min_dist <= max_dist <= seq_len - 2, (
            f"need 2 <= min_dist <= max_dist <= seq_len-2, "
            f"got [{min_dist}, {max_dist}] with seq_len={seq_len}"
        )
        self.min_dist, self.max_dist = min_dist, max_dist

        rng = torch.Generator()
        rng.manual_seed(seed)

        sequences: list[torch.Tensor] = []
        targets: list[int] = []
        foils: list[int] = []
        first_positions: list[int] = []

        for _ in range(n_sequences):
            # Sample key pair (A, B), A ≠ B
            A = int(torch.randint(0, vocab_size, (1,), generator=rng))
            B = int(torch.randint(0, vocab_size - 1, (1,), generator=rng))
            if B >= A:
                B += 1

            # Sample foil B' from vocab \ {A, B} — guaranteed ≠ B ≠ A
            foil_vocab = [t for t in range(vocab_size) if t != A and t != B]
            foil_idx = int(torch.randint(0, len(foil_vocab), (1,), generator=rng))
            B_prime = foil_vocab[foil_idx]

            # Sample distance and place A₁
            d = int(torch.randint(min_dist, max_dist + 1, (1,), generator=rng))
            first_pos = (seq_len - 1) - d          # ∈ [1, seq_len-4]

            # Filler tokens from vocab \ {A, B} (no false positives)
            filler_indices = torch.randint(0, len(foil_vocab), (seq_len,), generator=rng)
            seq = torch.tensor([foil_vocab[i] for i in filler_indices.tolist()])

            seq[first_pos] = A
            seq[first_pos + 1] = B
            seq[seq_len - 1] = A                   # trigger

            sequences.append(seq)
            targets.append(B)
            foils.append(B_prime)
            first_positions.append(first_pos)

        self.data = torch.stack(sequences)                 # (N, seq_len)
        self.targets = torch.tensor(targets)               # (N,)
        self.foils = torch.tensor(foils)                   # (N,)  — per-example B'
        self.first_positions = torch.tensor(first_positions)  # (N,) — pos of A₁

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.data[idx], self.targets[idx], self.foils[idx]


def sample_induction_batch(
    batch_size: int,
    seq_len: int = 64,
    vocab_size: int = 64,
    min_dist: int = 3,
    max_dist: int | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fresh, fully vectorised induction batch — for ONLINE training.

    Training on a fixed dataset for many epochs lets the model memorise
    (sequence → B) pairs instead of learning the induction algorithm: train
    loss collapses while validation stays at chance. Sampling fresh sequences
    every step removes the memorisation solution entirely — the only way to
    reduce loss is the general algorithm. This is the standard setup in the
    induction-heads literature.

    Returns (seqs (B, L), targets B (B,), foils B' (B,)).
    Same distribution as InductionDataset (which remains the fixed,
    reproducible dataset for evaluation/patching).
    """
    if max_dist is None:
        max_dist = seq_len - 2
    assert 2 <= min_dist <= max_dist <= seq_len - 2
    V = vocab_size

    A = torch.randint(0, V, (batch_size,), generator=generator)
    B = torch.randint(0, V - 1, (batch_size,), generator=generator)
    B = B + (B >= A).long()                       # A ≠ B

    # Sample from vocab \ {A, B} via the shift trick (requires ex1 < ex2)
    ex1 = torch.minimum(A, B)
    ex2 = torch.maximum(A, B)

    foils = torch.randint(0, V - 2, (batch_size,), generator=generator)
    foils = foils + (foils >= ex1).long()
    foils = foils + (foils >= ex2).long()         # B' ∉ {A, B}

    filler = torch.randint(0, V - 2, (batch_size, seq_len), generator=generator)
    filler = filler + (filler >= ex1[:, None]).long()
    filler = filler + (filler >= ex2[:, None]).long()   # no false positives

    d = torch.randint(min_dist, max_dist + 1, (batch_size,), generator=generator)
    first_pos = (seq_len - 1) - d

    seqs = filler
    rows = torch.arange(batch_size)
    seqs[rows, first_pos] = A
    seqs[rows, first_pos + 1] = B
    seqs[:, seq_len - 1] = A                      # trigger
    return seqs, B, foils


def sample_dense_induction_batch(
    batch_size: int,
    seq_len: int = 64,
    vocab_size: int = 64,
    n_pairs: int = 4,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    DENSE induction batch — the training signal that actually makes the
    circuit emerge.

    Diagnosis that motivated this: with loss at ONE position per sequence,
    online training pins the loss at ln(V) for thousands of steps — partial
    circuit components reduce the loss by ~nothing until the whole circuit
    works, so SGD never finds the staircase. The induction-heads literature
    trains with LM loss at MANY positions where induction helps. This sampler
    provides that: each sequence plants n_pairs bigrams [A_i, B_i], each
    repeated once later, and the loss mask marks every second-A position
    (where the next token B_i is predictable by induction). With n_pairs=4
    that is 4 supervision points per sequence at 4 different distances, and
    the state must hold several bindings concurrently.

    Construction: 2·n_pairs two-token slots are placed at sorted random
    positions with full positional coverage (no block grid). Earlier
    versions used a rigid 8-position block grid with in-block offsets 0-6;
    that left exact positional holes — A never sat at positions ≡ 7 (mod 8),
    B never at ≡ 0, first occurrences never in the final block — and the
    trained APE transformer inherited those holes surgically (perfect
    induction everywhere except distances that put A₁ on an untrained
    position). Slots are paired randomly; the earlier slot of each pair is
    the first occurrence. Distances range ~2 to ~seq_len − 2.

    Returns (seqs (B, L), loss_mask (B, L) bool, foils (B,)):
      loss at masked position t = CE(logits[t], seqs[t+1])  (next-token LM)
      foils: one token per row outside all pairs (for the logging LD metric).

    Evaluation stays on the SPARSE single-pair probe (InductionDataset): a
    general induction circuit trained densely must transfer to it, and that
    transfer is itself a check worth reporting.
    """
    k = n_pairs
    n_slots = 2 * k
    V = vocab_size
    assert V > 2 * k + 1
    # Slots occupy [p, p+1]; need p_{i+1} >= p_i + 2 and p_last <= L-2.
    q_max = (seq_len - 2) - 2 * (n_slots - 1)
    assert q_max >= 1, f"seq_len={seq_len} too short for n_pairs={k}"

    # 2k mutually-distinct tokens per row (argsort trick)
    toks = torch.argsort(
        torch.rand(batch_size, V, generator=generator), dim=1
    )[:, :2 * k]                                        # (B, 2k)
    A_toks = toks[:, :k]
    B_toks = toks[:, k:]
    ex_sorted, _ = torch.sort(toks, dim=1)              # (B, 2k) for exclusion

    # Filler from vocab \ {all pair tokens} via iterated shift trick
    seqs = torch.randint(0, V - 2 * k, (batch_size, seq_len), generator=generator)
    for j in range(2 * k):
        seqs = seqs + (seqs >= ex_sorted[:, j:j + 1]).long()

    # Non-overlapping slot positions with FULL coverage of [0, L-2]:
    # sample q sorted (ties allowed → touching slots, fine), p_i = q_i + 2i
    q = torch.randint(0, q_max + 1, (batch_size, n_slots), generator=generator)
    q, _ = torch.sort(q, dim=1)
    slot_pos = q + 2 * torch.arange(n_slots)            # (B, 2k), ascending

    # Pair slots randomly; earlier slot = first occurrence
    perm = torch.argsort(torch.rand(batch_size, n_slots, generator=generator), dim=1)
    rows = torch.arange(batch_size)
    loss_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)

    for i in range(k):
        s1, s2 = perm[:, 2 * i], perm[:, 2 * i + 1]
        first_slot = torch.minimum(s1, s2)              # smaller index = earlier pos
        second_slot = torch.maximum(s1, s2)
        p1 = slot_pos[rows, first_slot]                 # first [A_i, B_i]
        p2 = slot_pos[rows, second_slot]                # repeated [A_i, B_i]
        seqs[rows, p1] = A_toks[:, i]
        seqs[rows, p1 + 1] = B_toks[:, i]
        seqs[rows, p2] = A_toks[:, i]
        seqs[rows, p2 + 1] = B_toks[:, i]
        loss_mask[rows, p2] = True                      # predict B_i at t+1

    # One foil per row, outside all pair tokens
    foils = torch.randint(0, V - 2 * k, (batch_size,), generator=generator)
    for j in range(2 * k):
        foils = foils + (foils >= ex_sorted[:, j]).long()

    return seqs, loss_mask, foils


class CopyDataset(Dataset):
    """
    Baseline: sequences [A, B, C, ..., A, B, C, ...]
    The model must copy the first half into the second half.
    Target at each position in the second half is the corresponding token in the first half.

    We return (input_ids, target_id) where target_id is the token at position seq_len - 1
    (which is seq[seq_len//2 - 1] from the first half).
    """

    def __init__(
        self,
        n_sequences: int,
        seq_len: int = 64,
        vocab_size: int = 64,
        seed: int = 42,
    ):
        assert seq_len % 2 == 0
        half = seq_len // 2
        rng = torch.Generator()
        rng.manual_seed(seed)

        sequences: list[torch.Tensor] = []
        targets: list[int] = []

        for _ in range(n_sequences):
            first_half = torch.randint(0, vocab_size, (half,), generator=rng)
            seq = torch.cat([first_half, first_half])
            sequences.append(seq)
            # target at last position = last token of first half
            targets.append(int(first_half[-1]))

        self.data = torch.stack(sequences)
        self.targets = torch.tensor(targets)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.targets[idx]


def make_loader(
    dataset: Dataset,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def induction_pair_positions(seq: torch.Tensor, A: int) -> tuple[int, int]:
    """Return (first_pos, second_pos) of A in seq. Assumes exactly two occurrences."""
    positions = (seq == A).nonzero(as_tuple=True)[0].tolist()
    assert len(positions) >= 2, f"Expected >=2 occurrences of {A}, found {positions}"
    return positions[0], positions[-1]
