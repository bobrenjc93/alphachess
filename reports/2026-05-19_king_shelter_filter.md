# King-Shelter Safety Filter

Timestamp: `2026-05-19T14:37:25-07:00`

## Change

The optional root king-safety filter now includes a pawn-shelter term for
non-endgame positions with a king still on the back two ranks. It penalizes:

- missing near pawns on the king file and adjacent files
- missing far pawns after the near pawn is gone
- open files near the king
- enemy rook/queen presence on open king-adjacent files

The root filter also caps each candidate move by its immediate static
king-shelter score before considering the short full-width lookahead. This
catches immediate pawn-shield weakening even when the lookahead's candidate move
set misses the attacking continuation.

## Verification

Tests:

```text
tests/test_mcts.py: 18 passed
full suite: 82 passed
```

New regression:

```text
test_mcts_root_king_safety_filter_penalizes_castled_pawn_shelter
```

The regression is based on a recent Stockfish loss line where `h2h3` weakened a
castled white king before a direct attack.

## Direct Checks

Both checks used local Stockfish with `engine_time=0.05`, `64` simulations,
`material_value_weight=0.15`, `material_value_search_plies=2`,
`root_king_safety_search_plies=1`, `root_king_safety_max_loss_cp=50`, and
`max_plies=180`.

| Checkpoint | Direct Stockfish score | PGN |
| --- | ---: | --- |
| `policyhead-16k-leafloss-qvalue-vw000-material015` | `0.0/2` | `reports/policyhead_16k_kingshelter50_stockfish.pgn` |
| `policyhead-hardneg16k-mix-bw005-v1` | `0.0/2` | `reports/policyhead_hardneg16k_mix_kingshelter50_stockfish.pgn` |

The shelter heuristic catches the targeted pawn-shield weakening pattern, but
it did not recover the direct Stockfish gate. The losses are still broader than
one static king-shelter issue.
