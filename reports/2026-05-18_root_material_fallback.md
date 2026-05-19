# Root Material Filter Fallback

Date: 2026-05-18

## Change

The root material filter used to disable itself when every legal root move was
below the configured material threshold. That let severe tactical blunders
remain in forced-response positions.

Changed the all-unsafe fallback to keep only the best-scoring material moves
instead of returning every legal move.

## Regression Position

From a failed Stockfish smoke:

```text
r1bqk2r/1pNp1ppp/p1n1pn2/8/1b2PB2/2N5/PPP2PPP/R2QKB1R b KQkq - 1 8
```

Before the change, `8...Qxc7` remained legal under the root material guard even
though `9.Bxc7` wins the queen. After the change, the guard keeps only:

```text
Kf8
Ke7
```

and prunes:

```text
Qxc7
```

## Verification

```text
uv run pytest
43 passed
```

Targeted behavior check:

```text
Qxc7 kept: false
kept: Kf8, Ke7
```

## Evaluation

Checkpoint:

```text
experiments/focus-poisonedv2-qvalue-vw025-material015/checkpoints/iter_0001/latest.pt
```

Settings:

```text
material_value_weight=0.15
material_value_search_plies=2
root_material_search_plies=2
root_material_max_loss_cp=150
simulations=16
engine_time=0.05
```

Result after the fallback fix:

```text
stockfish 16 sims: 0.0/2
PGN: reports/focus_poisonedv2_rootguard150_fixed_vs_stockfish_16sims.pgn
```

Pre-fix comparison PGN:

```text
reports/focus_poisonedv2_rootguard150_vs_stockfish_16sims.pgn
```

The search fix removes a known bad fallback and changes the failed lines, but it
does not produce a direct Stockfish breakthrough by itself.
