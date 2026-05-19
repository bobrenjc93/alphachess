# Stockfish-Rebalanced Puzzle-Line Material-Prior Iteration

Date: 2026-05-18

## GPU Iteration

- Base checkpoint: `experiments/focus-puzzlelines30-vw025-material015/checkpoints/iter_0001/latest.pt`
- Candidate checkpoint: `experiments/focus-puzzlelines20-vw025-material015/checkpoints/iter_0001/latest.pt`
- Runner: `gpu-dev submit`, reservation `4e44ce72`, 1x L4
- Self-play: 16 games, 48 simulations, 180 max plies
- Evaluation blend: `--material-value-weight 0.15`
- Root material guard: disabled (`root_material_search_plies=0`)
- Training: 1 epoch, batch size 128, learning rate `0.00004`
- Value loss weight: `0.25`
- Loss: `--legal-policy-loss`
- Promotion threshold: `0.55`
- Data weights:
  - accumulated self-play total weight `0.10`
  - `data/teacher/stockfish_multipv_elo1800_4096`, weight `0.55`
  - `data/puzzles/lines_1200_2400_100k`, weight `0.20`
  - `data/teacher/alpha_loss_reports_v3`, weight `0.15`

Promotion gate against the base checkpoint:

```text
score=6.0/8
wins=4
draws=4
losses=0
score_rate=0.75
promoted=true
```

This is the new internal best checkpoint.

Final candidate training metrics:

```text
loss=1.6473
policy_loss=1.6093
policy_acc=0.6111
value_loss=0.1517
epoch_loss=1.7733
val_loss=2.2205
val_policy_loss=2.0235
val_policy_acc=0.4566
val_value_loss=0.7878
```

## Diagnostic Validation

Standalone validation on the fixed replay datasets:

```text
val_policy_acc=0.3548
val_source_0_policy_acc=0.5986  # Stockfish MultiPV 4096
val_source_1_policy_acc=0.3345  # old first-move puzzles 50k
val_source_2_policy_acc=0.5902  # AlphaChess loss replay v2
```

Standalone validation on the new puzzle-line dataset:

```text
val_policy_acc=0.4491
val_value_loss=0.8182
```

This is the strongest fixed Stockfish MultiPV diagnostic so far, improving
from `0.5762` to `0.5986`.

## Evaluation

Uniform opponent, material weight `0.15`:

```text
games=4
score=4.0
wins=4
draws=0
losses=0
```

Stockfish smoke at 16 simulations, material weight `0.15`:

```text
games=2
score=0.0
wins=0
draws=0
losses=2
pgn=reports/focus_puzzlelines20_vw025_material015_vs_stockfish_16sims.pgn
```

Stockfish smoke at 64 simulations, material weight `0.15`:

```text
games=1
score=0.0
wins=0
draws=0
losses=1
pgn=reports/focus_puzzlelines20_vw025_material015_s64_vs_stockfish.pgn
```

Stockfish smoke at 128 simulations, material weight `0.15`:

```text
games=1
score=0.0
wins=0
draws=0
losses=1
pgn=reports/focus_puzzlelines20_vw025_material015_s128_vs_stockfish.pgn
```

## Conclusion

The Stockfish-rebalanced puzzle-line run is a strong internal promotion and
the best supervised Stockfish diagnostic so far. It still loses direct
Stockfish games, so the model remains far from superhuman. The next branch
should continue from this checkpoint and either expand higher-quality
Stockfish labels or improve search enough to convert the better policy
diagnostic into actual play.
