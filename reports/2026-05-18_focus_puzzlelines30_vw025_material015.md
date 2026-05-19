# Reduced-Value Puzzle-Line Material-Prior Iteration

Date: 2026-05-18

## GPU Iteration

- Base checkpoint: `experiments/focus-material015-strict/checkpoints/iter_0001/latest.pt`
- Candidate checkpoint: `experiments/focus-puzzlelines30-vw025-material015/checkpoints/iter_0001/latest.pt`
- Runner: `gpu-dev submit`, reservation `c3f6301c`, 1x L4
- Self-play: 16 games, 48 simulations, 180 max plies
- Evaluation blend: `--material-value-weight 0.15`
- Root material guard: disabled (`root_material_search_plies=0`)
- Training: 1 epoch, batch size 128, learning rate `0.00004`
- Value loss weight: `0.25`
- Loss: `--legal-policy-loss`
- Promotion threshold: `0.55`
- Data weights:
  - accumulated self-play total weight `0.10`
  - `data/teacher/stockfish_multipv_elo1800_4096`, weight `0.45`
  - `data/puzzles/lines_1200_2400_100k`, weight `0.30`
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
loss=1.6914
policy_loss=1.5992
policy_acc=0.5360
value_loss=0.3688
epoch_loss=1.9603
val_loss=2.2721
val_policy_loss=2.0689
val_policy_acc=0.4477
val_value_loss=0.8128
```

## Diagnostic Validation

Standalone validation on the fixed replay datasets:

```text
val_policy_acc=0.3570
val_source_0_policy_acc=0.5632  # Stockfish MultiPV 4096
val_source_1_policy_acc=0.3399  # old first-move puzzles 50k
val_source_2_policy_acc=0.5574  # AlphaChess loss replay v2
```

Standalone validation on the new puzzle-line dataset:

```text
val_policy_acc=0.4450
val_value_loss=0.8383
```

Reducing value loss weight restored internal strength compared with the
full-value puzzle-line run, but fixed Stockfish MultiPV accuracy still trails
the previous best diagnostic (`0.5762`).

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
pgn=reports/focus_puzzlelines30_vw025_material015_vs_stockfish_16sims.pgn
```

Stockfish smoke at 64 simulations, material weight `0.15`:

```text
games=1
score=0.0
wins=0
draws=0
losses=1
pgn=reports/focus_puzzlelines30_vw025_material015_s64_vs_stockfish.pgn
```

## Conclusion

The reduced-value puzzle-line run is a real internal promotion and should be
used as the next self-play base. It does not solve the Stockfish gate. The next
branch should preserve the reduced value weighting and recover Stockfish policy
alignment, likely by increasing the Stockfish teacher share or adding stronger
Stockfish labels while keeping puzzle-line policy as an auxiliary source.
