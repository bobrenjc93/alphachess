# Puzzle-Line Reduced-Value Iteration 2

Date: 2026-05-18

## GPU Iteration

- Run directory: `experiments/focus-puzzlelines20-vw025-material015`
- Base checkpoint: `experiments/focus-puzzlelines20-vw025-material015/checkpoints/iter_0001/latest.pt`
- Candidate checkpoint: `experiments/focus-puzzlelines20-vw025-material015/checkpoints/iter_0002/latest.pt`
- Runner: `gpu-dev submit`, reservation `bfac92bf`, 1x L4
- Self-play: 16 games, 48 simulations, 180 max plies
- Evaluation blend: `--material-value-weight 0.15`
- Training: 1 epoch, batch size 128, learning rate `0.00003`
- Value loss weight: `0.25`
- Loss: `--legal-policy-loss`
- Promotion threshold: `0.55`
- Data weights:
  - accumulated self-play total weight `0.10`
  - `data/teacher/stockfish_multipv_elo1800_4096`, weight `0.55`
  - `data/puzzles/lines_1200_2400_100k`, weight `0.20`
  - `data/teacher/alpha_loss_reports_v3`, weight `0.15`

Promotion gate against iter_0001:

```text
score=2.0/8
wins=0
draws=4
losses=4
score_rate=0.25
promoted=false
```

The best checkpoint remains
`experiments/focus-puzzlelines20-vw025-material015/checkpoints/iter_0001/latest.pt`.

Final candidate training metrics:

```text
loss=1.5621
policy_loss=1.5095
policy_acc=0.6250
value_loss=0.2104
epoch_loss=1.7112
val_loss=2.2129
val_policy_loss=2.0189
val_policy_acc=0.4665
val_value_loss=0.7760
```

## Diagnostic Validation

Standalone validation on the fixed replay datasets:

```text
val_policy_acc=0.3555
val_source_0_policy_acc=0.5647  # Stockfish MultiPV 4096
val_source_1_policy_acc=0.3381  # old first-move puzzles 50k
val_source_2_policy_acc=0.5902  # AlphaChess loss replay v2
```

The fixed Stockfish diagnostic regressed from `0.5986` on iter_0001 to
`0.5647`.

## Conclusion

This branch is rejected. A second immediate iteration of the same reduced-value
puzzle-line recipe overfit the replay mix and lost strength against iter_0001.
The next useful branch should alter the data or search recipe rather than
continuing this exact loop.
