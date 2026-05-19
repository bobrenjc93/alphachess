# Puzzle-Line Material-Prior Replay Iteration

Date: 2026-05-18

## GPU Iteration

- Base checkpoint: `experiments/focus-material015-strict/checkpoints/iter_0001/latest.pt`
- Candidate checkpoint: `experiments/focus-puzzlelines30-material015/checkpoints/iter_0001/latest.pt`
- Runner: `gpu-dev submit`, reservation `79563984`, 1x L4
- Self-play: 16 games, 48 simulations, 180 max plies
- Evaluation blend: `--material-value-weight 0.15`
- Root material guard: disabled (`root_material_search_plies=0`)
- Training: 1 epoch, batch size 128, learning rate `0.00004`
- Loss: `--legal-policy-loss`
- Promotion threshold: `0.55`
- Data weights:
  - accumulated self-play total weight `0.10`
  - `data/teacher/stockfish_multipv_elo1800_4096`, weight `0.45`
  - `data/puzzles/lines_1200_2400_100k`, weight `0.30`
  - `data/teacher/alpha_loss_reports_v3`, weight `0.15`

Promotion gate against the base checkpoint:

```text
score=4.0/8
wins=4
draws=0
losses=4
score_rate=0.50
promoted=false
```

The best checkpoint remains
`experiments/focus-material015-strict/checkpoints/iter_0001/latest.pt`.

Final candidate training metrics:

```text
loss=1.9313
policy_loss=1.6145
policy_acc=0.5280
value_loss=0.3168
epoch_loss=2.2060
val_loss=2.8512
val_policy_loss=2.0757
val_policy_acc=0.4446
val_value_loss=0.7755
```

## Diagnostic Validation

Standalone validation on the fixed replay datasets:

```text
val_policy_acc=0.3569
val_source_0_policy_acc=0.5603  # Stockfish MultiPV 4096
val_source_1_policy_acc=0.3399  # old first-move puzzles 50k
val_source_2_policy_acc=0.5738  # AlphaChess loss replay v2
```

Standalone validation on the new puzzle-line dataset:

```text
val_policy_acc=0.4433
val_value_loss=0.7958
```

The model learns the puzzle-line policy labels, but the signed puzzle-line
values produce high value loss and the fixed Stockfish diagnostic still
regresses from `0.5762` to `0.5603`.

## Conclusion

This branch is rejected. Full puzzle-line policy supervision is more promising
than first-move-only puzzle weighting, but using normal value loss on signed
puzzle-line positions appears too disruptive. The next run should keep the
line-policy signal while reducing iteration value loss weight.
