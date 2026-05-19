# Loss-v4 Puzzle-Line Reduced-Value Iteration

Date: 2026-05-18

## Data

Generated refreshed AlphaChess-loss replay:

```text
out=data/teacher/alpha_loss_reports_v4
sources=41 Stockfish PGNs
games_seen=65
games_used=65
positions=222
min_value_delta=0.10
engine_time=0.03
player_name=AlphaChess
multipv=4
policy_temperature_cp=200.0
```

The data directory is an ignored local training artifact.

## GPU Iteration

- Base checkpoint: `experiments/focus-puzzlelines20-vw025-material015/checkpoints/iter_0001/latest.pt`
- Candidate checkpoint: `experiments/focus-lossv4-puzzlelines20-vw025-material015/checkpoints/iter_0001/latest.pt`
- Runner: `gpu-dev submit`, reservation `3729b3b7`, 1x L4
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
  - `data/teacher/alpha_loss_reports_v4`, weight `0.15`

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
`experiments/focus-puzzlelines20-vw025-material015/checkpoints/iter_0001/latest.pt`.

Final candidate training metrics:

```text
loss=3.2469
policy_loss=3.1233
policy_acc=0.3750
value_loss=0.4947
epoch_loss=1.8328
val_loss=2.1848
val_policy_loss=1.9880
val_policy_acc=0.4654
val_value_loss=0.7869
```

## Diagnostic Validation

Standalone validation on the fixed replay datasets:

```text
val_policy_acc=0.3576
val_source_0_policy_acc=0.5930  # Stockfish MultiPV 4096
val_source_1_policy_acc=0.3380  # old first-move puzzles 50k
val_source_2_policy_acc=0.5738  # AlphaChess loss replay v2
```

The fixed Stockfish diagnostic stayed close to the current best (`0.5986`) but
the candidate failed the internal league.

## Conclusion

This branch is rejected. Expanding the loss replay from 114 to 222 positions
did not improve internal strength when used as a drop-in replacement at the
same `0.15` sampler weight. The current best remains the reduced-value
puzzle-line checkpoint with the original loss-v3 replay.
