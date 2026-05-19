# Higher-Time Stockfish Teacher Puzzle-Line Iteration

Date: 2026-05-18

## Data

Generated a higher-analysis-time Stockfish MultiPV teacher set:

```text
out=data/teacher/stockfish_multipv_elo1800_4096_t005
source=data/raw/lichess_db_standard_rated_2013-01.pgn.zst
engine_time=0.05
games_seen=4789
games_used=215
positions=4096
files=4
multipv=4
policy_temperature_cp=200.0
```

The data directory is an ignored local training artifact.

## GPU Iteration

- Base checkpoint: `experiments/focus-puzzlelines20-vw025-material015/checkpoints/iter_0001/latest.pt`
- Candidate checkpoint: `experiments/focus-sf005-puzzlelines20-vw025-material015/checkpoints/iter_0001/latest.pt`
- Runner: `gpu-dev submit`, reservation `6101f175`, 1x L4
- Self-play: 16 games, 48 simulations, 180 max plies
- Evaluation blend: `--material-value-weight 0.15`
- Training: 1 epoch, batch size 128, learning rate `0.00003`
- Value loss weight: `0.25`
- Loss: `--legal-policy-loss`
- Promotion threshold: `0.55`
- Data weights:
  - accumulated self-play total weight `0.10`
  - `data/teacher/stockfish_multipv_elo1800_4096_t005`, weight `0.55`
  - `data/puzzles/lines_1200_2400_100k`, weight `0.20`
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
`experiments/focus-puzzlelines20-vw025-material015/checkpoints/iter_0001/latest.pt`.

Final candidate training metrics:

```text
loss=1.8491
policy_loss=1.7897
policy_acc=0.5385
value_loss=0.2376
epoch_loss=1.9627
val_loss=2.1659
val_policy_loss=1.9704
val_policy_acc=0.4554
val_value_loss=0.7820
```

## Diagnostic Validation

Standalone validation on the fixed old replay datasets:

```text
val_policy_acc=0.3530
val_source_0_policy_acc=0.5266  # old Stockfish MultiPV 4096
val_source_1_policy_acc=0.3386  # old first-move puzzles 50k
val_source_2_policy_acc=0.5410  # AlphaChess loss replay v2
```

Standalone validation on the training-recipe replay datasets:

```text
val_policy_acc=0.4633
val_source_0_policy_acc=0.5188  # 0.05s Stockfish MultiPV 4096
val_source_1_policy_acc=0.4606  # puzzle lines 100k
val_source_2_policy_acc=0.7719  # AlphaChess loss replay v3
```

## Conclusion

This branch is rejected. Replacing the original fixed Stockfish teacher with
the `0.05s` regenerated teacher hurt the internal league and sharply regressed
the old fixed Stockfish diagnostic. The higher-time teacher is not a drop-in
replacement; future teacher expansion should either preserve the original
fixed set in the mix or generate labels on exactly the same diagnostic
positions for a clean quality comparison.
