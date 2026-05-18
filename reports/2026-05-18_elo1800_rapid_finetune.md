# Elo 1800+ Rapid/Classical Fine-Tune

## Filtered Data

- Source: `data/raw/lichess_db_standard_rated_2013-01.pgn.zst`
- Filter: both players `WhiteElo >= 1800` and `BlackElo >= 1800`
- Time filter: initial clock at least 180 seconds
- Games scanned: 121,332
- Games imported: 4,884
- Positions: 365,168
- Local data path: `data/expert/lichess_2013_01_elo1800_rapid`

## Training

- Starting checkpoint: `checkpoints/expert_lichess_10k_e2/latest.pt`
- Output checkpoint: `checkpoints/expert_lichess_elo1800_rapid_ft/latest.pt`
- Epochs: 1
- Batch size: 256
- Optimizer steps: 1,284

Final checkpoint metrics:

```text
loss=3.1735
policy_loss=2.3821
policy_acc=0.3235
value_loss=0.7914
epoch_loss=3.1445
val_loss=3.0568
val_policy_loss=2.2762
val_policy_acc=0.3673
val_value_loss=0.7806
```

## Evaluation

Uniform MCTS smoke:

```text
games=4
score=3.0
score_rate=0.75
wins=2
draws=2
losses=0
```

Stockfish 18 smoke:

```text
games=2
score=0.0
score_rate=0.0
wins=0
draws=0
losses=2
```

This is the best expert-move validation accuracy so far, but it still loses the
engine smoke. The next useful step is larger, newer high-Elo data and stronger
search/inference throughput.
