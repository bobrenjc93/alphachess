# Mixed Expert + Mate Puzzle Replay

## Data Mix

- Broad expert data: `data/expert/lichess_2013_01_elo1800_rapid`
- Puzzle data: `data/puzzles/mate_1200_2400_5k`
- Puzzle shard repeated: 20x
- Starting checkpoint: `checkpoints/expert_lichess_elo1800_rapid_ft/latest.pt`
- Output checkpoint: `checkpoints/mixed_elo1800_matepuzzles20_ft/latest.pt`

## Training

- Epochs: 1
- Batch size: 256
- Learning rate: 0.0005
- Optimizer steps: 1,636

Final checkpoint metrics:

```text
loss=2.0940
policy_loss=1.5418
policy_acc=0.5978
value_loss=0.5521
epoch_loss=2.2279
val_loss=2.0914
val_policy_loss=1.5127
val_policy_acc=0.5639
val_value_loss=0.5787
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

Sample losses are saved at `reports/mixed_matepuzzles20_loss_sample.pgn`.

The puzzle mix improves the supervised validation metric strongly but still
fails at actual play against Stockfish. The loss games show the model continues
to hang material and enter forced mates, so supervised tactics alone is not yet
translating into robust search-time play.
