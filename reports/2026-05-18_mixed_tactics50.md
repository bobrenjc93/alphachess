# Mixed Expert + Tactical Replay

## Data Mix

- Broad expert data: `data/expert/lichess_2013_01_elo1800_rapid`
- Tactical teacher data: `data/teacher/stockfish_tactics_elo1800_512`
- Tactical shard repeated: 50x
- Starting checkpoint: `checkpoints/expert_lichess_elo1800_rapid_ft/latest.pt`
- Output checkpoint: `checkpoints/mixed_elo1800_tactics50_ft/latest.pt`

The repetition makes the small tactical set visible during training without
throwing away broad expert opening/middlegame coverage.

## Training

- Epochs: 1
- Batch size: 256
- Learning rate: 0.0005
- Optimizer steps: 1,374

Final checkpoint metrics:

```text
loss=2.5393
policy_loss=1.8574
policy_acc=0.4657
value_loss=0.6818
epoch_loss=2.4332
val_loss=2.3805
val_policy_loss=1.7455
val_policy_acc=0.4867
val_value_loss=0.6351
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

Sample losses are saved at `reports/mixed_tactics50_loss_sample.pgn`.

The mixed replay improves validation accuracy on the mixed data but does not
yet improve the Stockfish gate. The loss PGNs still show hanging material and
mate-net blindness, so scale and search quality remain the limiting factors.
