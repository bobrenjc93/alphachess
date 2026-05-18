# Expert Bootstrap Resume: Lichess 2013-01 10k E2

## Input

- Starting checkpoint: `checkpoints/expert_lichess_10k/latest.pt`
- Training data: `data/expert/lichess_2013_01_10k`
- Positions: 651,314
- GPU: 1x L4 via `gpu-dev`

## Training

- Output checkpoint: `checkpoints/expert_lichess_10k_e2/latest.pt`
- Epochs: 1 additional epoch
- Batch size: 256
- Optimizer steps in this run: 2,290
- Model: 128 channels, 6 residual blocks

Final checkpoint metrics:

```text
loss=3.2757
policy_loss=2.4485
policy_acc=0.3266
value_loss=0.8273
epoch_loss=3.0962
val_loss=3.2873
val_policy_loss=2.4914
val_policy_acc=0.3282
val_value_loss=0.7959
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

This checkpoint is a stronger supervised bootstrap by policy accuracy, but it
still fails the engine gate and should not be treated as strong chess play.
