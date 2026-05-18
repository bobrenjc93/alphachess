# Expert Bootstrap: Lichess 2013-01 10k

## Data

- Source: `https://database.lichess.org/standard/lichess_db_standard_rated_2013-01.pgn.zst`
- Imported subset: first 10,000 games
- Positions: 651,314
- Local data path: `data/expert/lichess_2013_01_10k`
- Imported size: 47MB with sparse expert action labels

## Training

- Command: `scripts/submit_gpu_expert_train.sh`
- GPU: 1x L4 via `gpu-dev`
- Output checkpoint: `checkpoints/expert_lichess_10k/latest.pt`
- Model: 128 channels, 6 residual blocks
- Epochs: 1
- Batch size: 256
- Optimizer steps: 2,290

Final checkpoint metrics:

```text
loss=3.4395
policy_loss=2.6077
value_loss=0.8318
epoch_loss=4.1044
val_loss=3.4062
val_policy_loss=2.6055
val_value_loss=0.8007
```

## Smoke Evaluation

```bash
uv run alpha-chess eval \
  --checkpoint checkpoints/expert_lichess_10k/latest.pt \
  --games 4 \
  --simulations 8 \
  --max-plies 120
```

Result:

```text
score=3.0/4
score_rate=0.75
wins=2
draws=2
losses=0
```

This is an expert-bootstrap checkpoint, not a superhuman model. The next strength gate should evaluate against a UCI engine such as Stockfish and then use this checkpoint as the initial model for self-play iterations.
