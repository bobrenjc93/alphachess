# Stockfish Teacher Smoke

## Teacher Data

- Source PGN: `data/raw/lichess_db_standard_rated_2013-01.pgn.zst`
- Filter: both players Elo 1800+, initial clock at least 180 seconds
- Teacher: `tools/stockfish/bin/stockfish`
- Engine time per position: 0.005 seconds
- Position stride: every 8 plies from accepted games
- Positions generated: 1,024
- Local data path: `data/teacher/stockfish_elo1800_rapid_1024`

## Fine-Tune

- Starting checkpoint: `checkpoints/expert_lichess_elo1800_rapid_ft/latest.pt`
- Output checkpoint: `checkpoints/stockfish_teacher_1024_ft/latest.pt`
- Epochs: 8
- Batch size: 128
- Learning rate: 0.0005

Final checkpoint metrics:

```text
loss=0.2588
policy_loss=0.2134
policy_acc=1.0000
value_loss=0.0454
epoch_loss=0.2659
val_loss=3.0772
val_policy_loss=2.9877
val_policy_acc=0.3627
val_value_loss=0.0895
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

Sample losses are saved at `reports/stockfish_teacher_1024_loss_sample.pgn`.
The model still misses short tactics and mate threats, so the next teacher run
needs far more positions and probably tactical sampling rather than only
opening/middlegame positions from human PGNs.
