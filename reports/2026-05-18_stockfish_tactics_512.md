# Stockfish Tactical Teacher Smoke

## Tactical Teacher Data

- Source PGN: `data/raw/lichess_db_standard_rated_2013-01.pgn.zst`
- Filter: both players Elo 1800+, initial clock at least 180 seconds
- Teacher: `tools/stockfish/bin/stockfish`
- Engine time per position: 0.005 seconds
- Candidate positions: every ply (`position_stride=1`)
- Kept only positions where the game move dropped value by at least 0.35
- Positions generated: 512
- Mean value drop: about 0.57
- Local data path: `data/teacher/stockfish_tactics_elo1800_512`

## Fine-Tune

- Starting checkpoint: `checkpoints/expert_lichess_elo1800_rapid_ft/latest.pt`
- Output checkpoint: `checkpoints/stockfish_tactics_512_ft/latest.pt`
- Epochs: 10
- Batch size: 128
- Learning rate: 0.0003

Final checkpoint metrics:

```text
loss=1.0351
policy_loss=0.9844
policy_acc=0.7273
value_loss=0.0507
epoch_loss=1.1730
val_loss=3.5437
val_policy_loss=3.4486
val_policy_acc=0.1961
val_value_loss=0.0952
```

## Evaluation

Stockfish 18 smoke:

```text
games=2
score=0.0
score_rate=0.0
wins=0
draws=0
losses=2
```

Sample losses are saved at `reports/stockfish_tactics_512_loss_sample.pgn`.

The tiny tactical set overfits and does not improve the Stockfish gate. The
generator is useful, but the model needs a much larger tactical teacher set and
probably mixed replay with expert games to avoid narrowing the policy too much.
