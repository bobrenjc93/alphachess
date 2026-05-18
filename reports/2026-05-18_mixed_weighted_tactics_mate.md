# Mixed Weighted Tactics + Mate Fine-Tune

Date: 2026-05-18

## Training

- Base checkpoint: `checkpoints/mixed_elo1800_matepuzzles20_ft/latest.pt`
- Output checkpoint: `checkpoints/mixed_weighted_elo1800_tactics_mate_ft/latest.pt`
- Runner: `gpu-dev submit`, reservation `7144e859`, 1x L4
- Epochs: 1
- Batch size: 512
- Data:
  - `data/expert/lichess_2013_01_elo1800_rapid`, weight `0.65`
  - `data/teacher/stockfish_tactics_elo1800_512`, weight `0.15`
  - `data/puzzles/mate_1200_2400_5k`, weight `0.20`

Final checkpoint metrics:

```text
loss=1.4914
policy_loss=1.1023
policy_acc=0.6967
value_loss=0.3891
epoch_loss=1.5071
val_loss=2.4902
val_policy_loss=1.8397
val_policy_acc=0.4652
val_value_loss=0.6505
```

## Evaluation

Uniform opponent smoke:

```bash
uv run alpha-chess eval \
  --checkpoint checkpoints/mixed_weighted_elo1800_tactics_mate_ft/latest.pt \
  --games 4 \
  --simulations 8 \
  --max-plies 100
```

Result: `score=3.0/4`, `wins=2`, `draws=2`, `losses=0`.

Stockfish smoke:

```bash
uv run alpha-chess eval \
  --checkpoint checkpoints/mixed_weighted_elo1800_tactics_mate_ft/latest.pt \
  --opponent stockfish \
  --engine-path tools/stockfish/bin/stockfish \
  --engine-time 0.01 \
  --games 2 \
  --simulations 16 \
  --max-plies 120 \
  --pgn-out reports/mixed_weighted_stockfish_loss_sample.pgn
```

Result: `score=0.0/2`, `wins=0`, `draws=0`, `losses=2`.

Higher-search Stockfish smoke:

```bash
uv run alpha-chess eval \
  --checkpoint checkpoints/mixed_weighted_elo1800_tactics_mate_ft/latest.pt \
  --opponent stockfish \
  --engine-path tools/stockfish/bin/stockfish \
  --engine-time 0.01 \
  --games 1 \
  --simulations 64 \
  --max-plies 120 \
  --pgn-out reports/mixed_weighted_s64_stockfish_loss_sample.pgn
```

Result: `score=0.0/1`, `wins=0`, `draws=0`, `losses=1`.

## Conclusion

The weighted replay path works end to end and produced a synced GPU checkpoint, but
this run is not a strength improvement. Validation policy accuracy dropped versus
the previous mate-heavy mixed run, and the checkpoint still loses both Stockfish
smoke games, including a 64-simulation check. Do not promote it.
