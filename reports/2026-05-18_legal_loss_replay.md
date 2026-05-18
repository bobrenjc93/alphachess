# Legal-Masked Loss Replay Fine-Tune

Date: 2026-05-18

## Training

- Base checkpoint: `checkpoints/mixed_elo1800_matepuzzles20_ft/latest.pt`
- Output checkpoint: `checkpoints/legal_weighted_loss_replay_ft/latest.pt`
- Runner: `gpu-dev submit`, reservation `68384261`, 1x L4
- Epochs: 1
- Batch size: 512
- Loss: `--legal-policy-loss`
- Data:
  - `data/expert/lichess_2013_01_elo2000`, weight `0.55`
  - `data/teacher/stockfish_tactics_elo1800_512`, weight `0.15`
  - `data/puzzles/mate_1200_2400_5k`, weight `0.20`
  - `data/teacher/alpha_loss_reports`, weight `0.10`

Final checkpoint metrics:

```text
loss=1.2796
policy_loss=0.8815
policy_acc=0.7545
value_loss=0.3981
epoch_loss=1.6606
val_loss=2.5528
val_policy_loss=1.9184
val_policy_acc=0.4514
val_value_loss=0.6344
```

## Evaluation

Uniform opponent smoke:

```bash
uv run alpha-chess eval \
  --checkpoint checkpoints/legal_weighted_loss_replay_ft/latest.pt \
  --games 4 \
  --simulations 8 \
  --max-plies 100
```

Result: `score=2.0/4`, `wins=0`, `draws=4`, `losses=0`.

Stockfish smoke:

```bash
uv run alpha-chess eval \
  --checkpoint checkpoints/legal_weighted_loss_replay_ft/latest.pt \
  --opponent stockfish \
  --engine-path tools/stockfish/bin/stockfish \
  --engine-time 0.01 \
  --games 2 \
  --simulations 16 \
  --max-plies 120 \
  --pgn-out reports/legal_loss_replay_stockfish_loss_sample.pgn
```

Result: `score=0.0/2`, `wins=0`, `draws=0`, `losses=2`.

Higher-search Stockfish smoke:

```bash
uv run alpha-chess eval \
  --checkpoint checkpoints/legal_weighted_loss_replay_ft/latest.pt \
  --opponent stockfish \
  --engine-path tools/stockfish/bin/stockfish \
  --engine-time 0.01 \
  --games 1 \
  --simulations 64 \
  --max-plies 120 \
  --pgn-out reports/legal_loss_replay_s64_stockfish_loss_sample.pgn
```

Result: `score=0.0/1`, `wins=0`, `draws=0`, `losses=1`.

## Conclusion

The legal-masked policy loss works end to end and the GPU job completed, but this
checkpoint is not stronger. It draws the uniform smoke games and still loses all
Stockfish smoke games, including a 64-simulation check. Do not promote it.
