# Legal-Masked MultiPV Loss Replay Fine-Tune

Date: 2026-05-18

## Teacher Data

Generated a soft Stockfish teacher set with MultiPV policy targets:

```bash
uv run alpha-chess stockfish-teacher \
  --pgn data/raw/lichess_db_standard_rated_2013-01.pgn.zst \
  --out data/teacher/stockfish_multipv_elo1800_1024 \
  --engine-path tools/stockfish/bin/stockfish \
  --engine-time 0.01 \
  --max-positions 1024 \
  --min-elo 1800 \
  --min-initial-seconds 180 \
  --position-stride 8 \
  --multipv 4 \
  --policy-temperature-cp 200 \
  --chunk-size 512
```

Result: 1,024 positions across 106 games. The generated NPZs include dense
`policies` with about four nonzero legal moves per row.

## Training

- Base checkpoint: `checkpoints/mixed_elo1800_matepuzzles20_ft/latest.pt`
- Output checkpoint: `checkpoints/legal_multipv_loss_replay_ft/latest.pt`
- Runner: `gpu-dev submit`, reservation `6acc5277`, 1x L4
- Epochs: 1
- Batch size: 512
- Loss: `--legal-policy-loss`
- Data:
  - `data/expert/lichess_2013_01_elo2000`, weight `0.45`
  - `data/teacher/stockfish_multipv_elo1800_1024`, weight `0.20`
  - `data/puzzles/mate_1200_2400_5k`, weight `0.20`
  - `data/teacher/alpha_loss_reports`, weight `0.15`

Final checkpoint metrics:

```text
loss=1.6300
policy_loss=1.3517
policy_acc=0.6810
value_loss=0.2783
epoch_loss=1.7351
val_loss=2.5556
val_policy_loss=1.9114
val_policy_acc=0.4605
val_value_loss=0.6443
```

## Evaluation

Uniform opponent smoke:

```bash
uv run alpha-chess eval \
  --checkpoint checkpoints/legal_multipv_loss_replay_ft/latest.pt \
  --games 4 \
  --simulations 8 \
  --max-plies 100
```

Result: `score=2.0/4`, `wins=0`, `draws=4`, `losses=0`.

Stockfish smoke:

```bash
uv run alpha-chess eval \
  --checkpoint checkpoints/legal_multipv_loss_replay_ft/latest.pt \
  --opponent stockfish \
  --engine-path tools/stockfish/bin/stockfish \
  --engine-time 0.01 \
  --games 2 \
  --simulations 16 \
  --max-plies 120 \
  --pgn-out reports/legal_multipv_loss_replay_stockfish_loss_sample.pgn
```

Result: `score=0.0/2`, `wins=0`, `draws=0`, `losses=2`.

Higher-search Stockfish smoke:

```bash
uv run alpha-chess eval \
  --checkpoint checkpoints/legal_multipv_loss_replay_ft/latest.pt \
  --opponent stockfish \
  --engine-path tools/stockfish/bin/stockfish \
  --engine-time 0.01 \
  --games 1 \
  --simulations 64 \
  --max-plies 120 \
  --pgn-out reports/legal_multipv_loss_replay_s64_stockfish_loss_sample.pgn
```

Result: `score=0.0/1`, `wins=0`, `draws=0`, `losses=1`.

## Conclusion

MultiPV soft teacher targets work end to end and slightly improved validation
policy accuracy versus the prior legal-loss run. The checkpoint is still not a
strength improvement and still fails every Stockfish smoke game. Do not promote
it.
