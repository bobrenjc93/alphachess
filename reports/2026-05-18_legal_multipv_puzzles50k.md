# Legal-Masked MultiPV + 50k Puzzle Fine-Tune

Date: 2026-05-18

## Puzzle Data

Imported a broader Lichess puzzle set:

```bash
uv run alpha-chess import-puzzles \
  --puzzles data/raw/lichess_db_puzzle.csv.zst \
  --out data/puzzles/all_1200_2400_50k \
  --min-rating 1200 \
  --max-rating 2400 \
  --max-positions 50000 \
  --chunk-size 4096
```

Result: 50,000 positions, 13 NPZ chunks, 6.2 MB compressed.

## Training

- Base checkpoint: `checkpoints/mixed_elo1800_matepuzzles20_ft/latest.pt`
- Output checkpoint: `checkpoints/legal_multipv_puzzles50k_ft/latest.pt`
- Runner: `gpu-dev submit`, reservation `8f147892`, 1x L4
- Epochs: 1
- Batch size: 512
- Loss: `--legal-policy-loss`
- Data:
  - `data/expert/lichess_2013_01_elo2000`, weight `0.35`
  - `data/teacher/stockfish_multipv_elo1800_1024`, weight `0.20`
  - `data/puzzles/all_1200_2400_50k`, weight `0.35`
  - `data/teacher/alpha_loss_reports`, weight `0.10`

Final checkpoint metrics:

```text
loss=1.8520
policy_loss=1.5226
policy_acc=0.5476
value_loss=0.3294
epoch_loss=2.2187
val_loss=2.6895
val_policy_loss=2.2404
val_policy_acc=0.3668
val_value_loss=0.4491
```

## Evaluation

Uniform opponent smoke:

```bash
uv run alpha-chess eval \
  --checkpoint checkpoints/legal_multipv_puzzles50k_ft/latest.pt \
  --games 4 \
  --simulations 8 \
  --max-plies 100
```

Result: `score=3.0/4`, `wins=2`, `draws=2`, `losses=0`.

Stockfish smoke:

```bash
uv run alpha-chess eval \
  --checkpoint checkpoints/legal_multipv_puzzles50k_ft/latest.pt \
  --opponent stockfish \
  --engine-path tools/stockfish/bin/stockfish \
  --engine-time 0.01 \
  --games 2 \
  --simulations 16 \
  --max-plies 120 \
  --pgn-out reports/legal_multipv_puzzles50k_stockfish_loss_sample.pgn
```

Result: `score=0.0/2`, `wins=0`, `draws=0`, `losses=2`.

Higher-search Stockfish smoke:

```bash
uv run alpha-chess eval \
  --checkpoint checkpoints/legal_multipv_puzzles50k_ft/latest.pt \
  --opponent stockfish \
  --engine-path tools/stockfish/bin/stockfish \
  --engine-time 0.01 \
  --games 1 \
  --simulations 64 \
  --max-plies 120 \
  --pgn-out reports/legal_multipv_puzzles50k_s64_stockfish_loss_sample.pgn
```

Result: `score=0.0/1`, `wins=0`, `draws=0`, `losses=1`.

## Conclusion

The broader puzzle set restores the uniform smoke result to 3/4, but validation
policy accuracy drops sharply and the checkpoint still fails every Stockfish
smoke game. Do not promote it.
