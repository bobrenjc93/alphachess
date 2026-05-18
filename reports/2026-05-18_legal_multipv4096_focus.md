# Legal-Masked MultiPV 4096 Focus Fine-Tune

Date: 2026-05-18

## Teacher Data

Generated a larger soft Stockfish teacher set:

```bash
uv run alpha-chess stockfish-teacher \
  --pgn data/raw/lichess_db_standard_rated_2013-01.pgn.zst \
  --out data/teacher/stockfish_multipv_elo1800_4096 \
  --engine-path tools/stockfish/bin/stockfish \
  --engine-time 0.01 \
  --max-positions 4096 \
  --min-elo 1800 \
  --min-initial-seconds 180 \
  --position-stride 4 \
  --multipv 4 \
  --policy-temperature-cp 200 \
  --chunk-size 1024
```

Result: 4,096 positions across 215 games, with about four nonzero legal moves
per dense policy target.

## Training

- Base checkpoint: `checkpoints/legal_multipv_loss_replay_ft/latest.pt`
- Output checkpoint: `checkpoints/legal_multipv4096_focus_ft/latest.pt`
- Runner: `gpu-dev submit`, reservation `1ffedcb5`, 1x L4
- Epochs: 1
- Batch size: 512
- Learning rate: `0.0003`
- Loss: `--legal-policy-loss`
- Data:
  - `data/expert/lichess_2013_01_elo2000`, weight `0.25`
  - `data/teacher/stockfish_multipv_elo1800_4096`, weight `0.45`
  - `data/puzzles/mate_1200_2400_5k`, weight `0.10`
  - `data/teacher/alpha_loss_reports`, weight `0.20`

Final checkpoint metrics:

```text
loss=1.6372
policy_loss=1.4689
policy_acc=0.6636
value_loss=0.1683
epoch_loss=1.7806
val_loss=2.2795
val_policy_loss=1.7441
val_policy_acc=0.5114
val_value_loss=0.5354
```

Source validation from training split:

```text
val_source_0_policy_acc=0.4727  # expert
val_source_1_policy_acc=0.3976  # Stockfish MultiPV 4096
val_source_2_policy_acc=0.9857  # mate puzzles
val_source_3_policy_acc=1.0000  # AlphaChess loss replay, 2 examples
```

## Diagnostic Validation

Common diagnostic validation command:

```bash
uv run alpha-chess validate \
  --checkpoint checkpoints/legal_multipv4096_focus_ft/latest.pt \
  --data data/teacher/stockfish_multipv_elo1800_1024 data/puzzles/mate_1200_2400_5k data/teacher/alpha_loss_reports \
  --batch-size 512 \
  --legal-policy-loss \
  --device cpu
```

Result:

```text
val_policy_acc=0.9175
val_source_0_policy_acc=0.5439  # Stockfish MultiPV 1024
val_source_1_policy_acc=0.9936  # mate puzzles
val_source_2_policy_acc=1.0000  # AlphaChess loss replay
```

This is the best Stockfish MultiPV diagnostic accuracy so far, up from `0.5195`
on `checkpoints/legal_multipv_loss_replay_ft/latest.pt`.

## Evaluation

Uniform opponent smoke:

```bash
uv run alpha-chess eval \
  --checkpoint checkpoints/legal_multipv4096_focus_ft/latest.pt \
  --games 4 \
  --simulations 8 \
  --max-plies 100
```

Result: `score=4.0/4`, `wins=4`, `draws=0`, `losses=0`.

Stockfish smoke:

```bash
uv run alpha-chess eval \
  --checkpoint checkpoints/legal_multipv4096_focus_ft/latest.pt \
  --opponent stockfish \
  --engine-path tools/stockfish/bin/stockfish \
  --engine-time 0.01 \
  --games 2 \
  --simulations 16 \
  --max-plies 120 \
  --pgn-out reports/legal_multipv4096_focus_stockfish_loss_sample.pgn
```

Result: `score=0.0/2`, `wins=0`, `draws=0`, `losses=2`.

Higher-search Stockfish smoke:

```bash
uv run alpha-chess eval \
  --checkpoint checkpoints/legal_multipv4096_focus_ft/latest.pt \
  --opponent stockfish \
  --engine-path tools/stockfish/bin/stockfish \
  --engine-time 0.01 \
  --games 1 \
  --simulations 64 \
  --max-plies 120 \
  --pgn-out reports/legal_multipv4096_focus_s64_stockfish_loss_sample.pgn
```

Result: `score=0.0/1`, `wins=0`, `draws=0`, `losses=1`.

## Conclusion

This checkpoint is a better base than the prior legal MultiPV runs by diagnostic
policy accuracy and uniform play, but it still fails every Stockfish gate. Do not
promote it as a solved model.
