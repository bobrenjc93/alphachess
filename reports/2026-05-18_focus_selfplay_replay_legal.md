# Replay-Mixed Self-Play Iteration

Date: 2026-05-18

## Code Changes

This run used the iteration loop added in `9697926`, which allows fixed replay
datasets to be mixed with accumulated self-play using source-balanced sampling.

## GPU Iteration

- Base checkpoint: `checkpoints/legal_multipv4096_focus_ft/latest.pt`
- Candidate checkpoint: `experiments/focus-selfplay-replay-legal/checkpoints/iter_0001/latest.pt`
- Runner: `gpu-dev submit`, reservation `1d8b738d`, 1x L4
- Self-play: 8 games, 32 simulations, 160 max plies
- Training: 1 epoch, batch size 64, learning rate `0.001`
- Loss: `--legal-policy-loss`
- Data weights:
  - accumulated self-play total weight `0.25`
  - `data/teacher/stockfish_multipv_elo1800_4096`, weight `0.45`
  - `data/puzzles/all_1200_2400_50k`, weight `0.10`
  - `data/teacher/alpha_loss_reports`, weight `0.20`

Promotion gate against the base checkpoint:

```text
score=1.0/4
wins=0
draws=2
losses=2
score_rate=0.25
promoted=false
```

Final checkpoint training metrics:

```text
loss=1.4151
policy_loss=1.3854
policy_acc=0.6522
value_loss=0.0297
epoch_loss=1.3460
val_loss=2.9355
val_policy_loss=2.7268
val_policy_acc=0.2917
val_value_loss=0.2088
```

Training split source metrics:

```text
val_source_0_policy_acc=0.4737  # self-play, 95 examples
val_source_1_policy_acc=0.3365  # Stockfish MultiPV 4096, 416 examples
val_source_2_policy_acc=0.2843  # all puzzles 50k, 5001 examples
val_source_3_policy_acc=1.0000  # AlphaChess loss replay, 1 example
```

## Diagnostic Validation

```bash
uv run alpha-chess validate \
  --checkpoint experiments/focus-selfplay-replay-legal/checkpoints/iter_0001/latest.pt \
  --data data/teacher/stockfish_multipv_elo1800_4096 data/puzzles/all_1200_2400_50k data/teacher/alpha_loss_reports \
  --batch-size 256 \
  --legal-policy-loss \
  --device cpu
```

Result:

```text
val_policy_acc=0.3407
val_source_0_policy_acc=0.4573  # Stockfish MultiPV 4096
val_source_1_policy_acc=0.3309  # all puzzles 50k
val_source_2_policy_acc=1.0000  # AlphaChess loss replay
```

The Stockfish MultiPV accuracy regressed from the prior focused checkpoint, so
this candidate should not replace `checkpoints/legal_multipv4096_focus_ft/latest.pt`.

## Evaluation

Uniform opponent:

```text
games=4
score=4.0
wins=4
draws=0
losses=0
```

Stockfish smoke:

```bash
uv run alpha-chess eval \
  --checkpoint experiments/focus-selfplay-replay-legal/checkpoints/iter_0001/latest.pt \
  --opponent stockfish \
  --engine-path tools/stockfish/bin/stockfish \
  --engine-time 0.05 \
  --games 2 \
  --simulations 16 \
  --max-plies 160 \
  --pgn-out reports/focus_selfplay_replay_legal_vs_stockfish_16sims.pgn
```

Result: `score=0.0/2`, `wins=0`, `draws=0`, `losses=2`.

The losses were tactical mate failures:

- Game 1: as White, allowed `...Nf2#` after `10. h3 Qh4 11. g3 Bxg3`.
- Game 2: as Black, allowed `Qf8#` after losing material and coordination in a Sicilian structure.

## Conclusion

Replay-mixed iteration plumbing works, but this small self-play update is weaker
than the focused supervised checkpoint. The next useful change is to reduce
self-play learning rate and/or increase fixed replay weight before attempting
another promotion candidate.
