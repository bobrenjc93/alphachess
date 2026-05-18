# Strict Tactical Replay Iteration

Date: 2026-05-18

## GPU Iteration

- Base checkpoint: `experiments/focus-selfplay-replay-lowlr2/checkpoints/iter_0004/latest.pt`
- Candidate checkpoint: `experiments/focus-tactical-strict/checkpoints/iter_0001/latest.pt`
- Runner: `gpu-dev submit`, reservation `b0512594`, 1x L4
- Self-play: 24 games, 48 simulations, 180 max plies
- Training: 1 epoch, batch size 128, learning rate `0.00005`
- Loss: `--legal-policy-loss`
- Promotion threshold: `0.55`
- Data weights:
  - accumulated self-play total weight `0.10`
  - `data/teacher/stockfish_multipv_elo1800_4096`, weight `0.55`
  - `data/puzzles/all_1200_2400_50k`, weight `0.10`
  - `data/teacher/alpha_loss_reports_v2`, weight `0.25`

Promotion gate against the base checkpoint:

```text
score=6.0/8
wins=4
draws=4
losses=0
score_rate=0.75
promoted=true
```

Final checkpoint training metrics:

```text
loss=1.6389
policy_loss=1.5799
policy_acc=0.5433
value_loss=0.0590
epoch_loss=1.8661
val_loss=2.7374
val_policy_loss=2.5449
val_policy_acc=0.3498
val_value_loss=0.1925
```

## Diagnostic Validation

```bash
uv run alpha-chess validate \
  --checkpoint experiments/focus-tactical-strict/checkpoints/iter_0001/latest.pt \
  --data data/teacher/stockfish_multipv_elo1800_4096 data/puzzles/all_1200_2400_50k data/teacher/alpha_loss_reports_v2 \
  --batch-size 256 \
  --legal-policy-loss \
  --device cpu
```

Result:

```text
val_policy_acc=0.3530
val_source_0_policy_acc=0.5579  # Stockfish MultiPV 4096
val_source_1_policy_acc=0.3357  # all puzzles 50k
val_source_2_policy_acc=0.7541  # AlphaChess loss replay v2
```

This is the strongest 4k Stockfish MultiPV diagnostic so far.

## Evaluation

Uniform opponent:

```text
games=4
score=4.0
wins=4
draws=0
losses=0
```

Stockfish smoke at 16 simulations:

```text
games=2
score=0.0
wins=0
draws=0
losses=2
pgn=reports/focus_tactical_strict_vs_stockfish_16sims.pgn
```

Stockfish smoke at 64 simulations:

```text
games=1
score=0.0
wins=0
draws=0
losses=1
pgn=reports/focus_tactical_strict_s64_vs_stockfish.pgn
```

## Conclusion

The stricter tactical-filtered iteration is a real internal improvement and the
best supervised diagnostic checkpoint so far, but it still fails every Stockfish
gate. The next step needs a stronger tactical mechanism than shallow MCTS plus
supervised replay.
