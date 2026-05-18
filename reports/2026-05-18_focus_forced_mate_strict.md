# Forced-Mate Strict Replay Iteration

Date: 2026-05-18

## GPU Iteration

- Base checkpoint: `experiments/focus-tactical-strict/checkpoints/iter_0001/latest.pt`
- Candidate checkpoint: `experiments/focus-forced-mate-strict/checkpoints/iter_0001/latest.pt`
- Runner: `gpu-dev submit`, reservation `0b8a18f6`, 1x L4
- Self-play: 16 games, 48 simulations, 180 max plies
- Search: root forced-mate filter, 3 plies over checking lines
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
loss=2.0190
policy_loss=1.9535
policy_acc=0.5600
value_loss=0.0654
epoch_loss=1.7957
val_loss=2.6165
val_policy_loss=2.4596
val_policy_acc=0.3625
val_value_loss=0.1569
```

## Diagnostic Validation

```text
val_policy_acc=0.3572
val_source_0_policy_acc=0.5728  # Stockfish MultiPV 4096
val_source_1_policy_acc=0.3391  # all puzzles 50k
val_source_2_policy_acc=0.7213  # AlphaChess loss replay v2
```

This is the strongest fixed Stockfish MultiPV diagnostic so far.

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
pgn=reports/focus_forced_mate_strict_vs_stockfish_16sims.pgn
```

Stockfish smoke at 64 simulations:

```text
games=1
score=0.0
wins=0
draws=0
losses=1
pgn=reports/focus_forced_mate_strict_s64_vs_stockfish.pgn
```

## Conclusion

The forced-mate strict run improves internal league play and supervised
diagnostics, but it still loses every Stockfish game. The remaining gap is not
mate-in-one or short checking-line blindness alone; the engine needs broader
tactical search or much stronger value/policy training.
