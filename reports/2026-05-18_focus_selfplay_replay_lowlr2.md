# Low-LR Replay-Mixed Self-Play Iteration

Date: 2026-05-18

## GPU Iteration

- Base checkpoint: `checkpoints/legal_multipv4096_focus_ft/latest.pt`
- Candidate checkpoint: `experiments/focus-selfplay-replay-lowlr2/checkpoints/iter_0001/latest.pt`
- Runner: `gpu-dev submit`, reservation `09479966`, 1x L4
- Self-play: 16 games, 32 simulations, 160 max plies
- Training: 1 epoch, batch size 128, learning rate `0.0001`
- Loss: `--legal-policy-loss`
- Data weights:
  - accumulated self-play total weight `0.10`
  - `data/teacher/stockfish_multipv_elo1800_4096`, weight `0.60`
  - `data/puzzles/all_1200_2400_50k`, weight `0.10`
  - `data/teacher/alpha_loss_reports`, weight `0.20`

Promotion gate against the base checkpoint:

```text
score=3.0/4
wins=2
draws=2
losses=0
score_rate=0.75
promoted=true
```

Final checkpoint training metrics:

```text
loss=1.6874
policy_loss=1.5971
policy_acc=0.6032
value_loss=0.0903
epoch_loss=1.8125
val_loss=2.7626
val_policy_loss=2.5254
val_policy_acc=0.3475
val_value_loss=0.2372
```

Training split source metrics:

```text
val_source_0_policy_acc=0.6635  # self-play, 211 examples
val_source_1_policy_acc=0.4147  # Stockfish MultiPV 4096, 434 examples
val_source_2_policy_acc=0.3278  # all puzzles 50k, 4969 examples
val_source_3_policy_acc=1.0000  # AlphaChess loss replay, 3 examples
```

## Diagnostic Validation

```bash
uv run alpha-chess validate \
  --checkpoint experiments/focus-selfplay-replay-lowlr2/checkpoints/iter_0001/latest.pt \
  --data data/teacher/stockfish_multipv_elo1800_4096 data/puzzles/all_1200_2400_50k data/teacher/alpha_loss_reports \
  --batch-size 256 \
  --legal-policy-loss \
  --device cpu
```

Result:

```text
val_policy_acc=0.3384
val_source_0_policy_acc=0.4829  # Stockfish MultiPV 4096
val_source_1_policy_acc=0.3263  # all puzzles 50k
val_source_2_policy_acc=1.0000  # AlphaChess loss replay
```

Same-data baseline from `checkpoints/legal_multipv4096_focus_ft/latest.pt`:

```text
val_policy_acc=0.3267
val_source_0_policy_acc=0.4448  # Stockfish MultiPV 4096
val_source_1_policy_acc=0.3167  # all puzzles 50k
val_source_2_policy_acc=1.0000  # AlphaChess loss replay
```

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
pgn=reports/focus_selfplay_replay_lowlr2_vs_stockfish_16sims.pgn
```

Stockfish smoke at 64 simulations:

```text
games=1
score=0.0
wins=0
draws=0
losses=1
pgn=reports/focus_selfplay_replay_lowlr2_s64_vs_stockfish.pgn
```

The 64-simulation loss dropped material in the opening after `4. Bc4 Nxe4`
and was mated on move 23.

## Conclusion

This is the first replay-mixed self-play candidate in this sequence to promote
against the prior best checkpoint while improving the fixed Stockfish MultiPV
diagnostic. It is still far below the Stockfish gate and should be treated as a
better training base, not as a solved model.
