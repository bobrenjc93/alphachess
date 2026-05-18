# Material-Prior Strict Replay Iteration

Date: 2026-05-18

## GPU Iteration

- Base checkpoint: `experiments/focus-forced-mate-strict/checkpoints/iter_0001/latest.pt`
- Candidate checkpoint: `experiments/focus-material015-strict/checkpoints/iter_0001/latest.pt`
- Runner: `gpu-dev submit`, reservation `f6f576fa`, 1x L4
- Self-play: 16 games, 48 simulations, 180 max plies
- Search: root forced-mate filter, 3 plies over checking lines
- Evaluation blend: `--material-value-weight 0.15`
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
loss=1.7121
policy_loss=1.6600
policy_acc=0.5366
value_loss=0.0521
epoch_loss=1.7547
val_loss=2.6510
val_policy_loss=2.4738
val_policy_acc=0.3658
val_value_loss=0.1773
```

## Diagnostic Validation

Standalone validation on the replay datasets:

```text
val_policy_acc=0.3617
val_source_0_policy_acc=0.5762  # Stockfish MultiPV 4096
val_source_1_policy_acc=0.3437  # all puzzles 50k
val_source_2_policy_acc=0.7541  # AlphaChess loss replay v2
```

This is the strongest fixed Stockfish MultiPV diagnostic so far.

## Evaluation

Uniform opponent, material weight `0.15`:

```text
games=4
score=4.0
wins=4
draws=0
losses=0
```

Stockfish smoke at 16 simulations, material weight `0.15`:

```text
games=2
score=0.0
wins=0
draws=0
losses=2
pgn=reports/focus_material015_strict_vs_stockfish_16sims.pgn
```

Stockfish smoke at 64 simulations, material weight `0.15`:

```text
games=1
score=0.0
wins=0
draws=0
losses=1
pgn=reports/focus_material015_strict_s64_vs_stockfish.pgn
```

## Conclusion

The material-prior iteration promoted cleanly over the forced-mate strict base
and improved the fixed Stockfish MultiPV diagnostic from `0.5728` to `0.5762`.
It did not improve the Stockfish smoke gate: the candidate lost all sampled
Stockfish games at both 16 and 64 simulations. The next useful branch is to
test stronger material weights at inference or add broader tactical/value
supervision before spending more GPU on the same recipe.
