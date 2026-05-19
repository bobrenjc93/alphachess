# Puzzle-Heavy Material-Prior Replay Iteration

Date: 2026-05-18

## GPU Iteration

- Base checkpoint: `experiments/focus-material015-strict/checkpoints/iter_0001/latest.pt`
- Candidate checkpoint: `experiments/focus-puzzle30-material015/checkpoints/iter_0001/latest.pt`
- Runner: `gpu-dev submit`, reservation `c34fc6b5`, 1x L4
- Self-play: 16 games, 48 simulations, 180 max plies
- Evaluation blend: `--material-value-weight 0.15`
- Root material guard: disabled (`root_material_search_plies=0`)
- Training: 1 epoch, batch size 128, learning rate `0.00004`
- Loss: `--legal-policy-loss`
- Promotion threshold: `0.55`
- Data weights:
  - accumulated self-play total weight `0.10`
  - `data/teacher/stockfish_multipv_elo1800_4096`, weight `0.45`
  - `data/puzzles/all_1200_2400_50k`, weight `0.30`
  - `data/teacher/alpha_loss_reports_v3`, weight `0.15`

Promotion gate against the base checkpoint:

```text
score=2.0/8
wins=0
draws=4
losses=4
score_rate=0.25
promoted=false
```

The best checkpoint remains
`experiments/focus-material015-strict/checkpoints/iter_0001/latest.pt`.

Final candidate training metrics:

```text
loss=1.9693
policy_loss=1.8947
policy_acc=0.4717
value_loss=0.0746
epoch_loss=2.1588
val_loss=2.5216
val_policy_loss=2.4061
val_policy_acc=0.3711
val_value_loss=0.1155
```

## Diagnostic Validation

Standalone validation on the fixed replay datasets:

```text
val_policy_acc=0.3673
val_source_0_policy_acc=0.5591  # Stockfish MultiPV 4096
val_source_1_policy_acc=0.3513  # all puzzles 50k
val_source_2_policy_acc=0.5738  # AlphaChess loss replay v2
```

Puzzle accuracy improved from `0.3437` on the current best checkpoint to
`0.3513`, but the fixed Stockfish MultiPV diagnostic regressed from `0.5762`
to `0.5591` and the league result was poor.

## Conclusion

This branch is rejected. A heavier puzzle replay mix improves puzzle-label fit
but hurts internal strength and Stockfish policy alignment. Puzzle data is
useful as a small auxiliary source, but the current importer and weighting do
not provide enough tactical transfer to justify `0.30` sampler weight.
