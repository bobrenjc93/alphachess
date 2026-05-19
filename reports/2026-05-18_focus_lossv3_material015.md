# Loss-v3 Material-Prior Replay Iteration

Date: 2026-05-18

## GPU Iteration

- Base checkpoint: `experiments/focus-material015-strict/checkpoints/iter_0001/latest.pt`
- Candidate checkpoint: `experiments/focus-lossv3-material015/checkpoints/iter_0001/latest.pt`
- Runner: `gpu-dev submit`, reservation `0ba14b12`, 1x L4
- Self-play: 16 games, 48 simulations, 180 max plies
- Evaluation blend: `--material-value-weight 0.15`
- Root material guard: disabled (`root_material_search_plies=0`)
- Training: 1 epoch, batch size 128, learning rate `0.00005`
- Loss: `--legal-policy-loss`
- Promotion threshold: `0.55`
- Data weights:
  - accumulated self-play total weight `0.10`
  - `data/teacher/stockfish_multipv_elo1800_4096`, weight `0.50`
  - `data/puzzles/all_1200_2400_50k`, weight `0.10`
  - `data/teacher/alpha_loss_reports_v3`, weight `0.30`

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
loss=1.7615
policy_loss=1.7337
policy_acc=0.5094
value_loss=0.0278
epoch_loss=1.9305
val_loss=2.6522
val_policy_loss=2.5083
val_policy_acc=0.3573
val_value_loss=0.1438
```

## Diagnostic Validation

Standalone validation on the fixed replay datasets:

```text
val_policy_acc=0.3542
val_source_0_policy_acc=0.5093  # Stockfish MultiPV 4096
val_source_1_policy_acc=0.3411  # all puzzles 50k
val_source_2_policy_acc=0.6066  # AlphaChess loss replay v2
```

The fixed Stockfish MultiPV diagnostic regressed from `0.5762` on the current
best checkpoint to `0.5093`.

## Conclusion

This branch is rejected. Increasing the refreshed loss replay weight to `0.30`
hurt both the internal league and the fixed Stockfish diagnostic, despite
improving fit to some sampled training-split sources. The loss replay should
stay as a smaller corrective source unless it is expanded with more diverse,
higher-quality tactical labels.
