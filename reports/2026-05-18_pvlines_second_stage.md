# Second-Stage PV-Line Fine-tune

Date: 2026-05-18

## Run

```text
experiments/focus-pvlines2-pvlines-qvalue-vw025-material015
```

Started from:

```text
experiments/focus-pvlines-qvalue-vw025-material015/checkpoints/iter_0001/latest.pt
```

Config highlights:

```text
games=0
epochs=1
lr=0.00001
value_weight=0.25
material_value_weight=0.15
material_value_search_plies=2
self_play_weight=0.0
replay_data=stockfish_multipv_elo1800_4096, puzzle lines, alpha_loss_reports_v3, alpha_poisoned_captures_v2, alpha_loss_pvlines_v1
replay_weights=0.40,0.15,0.05,0.10,0.30
```

This tested whether the heavier PV-line branch could be improved by pushing the
new PV-line replay harder in a second replay-only fine-tune.

## Promotion

Promotion result against the first PV-line branch:

```text
score=0.0/8
wins=0
draws=0
losses=8
promoted=false
```

Checkpoint:

```text
experiments/focus-pvlines2-pvlines-qvalue-vw025-material015/checkpoints/iter_0001/latest.pt
```

## Conclusion

This branch is rejected. Increasing PV-line replay to `0.30` from the already
PV-tuned checkpoint collapses head-to-head strength against the first PV-line
branch. The useful signal is not "more PV replay"; future branches should either
improve the PV-line data quality/coverage or blend tactical supervision without
overwriting the broader policy.
