# Conservative PV-Line Qvalue Fine-tune

Date: 2026-05-18

## Run

```text
experiments/focus-pvlines10-qvalue-vw025-material015
```

Started from:

```text
experiments/focus-qvalue-puzzlelines20-vw025-material015/checkpoints/iter_0001/latest.pt
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
replay_weights=0.55,0.20,0.10,0.05,0.10
```

This branch reduced the PV-line replay weight from `0.15` to `0.10` and lowered
the learning rate to `0.00001`, aiming to keep more broad Stockfish-teacher
accuracy than `focus-pvlines-qvalue-vw025-material015`.

## Promotion

Promotion result against qvalue:

```text
score=8.0/8
wins=8
draws=0
losses=0
promoted=true
```

Checkpoint:

```text
experiments/focus-pvlines10-qvalue-vw025-material015/checkpoints/iter_0001/latest.pt
```

## Fixed Validation

```text
stockfish_multipv_elo1800_4096 policy_acc=0.5864
all_1200_2400_50k policy_acc=0.3314
alpha_loss_reports_v2 policy_acc=0.5902
puzzle lines policy_acc=0.4531
poisoned_captures_v2 policy_acc=0.6786
alpha_loss_pvlines_v1 policy_acc=0.2480
```

The internal promotion result was strong, but broad Stockfish accuracy still
regressed from qvalue's `0.6050`; poisoned and PV-line validation were worse
than the heavier PV-line branch.

## Direct Play

```text
stockfish 16 sims: 0.0/2
PGN=reports/focus_pvlines10_qvalue_vs_stockfish_16sims.pgn

stockfish 64 sims: 0.0/1
PGN=reports/focus_pvlines10_qvalue_s64_vs_stockfish.pgn
```

## Conclusion

This conservative PV-line blend is rejected as a direct-play candidate. It
beats qvalue in the internal gate, but it fails direct Stockfish and does not
retain the broader fixed validators. The heavier PV-line branch remains the
more interesting tactical checkpoint because it at least produced one
64-simulation Stockfish draw.
