# Local Check-Aware Parallel Iteration

Date: 2026-05-18

## Context

`gpu-dev` remained unavailable, so this was a CPU-only scaling smoke for the
new self-play worker path plus check-aware quiescence.

Run:

```text
experiments/focus-local-checkq-parallel8-vw025-material015
```

Started from:

```text
experiments/focus-qvalue-puzzlelines20-vw025-material015/checkpoints/iter_0001/latest.pt
```

Config highlights:

```text
device=cpu
games=8
self_play_workers=4
simulations=16
max_plies=140
eval_games=4
eval_simulations=16
epochs=1
lr=0.00002
value_weight=0.25
material_value_weight=0.15
material_value_search_plies=2
self_play_weight=0.15
replay_data=stockfish_multipv_elo1800_4096, puzzle lines, alpha_loss_reports_v3, alpha_poisoned_captures_v2
replay_weights=0.55,0.20,0.10,0.05
```

## Result

The local candidate promoted over the qvalue checkpoint in the small internal
match:

```text
score=4.0/4
wins=4
draws=0
losses=0
promoted=true
```

Checkpoint:

```text
experiments/focus-local-checkq-parallel8-vw025-material015/checkpoints/iter_0001/latest.pt
```

Training checkpoint metrics included:

```text
val_policy_acc=0.4659
val_source_0_policy_acc=0.8298
val_source_1_policy_acc=0.5450
val_source_2_policy_acc=0.4594
val_source_3_policy_acc=0.4286
val_source_4_policy_acc=0.0000
```

Fixed validation:

```text
stockfish_multipv_elo1800_4096 policy_acc=0.5505
all_1200_2400_50k policy_acc=0.3354
alpha_loss_reports_v2 policy_acc=0.5410
puzzle lines policy_acc=0.4567
poisoned_captures_v2 policy_acc=0.7143
```

Direct play:

```text
stockfish 16 sims: 0.0/2
PGN=reports/focus_local_checkq_parallel8_vs_stockfish_16sims.pgn
```

## Conclusion

The worker path functions locally and the small CPU run promoted internally, but
the resulting checkpoint regressed broad Stockfish teacher accuracy and still
failed direct Stockfish. This is not a new best checkpoint.
