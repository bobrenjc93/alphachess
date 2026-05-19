# Guarded Self-play Run

Date: 2026-05-18

## Run

```text
experiments/focus-rootguard150-poisonedv2-vw025-material015
```

Started from:

```text
experiments/focus-poisonedv2-qvalue-vw025-material015/checkpoints/iter_0001/latest.pt
```

Config highlights:

```text
games=16
simulations=48
epochs=1
lr=0.00002
value_weight=0.25
material_value_weight=0.15
material_value_search_plies=2
root_material_search_plies=2
root_material_max_loss_cp=150
self_play_weight=0.10
replay_data=stockfish_multipv_elo1800_4096, puzzle lines, alpha_loss_reports_v3, alpha_poisoned_captures_v2
replay_weights=0.55,0.20,0.10,0.15
```

This run used the fixed root material fallback from
`reports/2026-05-18_root_material_fallback.md`.

## Promotion

Promotion result against the poisoned-v2 checkpoint under the guarded eval
settings:

```text
score=8.0/8
wins=8
draws=0
losses=0
promoted=true
```

Checkpoint:

```text
experiments/focus-rootguard150-poisonedv2-vw025-material015/checkpoints/iter_0001/latest.pt
```

Training checkpoint metrics included:

```text
val_policy_acc=0.4635
val_source_0_policy_acc=0.6828
val_source_1_policy_acc=0.5603
val_source_2_policy_acc=0.4539
val_source_3_policy_acc=0.8000
val_source_4_policy_acc=1.0000
```

## Fixed Validation

```text
stockfish_multipv_elo1800_4096 policy_acc=0.5562
all_1200_2400_50k policy_acc=0.3346
alpha_loss_reports_v2 policy_acc=0.5738
puzzle lines policy_acc=0.4588
poisoned_captures_v2 policy_acc=0.6786
```

The run improved puzzle-line accuracy and kept the poisoned-capture slice high,
but broad Stockfish teacher accuracy regressed substantially.

## Direct Play

Settings:

```text
material_value_weight=0.15
material_value_search_plies=2
root_material_search_plies=2
root_material_max_loss_cp=150
```

Results:

```text
uniform 32 sims: 3.0/4
stockfish 16 sims: 0.0/2
PGN: reports/focus_rootguard150_poisonedv2_vs_stockfish_16sims.pgn
```

The fixed root material fallback removed the known `Qxc7` queen-loss pattern,
but the model still found other losing tactical paths. This is not a direct
Stockfish breakthrough.
