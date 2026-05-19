# Parallel32 Qvalue Run

Date: 2026-05-18

## Run

```text
experiments/focus-parallel32-qvalue-vw025-material015
```

Started from:

```text
experiments/focus-qvalue-puzzlelines20-vw025-material015/checkpoints/iter_0001/latest.pt
```

Config highlights:

```text
games=32
self_play_workers=4
simulations=48
c_puct=1.5
max_plies=180
epochs=1
lr=0.00002
value_weight=0.25
material_value_weight=0.15
material_value_search_plies=2
root_mate_search_plies=3
self_play_weight=0.15
replay_data=stockfish_multipv_elo1800_4096, puzzle lines, alpha_loss_reports_v3, alpha_poisoned_captures_v2
replay_weights=0.55,0.20,0.10,0.05
```

The run used the thread-based self-play workers, but remote CPU usage only
reached roughly 1.4 cores. The worker path is functional but not enough for real
self-play scaling; process-based workers are still needed.

## Promotion

Promotion result against qvalue:

```text
score=2.0/8
wins=0
draws=4
losses=4
promoted=false
```

Checkpoint:

```text
experiments/focus-parallel32-qvalue-vw025-material015/checkpoints/iter_0001/latest.pt
```

Training checkpoint metrics included:

```text
val_policy_acc=0.4737
val_source_0_policy_acc=0.7048
val_source_1_policy_acc=0.5827
val_source_2_policy_acc=0.4616
val_source_3_policy_acc=0.7143
```

## Fixed Validation

```text
stockfish_multipv_elo1800_4096 policy_acc=0.6052
all_1200_2400_50k policy_acc=0.3378
alpha_loss_reports_v2 policy_acc=0.6885
puzzle lines policy_acc=0.4588
poisoned_captures_v2 policy_acc=0.7500
```

## Direct Play

```text
stockfish 16 sims: 0.0/2
PGN=reports/focus_parallel32_qvalue_vs_stockfish_16sims.pgn
```

## Conclusion

The run improved several fixed validators and matched the qvalue checkpoint on
the broad Stockfish teacher set, but it failed promotion and still lost all
direct Stockfish games. It is a useful rejected candidate and evidence that the
current thread-based self-play workers do not provide enough scale.
