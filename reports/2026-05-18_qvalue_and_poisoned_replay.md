# Qvalue Material Run and Poisoned Replay Follow-up

Date: 2026-05-18

## Qvalue run

Run:

```text
experiments/focus-qvalue-puzzlelines20-vw025-material015
```

Started from:

```text
experiments/focus-puzzlelines20-vw025-material015/checkpoints/iter_0001/latest.pt
```

Config highlights:

```text
games=16
simulations=48
epochs=1
lr=0.00003
value_weight=0.25
material_value_weight=0.15
material_value_search_plies=2
self_play_weight=0.10
replay_data=stockfish_multipv_elo1800_4096, puzzle lines, alpha_loss_reports_v3
replay_weights=0.55,0.20,0.15
```

Promotion result:

```text
score=8.0/8
wins=8
draws=0
losses=0
promoted=true
```

Checkpoint:

```text
experiments/focus-qvalue-puzzlelines20-vw025-material015/checkpoints/iter_0001/latest.pt
```

Training checkpoint metrics included:

```text
val_policy_acc=0.4556
val_source_0_policy_acc=0.6339
val_source_1_policy_acc=0.5421
val_source_2_policy_acc=0.4485
val_source_3_policy_acc=0.5556
```

Fixed validation:

```text
stockfish_multipv_elo1800_4096 policy_acc=0.6050
all_1200_2400_50k policy_acc=0.3365
alpha_loss_reports_v2 policy_acc=0.5738
puzzle lines policy_acc=0.4539
poisoned_captures_v2 policy_acc=0.0357
```

Direct play:

```text
uniform 32 sims: 4.0/4
stockfish 16 sims: 0.0/2
stockfish 64 sims: 0.0/1
root material guard 16 sims: 0.0/2
```

PGNs:

```text
reports/focus_qvalue_puzzlelines20_vw025_material015_vs_stockfish_16sims.pgn
reports/focus_qvalue_puzzlelines20_vw025_material015_s64_vs_stockfish.pgn
reports/focus_qvalue_rootguard150_vs_stockfish_16sims.pgn
```

The qvalue run is the strongest broad fixed-validator checkpoint so far, but it
still fails direct Stockfish. The losses remain tactical: poisoned recaptures,
loose king positions, and sacrifice/recapture sequences.

## Poisoned replay data

Generated two focused replay sets from the newest failed Stockfish PGNs. The
useful one is:

```text
data/teacher/alpha_poisoned_captures_v2
```

Generation highlights:

```text
engine_time=0.05
min_value_delta=0.10
player_name=AlphaChess
multipv=4
position_stride=1
positions=28
```

Sources:

```text
reports/focus_puzzlelines20_vw025_material015_s128_vs_stockfish.pgn
reports/focus_puzzlelines20_vw025_material015_s64_vs_stockfish.pgn
reports/focus_puzzlelines20_vw025_material015_vs_stockfish_16sims.pgn
reports/material_qsearch2_puzzlelines20_s64_vs_stockfish.pgn
reports/material_qsearch2_puzzlelines20_vs_stockfish_16sims.pgn
reports/material_qsearch2_w025_puzzlelines20_vs_stockfish_16sims.pgn
```

Before fine-tuning, both the previous best and the qvalue checkpoint were only
`1/28` on this slice.

## Poisoned replay fine-tune

Run:

```text
experiments/focus-poisonedv2-qvalue-vw025-material015
```

Started from:

```text
experiments/focus-qvalue-puzzlelines20-vw025-material015/checkpoints/iter_0001/latest.pt
```

Config highlights:

```text
games=0
epochs=1
lr=0.000015
value_weight=0.25
material_value_weight=0.15
material_value_search_plies=2
self_play_weight=0.0
replay_data=stockfish_multipv_elo1800_4096, puzzle lines, alpha_loss_reports_v3, alpha_poisoned_captures_v2
replay_weights=0.50,0.20,0.10,0.20
```

Promotion result versus qvalue:

```text
score=6.0/8
wins=4
draws=4
losses=0
promoted=true
```

Fixed validation:

```text
stockfish_multipv_elo1800_4096 policy_acc=0.5876
all_1200_2400_50k policy_acc=0.3324
alpha_loss_reports_v2 policy_acc=0.6721
puzzle lines policy_acc=0.4551
poisoned_captures_v2 policy_acc=0.7143
```

Direct play:

```text
uniform 32 sims: 4.0/4
stockfish 16 sims: 0.0/2
```

PGN:

```text
reports/focus_poisonedv2_qvalue_vs_stockfish_16sims.pgn
```

## Conclusion

The qvalue checkpoint is a real internal improvement and the best broad
validator checkpoint so far. The poisoned replay fine-tune fixed the targeted
blind spot and beat qvalue head-to-head, but it reduced broad Stockfish teacher
accuracy and still did not score against Stockfish directly.

Neither checkpoint is superhuman. The next useful direction is to keep the
poisoned slice as a diagnostic/training source, but address the broader tactical
failure mode rather than repeatedly overfitting the latest loss PGNs.
