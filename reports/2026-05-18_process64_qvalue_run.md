# Process64 Qvalue Run

Date: 2026-05-18

## Worker change

This run used the process-based self-play worker path from commit `a3b986e`.
Remote monitoring during self-play showed eight spawned Python workers at about
7.9 combined CPU cores and L4 utilization around 58-89%, compared with roughly
1.4 CPU cores in the earlier thread-based parallel run.

## Run

```text
experiments/focus-process64-qvalue-vw025-material015
```

Started from:

```text
experiments/focus-qvalue-puzzlelines20-vw025-material015/checkpoints/iter_0001/latest.pt
```

Config highlights:

```text
games=64
self_play_workers=8
simulations=48
c_puct=1.5
max_plies=180
epochs=1
lr=0.00002
value_weight=0.25
material_value_weight=0.15
material_value_search_plies=2
self_play_weight=0.15
replay_data=stockfish_multipv_elo1800_4096, puzzle lines, alpha_loss_reports_v3, alpha_poisoned_captures_v2
replay_weights=0.55,0.20,0.10,0.05
```

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
experiments/focus-process64-qvalue-vw025-material015/checkpoints/iter_0001/latest.pt
```

## Fixed Validation

```text
stockfish_multipv_elo1800_4096 policy_acc=0.6062
all_1200_2400_50k policy_acc=0.3386
alpha_loss_reports_v2 policy_acc=0.6066
puzzle lines policy_acc=0.4585
poisoned_captures_v2 policy_acc=0.7143
```

## Direct Play

```text
stockfish 16 sims: 0.0/2
PGN=reports/focus_process64_qvalue_vs_stockfish_16sims.pgn
```

Both losses were tactical king-safety failures. The first ended after
`18. Rxc8 Raxc8 19. Qxc8 Rxc8`, and the second allowed the forcing sequence
after `9...Qe7 10. Qxc5`.

## Conclusion

Process workers fixed the self-play scaling bottleneck, but the larger
process64 run did not improve head-to-head strength. It slightly improved fixed
Stockfish and puzzle-line diagnostics over qvalue while keeping poisoned
captures high, but it failed promotion and still scored zero in direct
Stockfish play. The current blocker is move quality/tactical robustness, not
worker throughput.
