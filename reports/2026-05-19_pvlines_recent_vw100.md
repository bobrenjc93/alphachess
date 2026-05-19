# Recent PV-Line Value-Weight Probe

Date: 2026-05-19

## Motivation

Root diagnostics on a direct Stockfish loss showed that low-prior engine-best
moves can be visited when policy priors are flattened, but the value head still
mis-ranks the tactical continuation. This branch tested whether higher value
loss weight improves that search signal.

## Run

```text
experiments/focus-pvlinesrecent-vw100-qvalue-material015
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
value_weight=1.0
material_value_weight=0.15
material_value_search_plies=2
self_play_weight=0.0
replay_data=stockfish_multipv_elo1800_4096, puzzle lines, alpha_loss_reports_v3, alpha_poisoned_captures_v2, alpha_loss_pvlines_recent_v1
replay_weights=0.50,0.20,0.10,0.05,0.15
```

## Promotion

Promotion result against qvalue:

```text
score=4.0/8
wins=0
draws=8
losses=0
promoted=true at threshold 0.50
```

Checkpoint:

```text
experiments/focus-pvlinesrecent-vw100-qvalue-material015/checkpoints/iter_0001/latest.pt
```

## Fixed Validation

```text
stockfish_multipv_elo1800_4096 policy_acc=0.5842
all_1200_2400_50k policy_acc=0.3327
alpha_loss_reports_v2 policy_acc=0.6557
puzzle lines policy_acc=0.4522
poisoned_captures_v2 policy_acc=0.5714
alpha_loss_pvlines_recent_v1 policy_acc=0.2331
```

The higher value weight improves old loss-v2 accuracy but hurts poisoned and
recent PV-line policy accuracy.

## Direct Play

```text
stockfish 16 sims: 0.0/2
PGN=reports/focus_pvlinesrecent_vw100_qvalue_vs_stockfish_16sims.pgn

stockfish 64 sims: 0.0/1
PGN=reports/focus_pvlinesrecent_vw100_qvalue_s64_vs_stockfish.pgn
```

## Conclusion

This value-weight probe is rejected as a direct-play improvement. The diagnostic
is still useful: tactical search is value-sensitive, but simply increasing
training value weight damages policy strength and does not produce a reliable
Stockfish result.
