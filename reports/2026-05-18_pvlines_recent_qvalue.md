# Recent PV-Line Qvalue Fine-tune

Date: 2026-05-18

## Data

Generated ignored local replay data from recent stronger-branch losses only:

```text
out=data/teacher/alpha_loss_pvlines_recent_v1
sources=19 qvalue/process/PV-line Stockfish PGNs
games_seen=35
games_used=35
positions=768
files=18
engine_time=0.03
min_value_delta=0.08
player_name=AlphaChess
multipv=4
position_stride=1
pv_plies=4
```

This was intended to avoid the historical PV-line dataset's alphabetical source
mix, which hit its cap after many older weaker-model PGNs.

## GPU Iteration

Run:

```text
experiments/focus-pvlinesrecent-qvalue-vw025-material015
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
replay_data=stockfish_multipv_elo1800_4096, puzzle lines, alpha_loss_reports_v3, alpha_poisoned_captures_v2, alpha_loss_pvlines_recent_v1
replay_weights=0.50,0.20,0.10,0.05,0.15
```

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
experiments/focus-pvlinesrecent-qvalue-vw025-material015/checkpoints/iter_0001/latest.pt
```

## Fixed Validation

```text
stockfish_multipv_elo1800_4096 policy_acc=0.5864
all_1200_2400_50k policy_acc=0.3323
alpha_loss_reports_v2 policy_acc=0.6393
puzzle lines policy_acc=0.4539
poisoned_captures_v2 policy_acc=0.7143
alpha_loss_pvlines_recent_v1 policy_acc=0.2604
```

The curated data improves the old loss-v2 diagnostic, but it still regresses
broad Stockfish accuracy relative to qvalue's `0.6050`.

## Direct Play

```text
stockfish 16 sims: 0.0/2
PGN=reports/focus_pvlinesrecent_qvalue_vs_stockfish_16sims.pgn

stockfish 64 sims, seed 9464: 0.5/1
PGN=reports/focus_pvlinesrecent_qvalue_s64_vs_stockfish.pgn

stockfish 64 sims, seed 9564: 0.0/2
PGN=reports/focus_pvlinesrecent_qvalue_s64_2g_vs_stockfish.pgn
```

## Conclusion

The curated recent PV-line branch is internally strong and can also find an
occasional 64-simulation Stockfish draw, but the signal is not stable across
seeds. Direct play still fails at 16 simulations and loses the 2-game 64-sim
follow-up. Like the first PV-line branch, this is tactical progress, not a
superhuman checkpoint.
