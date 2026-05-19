# PV-Line Qvalue Fine-tune

Date: 2026-05-18

## Generator Change

Commit `157c8f7` added `--pv-plies` to `alpha-chess stockfish-teacher`.
For each selected source position, the generator can now add re-analysed
Stockfish principal-variation continuation positions instead of only the first
best-move label.

Tests:

```text
uv run pytest tests/test_stockfish_teacher.py  # 6 passed
uv run pytest                                 # 49 passed
```

A real smoke with Stockfish and `--pv-plies 2` generated 6 positions.

## Data

Generated ignored local replay data:

```text
out=data/teacher/alpha_loss_pvlines_v1
sources=61 Stockfish PGNs from reports/
games_seen=31
games_used=31
positions=512
files=18
engine_time=0.02
min_value_delta=0.10
player_name=AlphaChess
multipv=4
position_stride=1
pv_plies=4
```

## GPU Iteration

Run:

```text
experiments/focus-pvlines-qvalue-vw025-material015
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
replay_data=stockfish_multipv_elo1800_4096, puzzle lines, alpha_loss_reports_v3, alpha_poisoned_captures_v2, alpha_loss_pvlines_v1
replay_weights=0.50,0.20,0.10,0.05,0.15
```

## Promotion

Promotion result against qvalue:

```text
score=6.0/8
wins=4
draws=4
losses=0
promoted=true
```

Checkpoint:

```text
experiments/focus-pvlines-qvalue-vw025-material015/checkpoints/iter_0001/latest.pt
```

## Fixed Validation

```text
stockfish_multipv_elo1800_4096 policy_acc=0.5852
all_1200_2400_50k policy_acc=0.3324
alpha_loss_reports_v2 policy_acc=0.5574
puzzle lines policy_acc=0.4548
poisoned_captures_v2 policy_acc=0.7500
alpha_loss_pvlines_v1 policy_acc=0.3301
```

The branch promoted internally and improved the poisoned slice, but broad
Stockfish-teacher accuracy regressed from qvalue's `0.6050`.

## Direct Play

```text
stockfish 16 sims: 0.0/2
PGN=reports/focus_pvlines_qvalue_vs_stockfish_16sims.pgn

stockfish 64 sims, seed 9064: 0.5/1
PGN=reports/focus_pvlines_qvalue_s64_vs_stockfish.pgn

stockfish 64 sims, seed 9164: 0.0/2
PGN=reports/focus_pvlines_qvalue_s64_2g_vs_stockfish.pgn
```

The single 64-simulation draw is the first recent direct Stockfish non-loss,
but the two-game 64-simulation follow-up still lost both games.

## Conclusion

PV-line replay is useful enough to beat qvalue in the internal gate and can
occasionally hold a direct Stockfish draw with more search. It is not yet a
reliable direct-play breakthrough, and the broad policy regression means the
next branch should blend PV-line replay more conservatively or improve tactical
search without giving up general Stockfish-teacher accuracy.
