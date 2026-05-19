# Latest-Loss Replay Repair Probe

Date: 2026-05-19

## Data

Generated a local Stockfish hard-label replay slice from recent direct
Stockfish losses:

`data/teacher/alpha_loss_leafmcts_v1`

Config:

- PGNs: hard-label gate, value-head gate, leaf-material probes, qvalue
  leaf-material probes, and policyhead-broad 64-sim loss
- `player_name=AlphaChess`
- `min_value_delta=0.08`
- `position_stride=1`
- `min_ply=6`
- `max_ply=90`
- `pv_plies=2`
- `game_line_plies=4`
- `engine_time=0.05`

Summary:

```text
games_seen=20
games_used=20
positions=574
bad_action_positions=82
files=8
```

## Run

`experiments/policyhead-leafloss-qvalue-vw000-material015`

Base checkpoint:

`experiments/policyhead-hardlabels-qvalue-vw000-material015/checkpoints/iter_0001/latest.pt`

Config:

- `policy_head_only=true`
- `prefer_action_labels=true`
- `value_weight=0.0`
- `lr=5e-6`
- `epochs=3`
- `games=0`
- `legal_policy_loss=true`
- `material_value_weight=0.15`
- `material_value_search_plies=2`
- replay data:
  - `data/teacher/stockfish_multipv_elo1800_4096_t005`
  - `data/puzzles/lines_1200_2400_100k`
  - `data/teacher/alpha_loss_reports_v4`
  - `data/teacher/alpha_loss_pvlines_v1`
  - `data/teacher/alpha_poisoned_captures_v2`
  - `data/teacher/alpha_loss_leafmcts_v1`
- replay weights: `0.50 0.15 0.08 0.08 0.02 0.17`

GPU runner:

- `gpu-dev` A100 reservation `30d330bf`
- completed and copied back normally

Candidate:

`experiments/policyhead-leafloss-qvalue-vw000-material015/checkpoints/iter_0001/latest.pt`

Training checkpoint metrics:

```text
epoch_loss=1.8778
val_loss=1.9969
val_policy_loss=1.9969
val_policy_acc=0.4570
val_source_5_policy_acc=0.2778
```

## Promotion

Parent match against the hard-label policy-head checkpoint:

- score: `2.0/8`
- wins/draws/losses: `0/4/4`
- promoted: `false`

The iteration driver skipped the Stockfish gate because the candidate failed
the parent match.

Manual direct Stockfish check:

- score: `0.0/2`
- wins/draws/losses: `0/0/2`
- PGN: `reports/policyhead_leafloss_qvalue_stockfish_16sims.pgn`

## Conclusion

Rejected. Injecting the latest direct-loss positions as hard replay data did
not improve the model. The candidate underperformed its parent and still lost
the direct Stockfish check. The new replay slice is probably too narrow and too
late-tactical to repair the broader king-safety failure by policy-head tuning
alone.
