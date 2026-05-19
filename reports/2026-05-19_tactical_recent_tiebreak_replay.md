# Tactical Recent Tiebreak Replay Probe

Date: 2026-05-19

## Data

Generated ignored local replay data:

`data/teacher/alpha_loss_tactical_recent_tiebreak_v1`

Summary:

- sources: latest opening MultiPV12, patched tie-break, and root-material loss PGNs
- games_seen: `11`
- games_used: `11`
- positions: `1437`
- files: `15`
- valid bad actions: `107`
- average nonzero policy moves: approximately `7.64`
- value range: `[-1.0, 1.0]`
- mean value: approximately `-0.0259`
- `engine_time=0.05`
- `min_value_delta=0.03`
- `player_name=AlphaChess`
- `multipv=8`
- `policy_temperature_cp=180`
- `position_stride=1`
- `pv_plies=6`
- `game_line_plies=6`

This was a focused DAgger-style source from the exact failures remaining after
the deterministic MCTS tie-break fix.

## Run

`experiments/focus-tacticalrecent-tiebreak-pvrecent-vw035-material015`

Base checkpoint:

`experiments/focus-pvlinesrecent-qvalue-vw025-material015/checkpoints/iter_0001/latest.pt`

Config:

- `iterations=2`
- `games=0`
- `epochs=1`
- `lr=1e-6`
- `value_weight=0.35`
- `legal_policy_loss=true`
- `material_value_weight=0.15`
- `material_value_search_plies=2`
- `eval_games=8`
- `eval_simulations=48`
- `stockfish_gate_games=2`
- `stockfish_gate_simulations=16`
- replay data:
  - `data/teacher/stockfish_multipv_elo1800_4096`
  - `data/puzzles/lines_1200_2400_100k`
  - `data/teacher/alpha_loss_tactical_recent_tiebreak_v1`
  - `data/teacher/alpha_poisoned_captures_v2`
  - `data/teacher/alpha_loss_pvlines_recent_v1`
  - `data/teacher/alpha_loss_gamelines_all_v2`
- replay weights: `0.52 0.16 0.10 0.04 0.10 0.08`

## Result

Both seeds failed the parent match, so the Stockfish gate did not run:

| Iteration | Parent Match | W/D/L | Stockfish Gate |
| --- | ---: | ---: | ---: |
| `1` | `2.0/8` | `0/4/4` | not run |
| `2` | `0.0/8` | `0/0/8` | not run |

Validation on `alpha_loss_tactical_recent_tiebreak_v1`:

| Checkpoint | Policy Acc | Policy Loss | Value Loss | Loss |
| --- | ---: | ---: | ---: | ---: |
| PV-recent parent | `0.2693` | `3.6071` | `0.1379` | `3.6554` |
| Iteration 1 | `0.2721` | `3.5216` | `0.1280` | `3.5664` |
| Iteration 2 | `0.2749` | `3.5315` | `0.1280` | `3.5763` |

## Conclusion

Rejected. The exact recent-failure replay source is learnable, but only
marginally at this weight and learning rate, and it severely damages the
PV-recent parent match. This is another sign that direct-play weakness is not
being solved by small supervised repairs on AlphaChess loss positions alone.
