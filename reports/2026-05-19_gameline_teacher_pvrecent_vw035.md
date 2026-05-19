# Game-Line Teacher PV-Recent Probe

Date: 2026-05-19

## Training run

Run:

`experiments/focus-gamelines-pvrecent-vw035-material015`

Base checkpoint:

`experiments/focus-pvlinesrecent-qvalue-vw025-material015/checkpoints/iter_0001/latest.pt`

This was a conservative follow-up to the qvalue-start game-line run: lower LR, lower game-line replay weight, and lower value weight, while keeping broad Stockfish replay dominant.

Config:

- `games=0`
- `epochs=1`
- `lr=8e-6`
- `value_weight=0.35`
- `legal_policy_loss=true`
- `material_value_weight=0.15`
- `material_value_search_plies=2`
- replay data:
  - `data/teacher/stockfish_multipv_elo1800_4096`
  - `data/puzzles/lines_1200_2400_100k`
  - `data/teacher/alpha_loss_reports_v3`
  - `data/teacher/alpha_poisoned_captures_v2`
  - `data/teacher/alpha_loss_gamelines_recent_v1`
- replay weights: `0.50 0.16 0.08 0.04 0.22`

Candidate checkpoint:

`experiments/focus-gamelines-pvrecent-vw035-material015/checkpoints/iter_0001/latest.pt`

## Promotion gate

Opponent:

`experiments/focus-pvlinesrecent-qvalue-vw025-material015/checkpoints/iter_0001/latest.pt`

Result:

- 48 simulations, 8 games
- score: `2.0/8`
- wins/draws/losses: `0/4/4`
- rejected internally

## Fixed validation

Legal-policy validation:

| Dataset | Policy Acc |
| --- | ---: |
| `stockfish_multipv_elo1800_4096` | `0.5857` |
| `puzzles/all_1200_2400_50k` | `0.3323` |
| `alpha_loss_reports_v2` | `0.6066` |
| `puzzles/lines_1200_2400_100k` | `0.4536` |
| `alpha_poisoned_captures_v2` | `0.7500` |
| `alpha_loss_pvlines_recent_v1` | `0.2786` |
| `alpha_loss_gamelines_recent_v1` | `0.3068` |

This branch preserved the poisoned-capture and puzzle-line diagnostics better than the qvalue-start game-line branch, but it still underperformed the PV-recent parent in the promotion match and did not materially improve the new game-line replay diagnostic.

## Conclusion

Rejected without direct Stockfish smoke. The branch failed the stronger PV-recent parent gate (`2.0/8`), so it is not a better candidate for direct-play testing.
