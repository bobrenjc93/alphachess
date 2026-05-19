# Conservative Broad16k PV-Recent Probe

Date: 2026-05-19

## Run

`experiments/focus-broad16k-pvrecent-low-vw025-material015`

Base checkpoint:

`experiments/focus-pvlinesrecent-qvalue-vw025-material015/checkpoints/iter_0001/latest.pt`

This probe tested whether a low-LR PV-recent fine-tune on the broader 16k
Stockfish MultiPV teacher could recover broad policy quality without the
targeted-replay regressions.

Config:

- `games=0`
- `epochs=1`
- `lr=2e-6`
- `value_weight=0.25`
- `legal_policy_loss=true`
- `material_value_weight=0.15`
- `material_value_search_plies=2`
- replay data:
  - `data/teacher/stockfish_multipv_elo1800_16384`
  - `data/teacher/stockfish_multipv_elo1800_4096`
  - `data/puzzles/lines_1200_2400_100k`
  - `data/teacher/alpha_poisoned_captures_v2`
  - `data/teacher/alpha_loss_pvlines_recent_v1`
  - `data/teacher/alpha_loss_gamelines_all_v2`
- replay weights: `0.60 0.12 0.12 0.04 0.08 0.04`

## Promotion

Candidate:

`experiments/focus-broad16k-pvrecent-low-vw025-material015/checkpoints/iter_0001/latest.pt`

Parent match:

- score: `0.0/8`
- wins/draws/losses: `0/0/8`
- promoted: `false`

The Stockfish gate did not run because the candidate failed the parent match.

## Fixed Validation

Legal-policy validation against the PV-recent parent:

| Dataset | Parent Acc | Candidate Acc |
| --- | ---: | ---: |
| `stockfish_multipv_elo1800_16384` | `0.3733` | `0.3649` |
| `stockfish_multipv_elo1800_4096` | `0.5864` | `0.5193` |
| `puzzles/all_1200_2400_50k` | `0.3323` | `0.3256` |
| `alpha_loss_reports_v2` | `0.6393` | `0.5574` |
| `puzzles/lines_1200_2400_100k` | `0.4539` | `0.4506` |
| `alpha_poisoned_captures_v2` | `0.7143` | `0.6786` |
| `alpha_loss_pvlines_recent_v1` | `0.2604` | `0.2617` |
| `alpha_loss_gamelines_all_v2` | `0.2827` | `0.2766` |

The branch regressed both the 16k teacher diagnostic it targeted and the fixed
4k broad Stockfish diagnostic.

## Conclusion

Rejected. More broad 16k replay, even conservatively from the PV-recent base,
does not improve the current model and badly fails the parent gate.
