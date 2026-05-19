# Policy Top-K Diagnostics

Timestamp: `2026-05-19T13:33:22-07:00`

Validation now reports policy top-3 and top-5 target-hit rates in addition to
top-1 policy accuracy. All runs below used legal-masked validation with
`--prefer-action-labels` so the metric is against the hard Stockfish action label.

## Latest Loss-Repair Set

Dataset: `data/teacher/alpha_loss_leafmcts_v1`, `574` positions.

| Checkpoint | Top-1 | Top-3 | Top-5 | Policy loss |
| --- | ---: | ---: | ---: | ---: |
| qvalue parent | `0.2544` | `0.5627` | `0.6760` | `2.7264` |
| policy-head broad | `0.2526` | `0.5523` | `0.6760` | `2.6860` |
| hard-label policy-head | `0.2613` | `0.5505` | `0.6864` | `2.6872` |

The target move is often present in the short list but not top-ranked. This
points more toward ranking/search calibration than total policy ignorance on
these recent failure positions.

## Broad 16k Stockfish Teacher Set

Dataset: `data/teacher/stockfish_multipv_elo1800_16384`, `16384` positions.

| Checkpoint | Top-1 | Top-3 | Top-5 | Policy loss |
| --- | ---: | ---: | ---: | ---: |
| qvalue parent | `0.3839` | `0.6553` | `0.7619` | `2.2301` |
| policy-head broad | `0.3787` | `0.6518` | `0.7635` | `2.2102` |
| hard-label policy-head | `0.4052` | `0.6627` | `0.7692` | `2.1537` |
| 16k plus latest-loss policy refresh | `0.4120` | `0.6698` | `0.7738` | `2.1428` |

The broader hard-label refresh improves supervised teacher metrics, but its
direct Stockfish check still scored `0.0/2`. Better supervised top-k accuracy is
therefore not sufficient by itself; the next change should target move selection
under search, value calibration, or direct play-time blunder filtering.
