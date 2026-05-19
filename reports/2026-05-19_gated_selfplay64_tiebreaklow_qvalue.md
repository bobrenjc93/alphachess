# Gated Self-Play 64 Tiebreak-Low Qvalue Run

Date: 2026-05-19

## Run

`experiments/gated-selfplay64-tiebreaklow-qvalue-vw025-material015`

Base checkpoint:

`experiments/focus-qvalue-puzzlelines20-vw025-material015/checkpoints/iter_0001/latest.pt`

Config:

- `games=64`
- `self_play_workers=8`
- `simulations=64`
- `max_plies=180`
- `epochs=1`
- `batch_size=128`
- `lr=4e-6`
- `value_weight=0.25`
- `legal_policy_loss=true`
- `material_value_weight=0.15`
- `material_value_search_plies=2`
- replay data:
  - `data/teacher/stockfish_multipv_elo1800_4096`
  - `data/puzzles/lines_1200_2400_100k`
  - `data/teacher/alpha_loss_reports_v3`
  - `data/teacher/alpha_poisoned_captures_v2`
- weights:
  - self-play: `0.10`
  - replay: `0.55 0.20 0.10 0.05`
- parent promotion gate: 16 games at 64 sims, 4 eval workers
- Stockfish gate enabled, but only after parent promotion:
  - `stockfish_gate_games=4`
  - `stockfish_gate_simulations=16`
  - `stockfish_gate_min_score=0.50`
  - `stockfish_gate_engine_path=tools/stockfish/bin/stockfish`
  - `stockfish_gate_engine_time=0.05`

## Self-play data

Generated 64 games in:

`experiments/gated-selfplay64-tiebreaklow-qvalue-vw025-material015/selfplay/iter_0001`

Self-play result summary:

- `1-0`: 26
- `0-1`: 21
- `1/2-1/2`: 15
- `*`: 2
- positions: 6,884
- min plies: 43
- max plies: 180
- average plies: 107.56
- average nonzero policy entries: 3.93

## Promotion

Candidate:

`experiments/gated-selfplay64-tiebreaklow-qvalue-vw025-material015/checkpoints/iter_0001/latest.pt`

Parent match:

- score: `4.0/16`
- wins/draws/losses: `0/8/8`
- promoted: `false`

The direct Stockfish gate did not run because the candidate failed the parent gate first.

## Fixed validation

Validation used CPU, batch size 512, legal policy loss, and `value_weight=0.25`.

| Dataset | Examples | Parent Policy Acc | Candidate Policy Acc | Delta |
| --- | ---: | ---: | ---: | ---: |
| `stockfish_multipv_elo1800_4096` | 4,096 | `0.6050` | `0.5520` | `-0.0530` |
| `puzzles/all_1200_2400_50k` | 50,000 | `0.3365` | `0.3357` | `-0.0009` |
| `alpha_loss_reports_v2` | 61 | `0.5738` | `0.6393` | `+0.0656` |
| `puzzles/lines_1200_2400_100k` | 100,000 | `0.4539` | `0.4537` | `-0.0001` |
| `alpha_poisoned_captures_v2` | 28 | `0.0357` | `0.2857` | `+0.2500` |
| self-play `iter_0001` | 6,884 | `0.6034` | `0.6136` | `+0.0102` |

Overall validation:

| Checkpoint | Policy Acc | Loss |
| --- | ---: | ---: |
| parent qvalue | `0.4276` | `2.3287` |
| candidate self-play64 | `0.4265` | `2.3230` |

## Conclusion

Rejected. The checkpoint fit the new 64-game self-play batch slightly better and improved the small alpha-loss and poisoned-capture probes, but it lost badly to the parent and regressed broad Stockfish teacher policy accuracy. Low-weight self-play from the current qvalue policy is still not a reliable improvement signal.

Next work should prioritize self-play quality filtering, a stronger search/training target, or a larger capacity run rather than adding more unfiltered low-weight self-play from this incumbent.
