# Replay Value-Only Policy-Broad Probe

Date: 2026-05-19

## Run

`experiments/replayvalueonly-policybroad-qvalue-vw025-material015`

Base checkpoint:

`experiments/focus-qvalue-puzzlelines20-vw025-material015/checkpoints/iter_0001/latest.pt`

This run reused the deterministic 64-game self-play batch from:

`experiments/gated-selfplay64-valueonlypolicy-qvalue-vw025-material015/selfplay/iter_0001`

No new self-play was generated.

## Config

- `games=0`
- `epochs=1`
- `batch_size=128`
- `lr=4e-6`
- `value_weight=0.25`
- `legal_policy_loss=true`
- `material_value_weight=0.15`
- `material_value_search_plies=2`
- replay data:
  - self-play `iter_0001`
  - `data/teacher/stockfish_multipv_elo1800_4096`
  - `data/puzzles/lines_1200_2400_100k`
  - `data/teacher/alpha_loss_reports_v3`
  - `data/teacher/alpha_poisoned_captures_v2`
- replay weights: `0.10 0.55 0.20 0.10 0.05`
- replay policy weights: `0.0 1.0 1.0 0.0 0.0`

The intent was to train policy only from broad Stockfish and puzzle-line replay,
while using self-play, alpha-loss, and poisoned-capture sources as value-only
signals.

## Promotion

Candidate:

`experiments/replayvalueonly-policybroad-qvalue-vw025-material015/checkpoints/iter_0001/latest.pt`

Parent match:

- score: `0.0/16`
- wins/draws/losses: `0/0/16`
- promoted: `false`

The direct Stockfish gate did not run because the candidate failed the parent
gate.

## Fixed validation

Candidate-only validation used CPU, batch size 512, legal policy loss, and
`value_weight=0.25`. Parent qvalue metrics are included for comparison from the
same fixed validator.

| Dataset | Parent Policy Acc | Candidate Policy Acc | Delta |
| --- | ---: | ---: | ---: |
| `stockfish_multipv_elo1800_4096` | `0.6050` | `0.6172` | `+0.0122` |
| `puzzles/all_1200_2400_50k` | `0.3365` | `0.3365` | `-0.0001` |
| `alpha_loss_reports_v2` | `0.5738` | `0.6393` | `+0.0656` |
| `puzzles/lines_1200_2400_100k` | `0.4539` | `0.4554` | `+0.0015` |
| `alpha_poisoned_captures_v2` | `0.0357` | `0.0357` | `+0.0000` |
| self-play `iter_0001` | `0.6034` | `0.6030` | `-0.0004` |

Overall validation:

| Checkpoint | Policy Acc | Loss |
| --- | ---: | ---: |
| parent qvalue | `0.4276` | `2.3287` |
| candidate policy-broad | `0.4289` | `2.3162` |

## Conclusion

Rejected. The per-source policy weighting did what it was meant to do on fixed
policy diagnostics: broad Stockfish and puzzle-line policy accuracy were
preserved or improved. But the candidate lost every parent game, so the value
or shared trunk updates made search play much worse.

The next useful branch is to protect the value/search behavior during policy
repair, for example by freezing the shared body/value head or adding a stronger
parent-anchor loss.
