# Policy-Head-Only Broad Replay Probe

Date: 2026-05-19

## Change Under Test

This run used policy-head-only training from `6728459`:

- shared trunk frozen
- value head frozen
- frozen modules kept in eval mode so their batch-norm statistics do not move
- only `policy_head` parameters train

The intent was to repair policy priors without damaging the value/search path,
after the policy-broad replay probe improved fixed policy metrics but lost
`0/16` to the qvalue parent.

## Run

`experiments/policyhead-broad-qvalue-vw000-material015`

Base checkpoint:

`experiments/focus-qvalue-puzzlelines20-vw025-material015/checkpoints/iter_0001/latest.pt`

Config:

- `games=0`
- `epochs=1`
- `batch_size=128`
- `lr=1e-5`
- `value_weight=0.0`
- `legal_policy_loss=true`
- `policy_head_only=true`
- `material_value_weight=0.15`
- `material_value_search_plies=2`
- replay data:
  - `data/teacher/stockfish_multipv_elo1800_4096`
  - `data/puzzles/lines_1200_2400_100k`
  - `data/teacher/alpha_loss_reports_v3`
  - `data/teacher/alpha_poisoned_captures_v2`
- replay weights: `0.65 0.25 0.08 0.02`

## Promotion

Candidate:

`experiments/policyhead-broad-qvalue-vw000-material015/checkpoints/iter_0001/latest.pt`

Parent match:

- score: `8.0/16`
- wins/draws/losses: `0/16/0`

Stockfish gate:

- score: `0.5/4`
- wins/draws/losses: `0/1/3`
- PGN: `reports/policyhead_broad_qvalue_stockfish_gate_16sims.pgn`
- promoted: `false`

Higher-search direct smoke:

- 64 simulations, 1 Stockfish game
- score: `0.0/1`
- PGN: `reports/policyhead_broad_qvalue_stockfish_64sims.pgn`

## Fixed validation

Candidate-only validation used CPU, batch size 512, legal policy loss, and
`value_weight=0.25`. Parent qvalue metrics are included for comparison from the
same fixed validator.

| Dataset | Parent Policy Acc | Candidate Policy Acc | Delta |
| --- | ---: | ---: | ---: |
| `stockfish_multipv_elo1800_4096` | `0.6050` | `0.5564` | `-0.0486` |
| `puzzles/all_1200_2400_50k` | `0.3365` | `0.3317` | `-0.0049` |
| `alpha_loss_reports_v2` | `0.5738` | `0.5082` | `-0.0656` |
| `puzzles/lines_1200_2400_100k` | `0.4539` | `0.4542` | `+0.0004` |
| `alpha_poisoned_captures_v2` | `0.0357` | `0.0714` | `+0.0357` |
| self-play `iter_0001` | `0.6034` | `0.5967` | `-0.0067` |

Overall validation:

| Checkpoint | Policy Acc | Loss |
| --- | ---: | ---: |
| parent qvalue | `0.4276` | `2.3287` |
| candidate policy-head-only | `0.4248` | `2.3054` |

## Conclusion

Rejected, but informative. Freezing the trunk and value head avoided the total
search collapse from the previous policy-broad replay probe: the candidate drew
all 16 parent games and scored the first direct Stockfish draw in this sequence.
However, the Stockfish gate was still only `0.5/4`, the 64-sim smoke lost, and
fixed broad policy accuracy regressed.

The branch is worth refining with a lower learning rate or narrower replay mix,
but this checkpoint is not a promotion candidate.
