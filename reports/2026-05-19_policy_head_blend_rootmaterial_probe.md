# Policy-Head Blend And Root-Material Probe

Date: 2026-05-19

## Motivation

The first policy-head-only broad checkpoint produced one Stockfish draw in its
gate, while stronger policy-head variants beat the qvalue parent but lost all
Stockfish games. This probe tested whether interpolating those policy-head
updates back toward the qvalue parent, or adding a shallow root material guard,
could keep the useful policy changes without the tactical collapse.

## Checkpoints

Base:

`experiments/focus-qvalue-puzzlelines20-vw025-material015/checkpoints/iter_0001/latest.pt`

Policy-head candidates:

- `experiments/policyhead-broad-qvalue-vw000-material015/checkpoints/iter_0001/latest.pt`
- `experiments/policyhead-tailstrong-qvalue-vw000-material015/checkpoints/iter_0001/latest.pt`
- `experiments/policyhead-badaction-qvalue-bw005-vw000-material015/checkpoints/iter_0001/latest.pt`

Generated local blend checkpoints:

- `experiments/blend-qvalue-policyhead-broad-w025/latest.pt`
- `experiments/blend-qvalue-policyhead-broad-w050/latest.pt`
- `experiments/blend-qvalue-policyhead-tailstrong-w025/latest.pt`
- `experiments/blend-qvalue-policyhead-badaction-w025/latest.pt`

## Direct Stockfish Results

Settings unless noted:

- `simulations=16`
- `games=2`
- `engine_time=0.05`
- `material_value_weight=0.15`
- `material_value_search_plies=2`
- CPU evaluation

| Probe | Score | W/D/L | PGN |
| --- | ---: | --- | --- |
| qvalue/broad blend `w=0.25` | `0.0/2` | `0/0/2` | `reports/blend_qvalue_policyhead_broad-w025_stockfish_16sims.pgn` |
| qvalue/broad blend `w=0.50` | `0.0/2` | `0/0/2` | `reports/blend_qvalue_policyhead_broad-w050_stockfish_16sims.pgn` |
| qvalue/tailstrong blend `w=0.25` | `0.0/2` | `0/0/2` | `reports/blend_qvalue_policyhead_tailstrong-w025_stockfish_16sims.pgn` |
| qvalue/bad-action blend `w=0.25` | `0.0/2` | `0/0/2` | `reports/blend_qvalue_policyhead_badaction-w025_stockfish_16sims.pgn` |
| policy-head broad + root material guard | `0.0/2` | `0/0/2` | `reports/policyhead_broad_qvalue_rootmaterial250_stockfish_16sims.pgn` |
| qvalue + value-based root selection | `0.0/2` | `0/0/2` | `reports/qvalue_value_select_stockfish_16sims.pgn` |
| policy-head broad + value-based root selection | `0.0/2` | `0/0/2` | `reports/policyhead-broad_value_select_stockfish_16sims.pgn` |

Root-material settings:

- `root_material_search_plies=2`
- `root_material_max_loss_cp=250`

## Conclusion

Rejected. The direct failure is not fixed by simple interpolation back toward
the qvalue parent, the shallow root material guard, or value-based root move
selection. These probes still lose tactically from both colors.
