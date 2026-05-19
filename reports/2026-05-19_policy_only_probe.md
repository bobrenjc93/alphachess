# Policy-Only Direct Probe

Timestamp: `2026-05-19T13:37:44-07:00`

This probe used `--simulations 0`, which expands the root and chooses the
highest-prior legal policy move without MCTS visits. It tests whether search is
actively harming an otherwise safer policy.

All games used local Stockfish with `engine_time=0.05`,
`material_value_weight=0.15`, `material_value_search_plies=2`, and
`max_plies=180`.

| Checkpoint | Direct Stockfish score | PGN |
| --- | ---: | --- |
| `policyhead-16k-leafloss-qvalue-vw000-material015` | `0.0/2` | `reports/policyhead_16k_stockfish_policyonly.pgn` |
| `policyhead-hardlabels-qvalue-vw000-material015` | `0.0/2` | `reports/policyhead_hardlabels_stockfish_policyonly.pgn` |

Policy-only play also collapses tactically. The direct-play issue is therefore
not just MCTS steering away from a safe policy; the policy head still needs
stronger tactical/opening reliability before search can amplify it.
