# Policy Exploration Probe

Timestamp: `2026-05-19T13:36:12-07:00`

The policy top-k diagnostics showed many Stockfish targets in the network top-5,
so this probe tested whether higher MCTS exploration and softer policy priors
would recover any direct Stockfish score.

All games used local Stockfish with `engine_time=0.05`, `64` MCTS simulations,
`material_value_weight=0.15`, `material_value_search_plies=2`, and
`max_plies=180`.

| Checkpoint | `c_puct` | Policy prior temperature | Direct Stockfish score | PGN |
| --- | ---: | ---: | ---: | --- |
| `policyhead-16k-leafloss-qvalue-vw000-material015` | `3.0` | `2.0` | `0.0/2` | `reports/policyhead_16k_stockfish_c3_t2_64sims.pgn` |
| `policyhead-16k-leafloss-qvalue-vw000-material015` | `5.0` | `3.0` | `0.0/2` | `reports/policyhead_16k_stockfish_c5_t3_64sims.pgn` |
| `policyhead-hardlabels-qvalue-vw000-material015` | `3.0` | `2.0` | `0.0/2` | `reports/policyhead_hardlabels_stockfish_c3_t2_64sims.pgn` |

Softer priors and higher exploration did not recover the gate. The top-k
diagnostic remains useful, but these results suggest that direct play still
needs stronger value calibration or a more tactical root-selection mechanism,
not just broader root exploration.
