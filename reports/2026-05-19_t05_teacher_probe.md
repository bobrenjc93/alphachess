# Higher-Time Stockfish Teacher Probe

Timestamp: `2026-05-19T22:33:45-07:00`

The latest fullnet192 checkpoints kept failing direct Stockfish through deeper
attacking sequences, so I first checked whether existing stricter root guards
helped the best current checkpoint:

```bash
uv run alpha-chess eval \
  --checkpoint experiments/policyhead192-broad65k-holdoutselect-v1/checkpoints/iter_0001/latest.pt \
  --games 2 \
  --simulations 16 \
  --opponent stockfish \
  --engine-path tools/stockfish/bin/stockfish \
  --engine-time 0.05 \
  --material-value-weight 0.15 \
  --root-mate-search-plies 5 \
  --root-material-search-plies 3 \
  --root-material-max-loss-cp 100 \
  --root-king-safety-search-plies 2 \
  --root-king-safety-max-loss-cp 100 \
  --pgn-out reports/policyhead192_broad65k_holdoutselect_strictguards_stockfish.pgn \
  --device cpu
```

Result: `0.0/2`. The existing root material, mate, and king-safety filters did
not rescue direct play.

## Teacher Data

I generated a smaller broad teacher slice at the same `engine_time=0.05` as the
direct Stockfish gate, instead of the cheap `0.005s` labels used for broad65k:

```text
data/teacher/stockfish_multipv_elo1800_8192_t05
```

Generation summary:

```text
source=data/raw/lichess_db_standard_rated_2013-01.pgn.zst
games_seen=4906
games_used=219
positions=8192
files=8
engine_time=0.05
multipv=4
policy_temperature_cp=200
min_elo=1800
min_initial_seconds=180
position_stride=2
```

Baseline validation on this higher-time slice:

| Checkpoint | Policy loss | Top-1 | Top-3 | Top-5 | Value loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| `policyhead192-broad65k-puzzlemix-v1` | `2.1111` | `0.5610` | `0.7396` | `0.8077` | `0.0440` |
| `policyhead192-broad65k-holdoutselect-v1` | `2.0915` | `0.5616` | `0.7394` | `0.8098` | `0.0440` |
| `policyhead-broad65k-expertmix-v1` | `1.8679` | `0.4391` | `0.7279` | `0.8369` | `0.0424` |

The `192x8` policy has much better top-1 on this overlapping broad slice, while
the `128x6` expert-mix checkpoint still has stronger top-5 and policy loss.

## T05 Policy-Head Tune

I trained a bounded CPU policy-head-only tune from the current fullnet192
holdout-selected checkpoint:

```text
experiments/policyhead192-t05-holdoutselect-v1
```

Config highlights:

- checkpoint: `experiments/policyhead192-broad65k-holdoutselect-v1/checkpoints/iter_0001/latest.pt`
- data: `stockfish_multipv_elo1800_8192_t05`, broad65k,
  `fullnet192_lossblunders_v1`, `fullnet192_holdoutselect_lossblunders_v1`
- holdout: `stockfish_multipv_elo1800_holdout8192_t005_skip65536`
- replay weights: `0.60 0.30 0.05 0.05`
- `policy_head_only=true`
- `lr=0.000008`
- `bad_action_weight=0.10`
- `select_best_by=holdout_policy_acc`

The selector chose epoch 3:

| Epoch | Train split top-1 | Holdout top-1 | Holdout top-3 | Holdout top-5 | Holdout policy loss | Saved as `latest.pt` |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `1` | `0.8381` | `0.3409` | `0.5479` | `0.6478` | `2.8300` | yes |
| `2` | `0.8379` | `0.3419` | `0.5454` | `0.6459` | `2.7709` | yes |
| `3` | `0.8381` | `0.3420` | `0.5464` | `0.6451` | `2.7555` | yes |

Selected checkpoint validation:

| Dataset | Top-1 | Top-3 | Top-5 | Policy loss | Bad-action loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| higher-time broad t05 | `0.5675` | `0.7498` | `0.8184` | `1.9736` | N/A |
| disjoint broad holdout | `0.3420` | `0.5464` | `0.6451` | `2.7555` | N/A |
| `fullnet192_lossblunders_v1` | `0.2899` | `0.5126` | `0.6008` | `2.5630` | `2.5472` |

Direct check:

| Check | Score | PGN |
| --- | ---: | --- |
| Stockfish gate | `0.0/2` | `reports/policyhead192_t05_holdoutselect_stockfish_gate.pgn` |

Read: the higher-time teacher improves supervised agreement with the `0.05s`
broad slice and the older fullnet loss replay, but it regresses the disjoint
broad holdout top-1 from `0.3439` to `0.3420` and still fails direct Stockfish.
This is not a promotion path.
