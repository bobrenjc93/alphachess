# Policy Distillation Anchor Probe

Timestamp: `2026-05-20T14:33:37-07:00`

## Summary

I used the new policy-distillation training anchor on the guarded-blend failure
slice. The code path works, but the first repair settings did not produce a
promotable candidate: every trained checkpoint missed the broad holdout guard.
Adding the mined context positions as an exact good-action book changed the
direct games, but the direct Stockfish score stayed `0.0/2`.

## Recreated Anchor

The transient guarded-blend checkpoint was not present in the local checkout, so
I recreated the reported `10%` blend from the committed parent:

```bash
CUDA_VISIBLE_DEVICES=0 uv run alpha-chess train \
  --checkpoint experiments/policyhead192-stockfish-confirmed-broad73k-top3-fullnet-v1/checkpoints/iter_0001/latest.pt \
  --data data/teacher/stockfish_multipv_elo1800_65536_t005 \
  --holdout-data data/teacher/stockfish_multipv_elo1800_holdout8192_t005_skip65536 \
  --out experiments/policyhead192-distill-anchor-v1/checkpoints/broad_recreate \
  --epochs 3 \
  --batch-size 512 \
  --lr 1e-7 \
  --weight-decay 1e-4 \
  --value-weight 0.05 \
  --bad-action-weight 0.3 \
  --bad-action-margin 1.0 \
  --legal-policy-loss \
  --policy-head-only \
  --device cuda
```

Broad epoch 3 reproduced the rejected broad-only metrics:

| Checkpoint | Holdout top-1 | Holdout top-3 | Holdout top-5 | Holdout policy loss |
| --- | ---: | ---: | ---: | ---: |
| broad epoch 3 | `0.3395` | `0.5386` | `0.6404` | `3.6365` |

Then I blended it back into the parent:

```bash
uv run alpha-chess blend-checkpoints \
  --checkpoint-a experiments/policyhead192-stockfish-confirmed-broad73k-top3-fullnet-v1/checkpoints/iter_0001/latest.pt \
  --checkpoint-b experiments/policyhead192-distill-anchor-v1/checkpoints/broad_recreate/epoch_0003.pt \
  --weight-b 0.10 \
  --out experiments/policyhead192-distill-anchor-v1/checkpoints/broad_epoch3_w0.10.pt
```

The recreated blend also matched the earlier guard-passing metrics:

| Checkpoint | Holdout top-1 | Holdout top-3 | Holdout top-5 | Holdout policy loss |
| --- | ---: | ---: | ---: | ---: |
| broad epoch 3, `w=0.10` | `0.3401` | `0.5426` | `0.6422` | `3.7688` |

## Distillation Repairs

I regenerated the 18-position first-blunder context slice from the committed
guarded-blend PGN:

```bash
uv run alpha-chess stockfish-teacher \
  --pgn reports/policyhead192_guarded_blend_broad_epoch3_w010_stockfish_gate.pgn \
  --out data/teacher/guarded_blend_gate_firstblunders_context_v1 \
  --engine-path tools/stockfish/bin/stockfish \
  --player-name AlphaChess \
  --position-stride 1 \
  --min-value-delta 0.08 \
  --multipv 4 \
  --policy-temperature-cp 180 \
  --first-blunder-only \
  --blunder-context-plies 2 \
  --pv-plies 4 \
  --game-line-plies 2 \
  --chunk-size 256
```

Result: `18` positions.

The first anchored repair used only the 18 context positions:

```bash
CUDA_VISIBLE_DEVICES=0 uv run alpha-chess train \
  --checkpoint experiments/policyhead192-distill-anchor-v1/checkpoints/broad_epoch3_w0.10.pt \
  --distill-checkpoint experiments/policyhead192-distill-anchor-v1/checkpoints/broad_epoch3_w0.10.pt \
  --policy-distill-weight 1.0 \
  --data data/teacher/guarded_blend_gate_firstblunders_context_v1 \
  --holdout-data data/teacher/stockfish_multipv_elo1800_holdout8192_t005_skip65536 \
  --out experiments/policyhead192-distill-anchor-v1/checkpoints/contextonly_distill_w1_lr5e8 \
  --epochs 5 \
  --batch-size 18 \
  --lr 5e-8 \
  --weight-decay 1e-4 \
  --value-weight 0.05 \
  --bad-action-weight 0.3 \
  --bad-action-margin 1.0 \
  --legal-policy-loss \
  --policy-head-only \
  --select-best-by holdout_policy_acc+holdout_policy_top3_acc \
  --select-best-require 'holdout_policy_acc>=0.3400' 'holdout_policy_top3_acc>=0.5420' \
  --device cuda
```

No epoch satisfied the guard. The best top-3 epoch reached `0.5424`, but top-1
fell to `0.3381`.

I then sampled broad positions for the distillation anchor while applying
supervised policy pressure only to the context source:

```bash
CUDA_VISIBLE_DEVICES=0 uv run alpha-chess train \
  --checkpoint experiments/policyhead192-distill-anchor-v1/checkpoints/broad_epoch3_w0.10.pt \
  --distill-checkpoint experiments/policyhead192-distill-anchor-v1/checkpoints/broad_epoch3_w0.10.pt \
  --policy-distill-weight 5.0 \
  --data data/teacher/stockfish_multipv_elo1800_65536_t005 data/teacher/guarded_blend_gate_firstblunders_context_v1 \
  --holdout-data data/teacher/stockfish_multipv_elo1800_holdout8192_t005_skip65536 \
  --out experiments/policyhead192-distill-anchor-v1/checkpoints/broadanchor_context_spw100_distill5_lr5e8 \
  --epochs 5 \
  --batch-size 512 \
  --lr 5e-8 \
  --weight-decay 1e-4 \
  --value-weight 0.05 \
  --bad-action-weight 0.3 \
  --bad-action-margin 1.0 \
  --data-weights 0.95 0.05 \
  --max-source-repeat 5 \
  --source-policy-weights 0.0 100.0 \
  --legal-policy-loss \
  --policy-head-only \
  --select-best-by holdout_policy_acc+holdout_policy_top3_acc \
  --select-best-require 'holdout_policy_acc>=0.3400' 'holdout_policy_top3_acc>=0.5420' \
  --device cuda
```

That also missed the guard: top-3 stayed in the rejected `0.5382-0.5386` band,
although context policy loss improved from `4.2763` to `4.0638` by epoch 2.

A stronger anchor and lower context policy multiplier behaved similarly:

| Run | Best holdout top-1 | Best holdout top-3 | Best context policy loss | Read |
| --- | ---: | ---: | ---: | --- |
| context-only, distill `1`, lr `5e-8` | `0.3391` | `0.5424` | `4.2830` | broad top-1 below guard |
| broad anchor, context policy `100`, distill `5` | `0.3394` | `0.5386` | `4.0638` | broad top-3 collapsed |
| broad anchor, context policy `10`, distill `100` | `0.3391` | `0.5386` | `4.0286` | broad top-3 collapsed |

The practical read is that batch-local policy KL is not enough by itself here.
It can reduce the tiny context loss, but it does not preserve the broad ranking
signal under this kind of narrow update. A useful follow-up would need either a
larger and more representative repair source or a selector-aware objective that
directly penalizes broad top-k regressions.

## Context-Book Gate

I also tested the recreated guard-passing blend with the first-blunder context
slice added as an exact good-action book:

```bash
CUDA_VISIBLE_DEVICES=0 uv run alpha-chess eval \
  --checkpoint experiments/policyhead192-distill-anchor-v1/checkpoints/broad_epoch3_w0.10.pt \
  --opponent stockfish \
  --engine-path tools/stockfish/bin/stockfish \
  --engine-time 0.05 \
  --games 2 \
  --simulations 16 \
  --device cuda \
  --material-value-weight 0.15 \
  --root-mate-search-plies 5 \
  --root-material-search-plies 3 \
  --root-material-max-loss-cp 100 \
  --root-king-safety-search-plies 2 \
  --root-king-safety-max-loss-cp 100 \
  --good-action-book data/teacher/stockfish_multipv_elo1800_65536_t005 data/teacher/stockfish_multipv_elo1800_8192_t05 data/teacher/guarded_blend_gate_firstblunders_context_v1 \
  --bad-action-book data/teacher/policyhead192_stockfish_confirmed_blunders_broad73k_top3_v1 \
  --pgn-out reports/policyhead192_guarded_blend_contextbook_stockfish_gate.pgn
```

Result: `{'games': 2.0, 'score': 0.0, 'score_rate': 0.0, 'wins': 0.0,
'draws': 0.0, 'losses': 2.0}`.

PGN file mtime: `2026-05-20T14:33:37-07:00`.

The exact context book did change the games. New first-blunder mining from that
gate found different lead-up mistakes:

| Game | First confirmed mistake with context | Stockfish target | Value delta | FEN |
| --- | --- | --- | ---: | --- |
| 1 | `h4` | `Bf4` | `0.5182` | `r1bq1rk1/p1p1bpp1/5n1p/3p2B1/8/3B4/PPP2PPP/RN1QR1K1 w - - 0 11` |
| 2 | `...Qb6` | `...a6` | `0.1967` | `rnbqkb1r/pp1pnppp/4p3/8/3NP3/2N5/PPP2PPP/R1BQKB1R b KQkq - 0 5` |

So exact context coverage is useful as a diagnostic, but it is still just
moving the failure surface. The direct gate remains scoreless.

## Second Context-Book Iteration

I added the first context-book gate's newly mined context positions as a second
exact good-action book and reran the same direct gate:

```bash
CUDA_VISIBLE_DEVICES=0 uv run alpha-chess eval \
  --checkpoint experiments/policyhead192-distill-anchor-v1/checkpoints/broad_epoch3_w0.10.pt \
  --opponent stockfish \
  --engine-path tools/stockfish/bin/stockfish \
  --engine-time 0.05 \
  --games 2 \
  --simulations 16 \
  --device cuda \
  --material-value-weight 0.15 \
  --root-mate-search-plies 5 \
  --root-material-search-plies 3 \
  --root-material-max-loss-cp 100 \
  --root-king-safety-search-plies 2 \
  --root-king-safety-max-loss-cp 100 \
  --good-action-book data/teacher/stockfish_multipv_elo1800_65536_t005 data/teacher/stockfish_multipv_elo1800_8192_t05 data/teacher/guarded_blend_gate_firstblunders_context_v1 data/teacher/guarded_blend_contextbook_firstblunders_context_v1 \
  --bad-action-book data/teacher/policyhead192_stockfish_confirmed_blunders_broad73k_top3_v1 \
  --pgn-out reports/policyhead192_guarded_blend_contextbook_v2_stockfish_gate.pgn
```

Result: `{'games': 2.0, 'score': 0.0, 'score_rate': 0.0, 'wins': 0.0,
'draws': 0.0, 'losses': 2.0}`.

PGN file mtime: `2026-05-20T14:40:48-07:00`.

The added exact coverage again shifted the games but not the score. New
first-blunder mining found:

| Game | First confirmed mistake with context | Stockfish target | Value delta | FEN |
| --- | --- | --- | ---: | --- |
| 1 | `Be3` | `b3` | `0.4835` | `r1bq1rk1/p3bppp/5n2/2pp4/8/2NB3P/PPP2PP1/R1BQ1RK1 w - - 0 11` |
| 2 | `...e5` | `...d6` | `0.0866` | `r1bq1rk1/pp2bppp/2nppn2/8/3NP3/2N1B3/PPPQBPPP/2KR3R b - - 8 9` |

This confirms that exact books can steer away from known losing choices, but the
current policy/search stack still exposes new adjacent tactical weaknesses
immediately. The next useful data step is to aggregate several of these
first-blunder contexts into a broader opening-stability source before trying
more supervised repair.

## Separate Distillation Data Follow-Up

I aggregated all three guarded-blend/context-book gates into one broader
first-blunder context source:

```bash
uv run alpha-chess stockfish-teacher \
  --pgn reports/policyhead192_guarded_blend_broad_epoch3_w010_stockfish_gate.pgn reports/policyhead192_guarded_blend_contextbook_stockfish_gate.pgn reports/policyhead192_guarded_blend_contextbook_v2_stockfish_gate.pgn \
  --out data/teacher/guarded_blend_all_contextbook_firstblunders_context_v1 \
  --engine-path tools/stockfish/bin/stockfish \
  --player-name AlphaChess \
  --position-stride 1 \
  --min-value-delta 0.08 \
  --multipv 4 \
  --policy-temperature-cp 180 \
  --first-blunder-only \
  --blunder-context-plies 2 \
  --pv-plies 4 \
  --game-line-plies 2 \
  --chunk-size 256
```

Result: `54` positions from `6` failed direct games.

I then added `--distill-data` so each supervised repair step can draw a separate
broad teacher-anchor batch instead of relying on the narrow supervised batch to
carry both jobs. The first run used the 54-position context source as supervised
data and broad65k as the independent anchor:

```bash
CUDA_VISIBLE_DEVICES=0 uv run alpha-chess train \
  --checkpoint experiments/policyhead192-distill-anchor-v1/checkpoints/broad_epoch3_w0.10.pt \
  --distill-checkpoint experiments/policyhead192-distill-anchor-v1/checkpoints/broad_epoch3_w0.10.pt \
  --distill-data data/teacher/stockfish_multipv_elo1800_65536_t005 \
  --distill-batch-size 1024 \
  --policy-distill-weight 10.0 \
  --data data/teacher/guarded_blend_all_contextbook_firstblunders_context_v1 \
  --holdout-data data/teacher/stockfish_multipv_elo1800_holdout8192_t005_skip65536 \
  --out experiments/policyhead192-distill-anchor-v1/checkpoints/all_context_distilldata_w10_lr1e8 \
  --epochs 5 \
  --batch-size 54 \
  --lr 1e-8 \
  --weight-decay 1e-4 \
  --value-weight 0.05 \
  --bad-action-weight 0.3 \
  --bad-action-margin 1.0 \
  --legal-policy-loss \
  --policy-head-only \
  --select-best-by holdout_policy_acc+holdout_policy_top3_acc \
  --select-best-require 'holdout_policy_acc>=0.3400' 'holdout_policy_top3_acc>=0.5420' \
  --device cuda
```

No epoch satisfied the broad guard.

| Checkpoint | Holdout top-1 | Holdout top-3 | Holdout top-5 | Holdout policy loss | Context policy loss | Read |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| starting blend | `0.3401` | `0.5426` | `0.6422` | `3.7688` | `3.5373` | guard-passing baseline |
| epoch 2 | `0.3397` | `0.5410` | `0.6410` | `3.7654` | `3.5188` | context loss improves, guard missed |
| epoch 5 | `0.3386` | `0.5402` | `0.6410` | `3.7748` | `3.5129` | more context fit, broader regression |

I also interpolated epoch 2 back toward the starting blend:

| Epoch-2 weight | Holdout top-1 | Holdout top-3 | Holdout top-5 | Context policy loss | Read |
| ---: | ---: | ---: | ---: | ---: | --- |
| `0.25` | `0.3392` | `0.5419` | `0.6422` | `3.5326` | top-1/top-3 below guard |
| `0.50` | `0.3394` | `0.5419` | `0.6417` | `3.5279` | top-1/top-3 below guard |
| `0.75` | `0.3394` | `0.5411` | `0.6414` | `3.5233` | top-1/top-3 below guard |

The separate anchor batch is a better training primitive than mixing anchor and
repair roles in one batch: it improved the aggregate context loss while avoiding
the severe `0.538x` top-3 collapse. It still did not keep enough broad top-k
ranking to justify a direct Stockfish gate.

## Aggregated Context-Book Gate

Finally, I used the merged 54-position source directly as the exact good-action
book, instead of passing the first two context slices separately:

```bash
CUDA_VISIBLE_DEVICES=0 uv run alpha-chess eval \
  --checkpoint experiments/policyhead192-distill-anchor-v1/checkpoints/broad_epoch3_w0.10.pt \
  --opponent stockfish \
  --engine-path tools/stockfish/bin/stockfish \
  --engine-time 0.05 \
  --games 2 \
  --simulations 16 \
  --device cuda \
  --material-value-weight 0.15 \
  --root-mate-search-plies 5 \
  --root-material-search-plies 3 \
  --root-material-max-loss-cp 100 \
  --root-king-safety-search-plies 2 \
  --root-king-safety-max-loss-cp 100 \
  --good-action-book data/teacher/stockfish_multipv_elo1800_65536_t005 data/teacher/stockfish_multipv_elo1800_8192_t05 data/teacher/guarded_blend_all_contextbook_firstblunders_context_v1 \
  --bad-action-book data/teacher/policyhead192_stockfish_confirmed_blunders_broad73k_top3_v1 \
  --pgn-out reports/policyhead192_guarded_blend_contextbook_all_stockfish_gate.pgn
```

Result: `{'games': 2.0, 'score': 0.0, 'score_rate': 0.0, 'wins': 0.0,
'draws': 0.0, 'losses': 2.0}`.

PGN file mtime: `2026-05-20T14:57:23-07:00`.

The merged exact book produced different lines from both previous context-book
checks, but the score was still zero. New first-blunder mining found:

| Game | First confirmed mistake with context | Stockfish target | Value delta | FEN |
| --- | --- | --- | ---: | --- |
| 1 | `Nd2` | `Qf3` | `0.0882` | `r1bqk2r/p3bppp/5n2/2pp4/8/3B3P/PPP2PP1/RNBQ1RK1 w kq - 0 10` |
| 2 | `...Bb4` | `...Nxd4` | `0.1173` | `r1bq1k1r/pp1p1ppp/2nN1n2/2b1p3/4P3/2N1B3/PPP2PPP/R2QKB1R b KQ - 1 9` |

Exact context coverage is therefore only a failure-surface probe at this point.
The model needs a broader opening-stability repair signal that can improve these
lead-up choices without dropping below the broad top-k guard.
