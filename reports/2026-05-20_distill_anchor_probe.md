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

## 64-Simulation Check

I reran the merged context-book gate at 64 MCTS simulations to check whether the
16-simulation result was search-starved:

```bash
CUDA_VISIBLE_DEVICES=0 uv run alpha-chess eval \
  --checkpoint experiments/policyhead192-distill-anchor-v1/checkpoints/broad_epoch3_w0.10.pt \
  --opponent stockfish \
  --engine-path tools/stockfish/bin/stockfish \
  --engine-time 0.05 \
  --games 2 \
  --simulations 64 \
  --device cuda \
  --material-value-weight 0.15 \
  --root-mate-search-plies 5 \
  --root-material-search-plies 3 \
  --root-material-max-loss-cp 100 \
  --root-king-safety-search-plies 2 \
  --root-king-safety-max-loss-cp 100 \
  --good-action-book data/teacher/stockfish_multipv_elo1800_65536_t005 data/teacher/stockfish_multipv_elo1800_8192_t05 data/teacher/guarded_blend_all_contextbook_firstblunders_context_v1 \
  --bad-action-book data/teacher/policyhead192_stockfish_confirmed_blunders_broad73k_top3_v1 \
  --pgn-out reports/policyhead192_guarded_blend_contextbook_all_64sims_stockfish_gate.pgn
```

Result: `{'games': 2.0, 'score': 0.0, 'score_rate': 0.0, 'wins': 0.0,
'draws': 0.0, 'losses': 2.0}`.

PGN file mtime: `2026-05-20T15:03:04-07:00`.

The higher-search games changed again but still failed tactically. New
first-blunder mining found:

| Game | First confirmed mistake with context | Stockfish target | Value delta | FEN |
| --- | --- | --- | ---: | --- |
| 1 | `Kh1` | `Bxg6` | `0.4704` | `3q1rk1/pb3p2/4r1pp/2ppB1bn/6Q1/1PNB3P/P1P2PP1/3RR1K1 w - - 6 19` |
| 2 | `...Bb4` | `...d6` | `0.0953` | `r1bqk2r/pp1p1ppp/2n2n2/1Nb1p3/4P3/2N1B3/PPP2PPP/R2QKB1R b KQkq - 1 8` |

The merged exact book plus 64 visits is still scoreless, so the current failure
is not just shallow search at the direct gate.

## Top-K Good-Action Book Check

The exact good-action loader was only using the `actions` array when a teacher
file had both `actions` and `policies`, so `--good-action-book-top-k` was
effectively ignored for the normal Stockfish MultiPV teacher files. I changed
the loader so the default `policy_top_k=1` behavior remains exact-best, while
`policy_top_k > 1` also admits the top positive policy moves.

With that fix in place, I reran the merged context-book gate with top-3
teacher alternatives:

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
  --good-action-book-top-k 3 \
  --bad-action-book data/teacher/policyhead192_stockfish_confirmed_blunders_broad73k_top3_v1 \
  --pgn-out reports/policyhead192_guarded_blend_contextbook_all_top3book_stockfish_gate.pgn
```

Result: `{'games': 2.0, 'score': 0.0, 'score_rate': 0.0, 'wins': 0.0,
'draws': 0.0, 'losses': 2.0}`.

PGN file mtime: `2026-05-20T15:15:06-07:00`.

The top-3 book produced another distinct pair of games, but still lost both.
New first-blunder mining at `engine_time=0.05` found:

| Game | First confirmed mistake with context | Stockfish target | Value delta | FEN |
| --- | --- | --- | ---: | --- |
| 1 | `Re2` | `Nb1` | `0.0833` | `1r3rk1/p2qbppp/4bn2/3p4/2p5/N1P1Q2P/PPB2PP1/R1B1R1K1 w - - 4 16` |
| 2 | `...Qe8` | `...Qe7` | `0.1569` | `r1bq1rk1/ppBp1ppp/2n1pn2/1N6/1b2P3/2N5/PPP2PPP/R2QKB1R b KQ - 6 8` |

This confirms the top-k flag now has the intended effect on combined
action/policy teacher files. It does not solve the current transfer problem:
even broader exact teacher alternatives still move the first failure surface
instead of producing a nonzero direct Stockfish result.

## 72-Position Context Repair Follow-Up

The top-3 exact-book failures were outside the merged good-action book. The
guarded blend's raw policy also ranked the new Stockfish targets very low:

| Game | Played move | Stockfish target | Target policy rank | Target policy prob |
| --- | --- | --- | ---: | ---: |
| 1 | `Re2` | `Nb1` | `32` / `42` legal moves | `0.000123` |
| 2 | `...Qe8` | `...Qe7` | `21` / `32` legal moves | `0.000306` |

I regenerated the aggregate first-blunder context source with the new top-3
book gate included and `engine_time=0.05`:

```bash
uv run alpha-chess stockfish-teacher \
  --pgn reports/policyhead192_guarded_blend_broad_epoch3_w010_stockfish_gate.pgn reports/policyhead192_guarded_blend_contextbook_stockfish_gate.pgn reports/policyhead192_guarded_blend_contextbook_v2_stockfish_gate.pgn reports/policyhead192_guarded_blend_contextbook_all_top3book_stockfish_gate.pgn \
  --out data/teacher/guarded_blend_top3book_all_firstblunders_context_t05_v1 \
  --engine-path tools/stockfish/bin/stockfish \
  --engine-time 0.05 \
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

Result: `72` positions from `8` failed direct games.

I then tested a more conservative independent-anchor repair and a stronger
single-step diagnostic from the guarded blend:

| Run | Best checkpoint read | Holdout top-1 | Holdout top-3 | Holdout top-5 | Context policy loss | Context top-1/top-3/top-5 | Read |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| starting blend | baseline | `0.3401` | `0.5426` | `0.6422` | `3.7314` | `0.1806`/`0.3750`/`0.5417` | Guard-passing broad baseline, but weak on the new context. |
| distill weight `50`, lr `5e-9` | epoch 1 | `0.3394` | `0.5420` | `0.6416` | `3.7249` | `0.1806`/`0.3750`/`0.5278` | Too conservative to move target ranks, but already below top-1 guard. |
| distill weight `10`, lr `1e-8` | epoch 1 | `0.3394` | `0.5420` | `0.6416` | `3.7249` | `0.1806`/`0.3750`/`0.5278` | Same practical trajectory as the higher-distill run. |
| distill weight `10`, lr `1e-5` | single step | `0.3394` | `0.5421` | `0.6415` | `3.7228` | `0.1806`/`0.3750`/`0.5278` | Slightly better context loss, still below top-1 guard. |
| `1e-5` update blended back at `25%` | interpolation | `0.3396` | `0.5425` | `0.6421` | `3.7291` | `0.1806`/`0.3750`/`0.5278` | Best interpolation recovered top-3/top-5 but still missed top-1. |

No checkpoint or interpolation satisfied the broad guard, so I did not spend a
direct Stockfish gate. The current repair signal improves loss marginally but
does not lift the actual target ranks; these failures need broader coverage or
a stronger objective than another tiny context replay.

## Bad-Action Margin Diagnostic

Because the first-blunder rows include both the played mistake and the
Stockfish target, I also tested stronger bad-action margin pressure on the same
72-position source.

| Run | Holdout top-1 | Holdout top-3 | Holdout top-5 | Context policy loss | Context top-3/top-5 | Target ranks after update | Read |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| bad-action weight `3.0`, lr `1e-5`, epoch 2 | `0.3394` | `0.5409` | `0.6406` | not selected | not selected | `34/42`, `21/32` | Missed the broad guard and did not lift the new targets. |
| bad-action weight `3.0`, lr `1e-3`, single-step overfit | `0.3374` | `0.5399` | `0.6415` | `3.5508` | `0.4028`/`0.5833` | `30/42`, `19/32` | The objective can move the context slice, but target ranks remain poor and broad holdout collapses. |

I did not run a direct Stockfish gate for either checkpoint. The margin loss is
not enough by itself: at guarded learning rates it does not move the targets,
and at overfit learning rates it damages broad ranking before the new targets
become plausible policy choices.

## Full-Network Top-K Book Checks

I also tested whether the `--good-action-book-top-k 3` fix helps the stronger
top-3 confirmed-blunder full-network checkpoint directly:

```bash
CUDA_VISIBLE_DEVICES=0 uv run alpha-chess eval \
  --checkpoint experiments/policyhead192-stockfish-confirmed-broad73k-top3-fullnet-v1/checkpoints/iter_0001/latest.pt \
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
  --good-action-book data/teacher/stockfish_multipv_elo1800_65536_t005 data/teacher/stockfish_multipv_elo1800_8192_t05 data/teacher/policyhead192_stockfish_confirmed_blunders_broad73k_top3_v1 \
  --good-action-book-top-k 3 \
  --bad-action-book data/teacher/policyhead192_stockfish_confirmed_blunders_broad73k_top3_v1 \
  --pgn-out reports/policyhead192_top3_fullnet_top3book_stockfish_gate.pgn
```

Result: `0.0/2`. PGN file mtime: `2026-05-20T15:35:02-07:00`.

First-blunder mining found:

| Game | First confirmed mistake with context | Stockfish target | Value delta | FEN |
| --- | --- | --- | ---: | --- |
| 1 | `Bxh6` | `Bf4` | `0.2089` | `r1bq1rk1/p1p1bpp1/5n1p/3p2B1/8/3B4/PPPQ1PPP/RN3RK1 w - - 0 11` |
| 2 | `...Bb4` | `...Nc6` | `0.1181` | `rnbqkb1r/pp1p1ppp/4pn2/8/3NP3/2N5/PPP2PPP/R1BQKB1R b KQkq - 2 5` |

Those overlap known guarded-blend context motifs, so I reran the full-network
checkpoint with the merged guarded-blend context source included:

```bash
CUDA_VISIBLE_DEVICES=0 uv run alpha-chess eval \
  --checkpoint experiments/policyhead192-stockfish-confirmed-broad73k-top3-fullnet-v1/checkpoints/iter_0001/latest.pt \
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
  --good-action-book data/teacher/stockfish_multipv_elo1800_65536_t005 data/teacher/stockfish_multipv_elo1800_8192_t05 data/teacher/guarded_blend_all_contextbook_firstblunders_context_v1 data/teacher/policyhead192_stockfish_confirmed_blunders_broad73k_top3_v1 \
  --good-action-book-top-k 3 \
  --bad-action-book data/teacher/policyhead192_stockfish_confirmed_blunders_broad73k_top3_v1 \
  --pgn-out reports/policyhead192_top3_fullnet_context_top3book_stockfish_gate.pgn
```

Result: `0.0/2`. PGN file mtime: `2026-05-20T15:38:49-07:00`.

First-blunder mining found:

| Game | First confirmed mistake with context | Stockfish target | Value delta | FEN |
| --- | --- | --- | ---: | --- |
| 1 | `Na3` | `h3` | `0.0866` | `r1bq1rk1/p1p1bppp/5n2/3p4/8/3B4/PPP2PPP/RNBQR1K1 w - - 4 10` |
| 2 | `...Qe8` | `...Qe7` | `0.1688` | `r1bq1rk1/ppBp1ppp/2n1pn2/1N6/1b2P3/2N5/PPP2PPP/R2QKB1R b KQ - 6 8` |

The stronger holdout checkpoint has the same direct-transfer problem as the
guarded blend. Exact coverage can redirect the games, but it still exposes new
low-rank opening and early-middlegame decisions immediately.

## Broad Opening/Context Mix

The tiny-context repairs were too narrow, so I tried a broader hard-label
policy-head mix from the guarded blend:

- `stockfish_multipv_elo1800_65536_t005`, weight `0.50`
- `stockfish_multipv_elo1800_8192_t05`, weight `0.20`
- `stockfish_opening_elo1800_8192_t03`, weight `0.25`
- `guarded_blend_top3book_all_firstblunders_context_t05_v1`, weight `0.05`,
  capped at `20` repeats per position
- LR `2e-6`, epochs `2`, hard action labels, bad-action weight `0.5`

Epoch 2 narrowly missed the top-3 guard raw, but blending it back into the
guarded blend produced guard-passing candidates:

| Candidate | Holdout top-1 | Holdout top-3 | Holdout top-5 | Context policy loss | Context top-1/top-3/top-5 | Read |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| raw epoch 2 | `0.3446` | `0.5419` | `0.6440` | N/A | N/A | Better top-1, but top-3 just below the `0.5420` guard. |
| `25%` blend | `0.3451` | `0.5448` | `0.6437` | `3.0278` | `0.2222`/`0.4167`/`0.5556` | Best selected by holdout top-1+top-3. |
| `50%` blend | `0.3459` | `0.5436` | `0.6426` | `2.9826` | `0.2222`/`0.4167`/`0.5556` | Higher top-1, lower top-3. |
| `75%` blend | `0.3455` | `0.5431` | `0.6428` | `2.9422` | `0.2222`/`0.4167`/`0.5417` | More context loss improvement, weaker top-3/top-5. |

I spent a direct gate on the `25%` blend:

```bash
CUDA_VISIBLE_DEVICES=0 uv run alpha-chess eval \
  --checkpoint experiments/policyhead192-distill-anchor-v1/checkpoints/broad_opening_context72_hardlabels_cap20_lr2e6_blend_0.25.pt \
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
  --good-action-book data/teacher/stockfish_multipv_elo1800_65536_t005 data/teacher/stockfish_multipv_elo1800_8192_t05 data/teacher/guarded_blend_top3book_all_firstblunders_context_t05_v1 data/teacher/policyhead192_stockfish_confirmed_blunders_broad73k_top3_v1 \
  --good-action-book-top-k 3 \
  --bad-action-book data/teacher/policyhead192_stockfish_confirmed_blunders_broad73k_top3_v1 \
  --pgn-out reports/policyhead192_broad_opening_context72_blend025_stockfish_gate.pgn
```

Result: `0.0/2`. PGN file mtime: `2026-05-20T15:47:37-07:00`.

First-blunder mining found:

| Game | First confirmed mistake with context | Stockfish target | Value delta | FEN |
| --- | --- | --- | ---: | --- |
| 1 | `Na3` | `h3` | `0.0931` | `r1bq1rk1/p1p1bppp/5n2/3p4/8/3B4/PPP2PPP/RNBQR1K1 w - - 4 10` |
| 2 | `...Qa5` | `...d5` | `0.1465` | `r1bqk1nr/pp1p1ppp/2n1p3/8/3NP3/P1P5/2P2PPP/R1BQKB1R b KQkq - 0 7` |

This is the best broad-holdout candidate in this probe, but it still does not
transfer to direct Stockfish play. The remaining failures are still early
opening choices outside the exact book.

I also spent a gate on the `50%` blend because it had stronger context loss and
the highest holdout top-1 while still satisfying the broad guard:

```bash
CUDA_VISIBLE_DEVICES=0 uv run alpha-chess eval \
  --checkpoint experiments/policyhead192-distill-anchor-v1/checkpoints/broad_opening_context72_hardlabels_cap20_lr2e6_blend_0.50.pt \
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
  --good-action-book data/teacher/stockfish_multipv_elo1800_65536_t005 data/teacher/stockfish_multipv_elo1800_8192_t05 data/teacher/guarded_blend_top3book_all_firstblunders_context_t05_v1 data/teacher/policyhead192_stockfish_confirmed_blunders_broad73k_top3_v1 \
  --good-action-book-top-k 3 \
  --bad-action-book data/teacher/policyhead192_stockfish_confirmed_blunders_broad73k_top3_v1 \
  --pgn-out reports/policyhead192_broad_opening_context72_blend050_stockfish_gate.pgn
```

Result: `0.0/2`. PGN file mtime: `2026-05-20T15:52:48-07:00`.

First-blunder mining found:

| Game | First confirmed mistake with context | Stockfish target | Value delta | FEN |
| --- | --- | --- | ---: | --- |
| 1 | `Qg5` | `Qa4` | `0.1520` | `r2q1rk1/p4ppp/2pb1n2/3p3b/5Q2/N2B3P/PPP2PP1/R1B1R1K1 w - - 3 14` |
| 2 | `...Bxb5` | `...Qa5+` | `0.3413` | `r2qkbnr/p2b1ppp/4p3/1N1p4/3Q1B2/8/PPP1BPPP/R3K2R b KQkq - 3 11` |

Both interpolations clear the broad holdout guard and both lose directly. The
broader supervised mix is useful for validation, but this candidate family is
still rejected for promotion.

## Aggregated Opening-Stability Source

I then aggregated all recent guarded-blend, full-network, and broad/opening
blend direct failures into a single first-blunder context source:

- PGNs: `10`
- Failed games used: `20`
- Positions: `180`
- Engine time: `0.05`
- First confirmed blunder only, `2` context plies, `4` PV plies, `2` game-line
  plies
- Output: `data/teacher/policyhead192_opening_stability_firstblunders_context_t05_v1`

I reran the broad/opening/context hard-label mix with the 180-position source in
place of the 72-position source. Epoch 2 selected under the broad guard:

| Checkpoint | Holdout top-1 | Holdout top-3 | Holdout top-5 | Stability source top-1/top-3/top-5 | Read |
| --- | ---: | ---: | ---: | ---: | --- |
| epoch 1 | `0.3446` | `0.5406` | `0.6440` | `0.3200`/`0.4800`/`0.6800` | Better top-1, top-3 below guard. |
| epoch 2 / latest | `0.3445` | `0.5421` | `0.6442` | `0.3200`/`0.5200`/`0.6800` | Selected; broader context top-3 improved. |

Direct gate:

```bash
CUDA_VISIBLE_DEVICES=0 uv run alpha-chess eval \
  --checkpoint experiments/policyhead192-distill-anchor-v1/checkpoints/broad_opening_stability180_hardlabels_cap20_lr2e6/latest.pt \
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
  --good-action-book data/teacher/stockfish_multipv_elo1800_65536_t005 data/teacher/stockfish_multipv_elo1800_8192_t05 data/teacher/policyhead192_opening_stability_firstblunders_context_t05_v1 data/teacher/policyhead192_stockfish_confirmed_blunders_broad73k_top3_v1 \
  --good-action-book-top-k 3 \
  --bad-action-book data/teacher/policyhead192_stockfish_confirmed_blunders_broad73k_top3_v1 \
  --pgn-out reports/policyhead192_opening_stability180_stockfish_gate.pgn
```

Result: `0.0/2`. PGN file mtime: `2026-05-20T16:00:05-07:00`.

First-blunder mining found:

| Game | First confirmed mistake with context | Stockfish target | Value delta | FEN |
| --- | --- | --- | ---: | --- |
| 1 | `Bd3` | `Bd4` | `0.2046` | `r2qr1k1/p3bppp/2p2nb1/3p4/8/2N1B2P/PPP1BPP1/2RQR1K1 w - - 3 15` |
| 2 | `...e5` | `...Nf6` | `0.1290` | `rnbqkbnr/pp1p1ppp/4p3/2pP4/4P3/8/PPP2PPP/RNBQKBNR b KQkq - 0 3` |

This is the best selected broad/opening/context candidate so far, but it still
fails direct play immediately. The new Black failure appears as early as ply 3,
so the next data step should broaden opening coverage rather than only append
more local first-blunder contexts.

## Higher-Time 16k Opening Source

To address the ply-3 opening failure, I generated a larger higher-time opening
window source:

```bash
uv run alpha-chess stockfish-teacher \
  --pgn data/raw/lichess_db_standard_rated_2013-01.pgn.zst \
  --out data/teacher/stockfish_opening_elo1800_16384_t05_ply24_v1 \
  --engine-path tools/stockfish/bin/stockfish \
  --engine-time 0.05 \
  --max-positions 16384 \
  --min-elo 1800 \
  --min-initial-seconds 180 \
  --multipv 4 \
  --policy-temperature-cp 180 \
  --position-stride 1 \
  --min-ply 0 \
  --max-ply 24 \
  --chunk-size 1024
```

Result: `16,384` opening-window positions.

I swapped this source into the broad/opening/stability hard-label recipe. The
raw run missed the top-3 guard, but interpolations cleared it:

| Candidate | Holdout top-1 | Holdout top-3 | Holdout top-5 | Stability source top-1/top-3/top-5 | Read |
| --- | ---: | ---: | ---: | ---: | --- |
| raw epoch 2 | `0.3447` | `0.5413` | `0.6443` | `0.4000`/`0.6000`/`0.7000` | Strong stability split, top-3 just below guard. |
| `25%` blend | `0.3452` | `0.5442` | `0.6434` | `0.2389`/`0.4167`/`0.5333` | Best holdout top-3. |
| `50%` blend | `0.3459` | `0.5430` | `0.6429` | `0.2444`/`0.4222`/`0.5333` | Best holdout top-1. |
| `75%` blend | `0.3456` | `0.5433` | `0.6439` | `0.2444`/`0.4278`/`0.5444` | Best stability metrics among the guard-passing blends. |

I gated the `75%` blend:

```bash
CUDA_VISIBLE_DEVICES=0 uv run alpha-chess eval \
  --checkpoint experiments/policyhead192-distill-anchor-v1/checkpoints/broad_opening16k_stability180_hardlabels_cap20_lr2e6_blend_0.75.pt \
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
  --good-action-book data/teacher/stockfish_multipv_elo1800_65536_t005 data/teacher/stockfish_multipv_elo1800_8192_t05 data/teacher/stockfish_opening_elo1800_16384_t05_ply24_v1 data/teacher/policyhead192_opening_stability_firstblunders_context_t05_v1 data/teacher/policyhead192_stockfish_confirmed_blunders_broad73k_top3_v1 \
  --good-action-book-top-k 3 \
  --bad-action-book data/teacher/policyhead192_stockfish_confirmed_blunders_broad73k_top3_v1 \
  --pgn-out reports/policyhead192_opening16k_stability180_blend075_stockfish_gate.pgn
```

Result: `0.0/2`. PGN file mtime: `2026-05-20T16:21:53-07:00`.

First-blunder mining found:

| Game | First confirmed mistake with context | Stockfish target | Value delta | FEN |
| --- | --- | --- | ---: | --- |
| 1 | `Bd3` | `Bd4` | `0.2360` | `r2qr1k1/p3bppp/2p2nb1/3p4/8/2N1B2P/PPP1BPP1/2RQR1K1 w - - 3 15` |
| 2 | `...Be7` | `...Bc5` | `0.1383` | `r1bqkb1r/1ppp1ppp/p1n2n2/4p3/B3P3/3P1N2/PPP2PPP/RNBQK2R b KQkq - 0 5` |

The higher-time opening source improves validation and changes the Black game,
but it still does not transfer to a nonzero direct result. The White
`Bd3`/`Bd4` motif persisted across the 180-source and 16k-opening branches.
