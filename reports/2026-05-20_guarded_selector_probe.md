# Guarded Composite Selector Probe

Timestamp: `2026-05-20T14:04:52-07:00`

## Summary

I added guarded composite checkpoint selection and ran two short policy-head
continuations to test the failure mode that showed up in the engine self-play
probe: policy loss or top-1 can improve while broad top-k ranking regresses.
The selector behaved as intended. Neither run met the broad holdout top-3 floor,
so they produced epoch checkpoints but no selected `latest.pt`, and I did not
spend a direct Stockfish gate on either.

## Parent Baseline

```bash
CUDA_VISIBLE_DEVICES=0 uv run alpha-chess validate \
  --checkpoint experiments/policyhead192-stockfish-confirmed-broad73k-top3-fullnet-v1/checkpoints/iter_0001/latest.pt \
  --data data/teacher/stockfish_multipv_elo1800_holdout8192_t005_skip65536 \
  --batch-size 512 \
  --value-weight 0.05 \
  --bad-action-weight 0.3 \
  --bad-action-margin 1.0 \
  --legal-policy-loss \
  --device cuda
```

Parent broad holdout:

| Metric | Value |
| --- | ---: |
| policy loss | `3.7873` |
| top-1 | `0.3395` |
| top-3 | `0.5421` |
| top-5 | `0.6425` |

## Training

```bash
CUDA_VISIBLE_DEVICES=0 uv run alpha-chess train \
  --checkpoint experiments/policyhead192-stockfish-confirmed-broad73k-top3-fullnet-v1/checkpoints/iter_0001/latest.pt \
  --data data/teacher/stockfish_multipv_elo1800_65536_t005 \
  --holdout-data data/teacher/stockfish_multipv_elo1800_holdout8192_t005_skip65536 \
  --out experiments/policyhead192-broad65k-guarded-composite-v1/checkpoints/iter_0001 \
  --epochs 5 \
  --batch-size 512 \
  --lr 1e-7 \
  --weight-decay 1e-4 \
  --value-weight 0.05 \
  --bad-action-weight 0.3 \
  --bad-action-margin 1.0 \
  --legal-policy-loss \
  --policy-head-only \
  --select-best-by holdout_policy_acc+holdout_policy_top3_acc \
  --select-best-require 'holdout_policy_acc>=0.3394' 'holdout_policy_top3_acc>=0.5420' \
  --device cuda
```

The command ended with:

```text
ValueError: No epoch satisfied select_best_by 'holdout_policy_acc+holdout_policy_top3_acc' with requirements holdout_policy_acc>=0.3394 holdout_policy_top3_acc>=0.5420
```

## Epoch Metrics

| Epoch | Holdout top-1 | Holdout top-3 | Holdout top-5 | Holdout policy loss | Eligible |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 | `0.3392` | `0.5388` | `0.6398` | `3.5951` | no |
| 2 | `0.3387` | `0.5386` | `0.6401` | `3.6139` | no |
| 3 | `0.3395` | `0.5386` | `0.6404` | `3.6365` | no |
| 4 | `0.3392` | `0.5381` | `0.6395` | `3.5842` | no |
| 5 | `0.3392` | `0.5386` | `0.6396` | `3.5962` | no |

## Read

This is a useful negative result. The broad-only continuation lowered policy
loss by about `0.15-0.20`, but it damaged top-3/top-5 ranking on the disjoint
holdout. That is exactly the kind of checkpoint the earlier single-metric or
loss-led workflow could accidentally gate. The guarded selector should be used
for the next mixed-source GPU run so the direct Stockfish gate only sees
checkpoints that preserve broad ranking metrics.

## Mixed-Source Follow-Up

I also tried a low-LR mixed-source policy-head pass with the same broad holdout
floors:

```bash
CUDA_VISIBLE_DEVICES=0 uv run alpha-chess train \
  --checkpoint experiments/policyhead192-stockfish-confirmed-broad73k-top3-fullnet-v1/checkpoints/iter_0001/latest.pt \
  --data data/teacher/stockfish_multipv_elo1800_65536_t005 data/teacher/stockfish_multipv_elo1800_8192_t05 data/teacher/policyhead192_stockfish_confirmed_blunders_broad73k_top3_v1 data/teacher/alpha_recent80_firstblunder_context_v1 \
  --holdout-data data/teacher/stockfish_multipv_elo1800_holdout8192_t005_skip65536 \
  --out experiments/policyhead192-guarded-mix-v1/checkpoints/iter_0001 \
  --epochs 5 \
  --batch-size 512 \
  --lr 5e-8 \
  --weight-decay 1e-4 \
  --value-weight 0.05 \
  --bad-action-weight 0.3 \
  --bad-action-margin 1.0 \
  --data-weights 0.70 0.10 0.10 0.10 \
  --legal-policy-loss \
  --policy-head-only \
  --select-best-by holdout_policy_acc+holdout_policy_top3_acc \
  --select-best-require 'holdout_policy_acc>=0.3394' 'holdout_policy_top3_acc>=0.5420' \
  --device cuda
```

This also ended with no selected epoch.

| Epoch | Holdout top-1 | Holdout top-3 | Holdout top-5 | Holdout policy loss | Eligible |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 | `0.3398` | `0.5377` | `0.6390` | `3.6270` | no |
| 2 | `0.3396` | `0.5381` | `0.6390` | `3.6228` | no |
| 3 | `0.3397` | `0.5380` | `0.6385` | `3.6281` | no |
| 4 | `0.3395` | `0.5377` | `0.6393` | `3.6207` | no |
| 5 | `0.3395` | `0.5378` | `0.6388` | `3.6243` | no |

This is sharper than the broad-only run: top-1 was at or slightly above the
parent baseline in several epochs, but top-3 stayed well below the `0.5420`
floor. That confirms top-1 alone is not a sufficient selector for the current
teacher mix.

## Blend Follow-Up

I then blended the parent with the best-looking rejected epochs. This is cheap
and directly tests whether a small step toward the candidate can keep the
parent's top-k ranking while taking the lower-loss/top-1 signal.

| Source checkpoint | Weight on candidate | Holdout top-1 | Holdout top-3 | Holdout top-5 | Holdout policy loss | Read |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| mixed-source epoch 1 | `0.05` | `0.3398` | `0.5420` | `0.6422` | `3.7775` | near guard |
| mixed-source epoch 1 | `0.10` | `0.3397` | `0.5421` | `0.6417` | `3.7680` | preserves top-3 |
| mixed-source epoch 1 | `0.20` | `0.3398` | `0.5421` | `0.6412` | `3.7496` | preserves top-3 |
| mixed-source epoch 1 | `0.30` | `0.3401` | `0.5406` | `0.6403` | `3.7321` | top-3 regresses |
| broad-only epoch 3 | `0.05` | `0.3402` | `0.5422` | `0.6421` | `3.7779` | clears guard |
| broad-only epoch 3 | `0.10` | `0.3401` | `0.5426` | `0.6422` | `3.7688` | best guard-passing blend |
| broad-only epoch 3 | `0.20` | `0.3401` | `0.5414` | `0.6410` | `3.7512` | top-3 regresses |
| broad-only epoch 3 | `0.30` | `0.3398` | `0.5410` | `0.6407` | `3.7345` | top-3 regresses |

The `broad-only epoch 3` blend at `0.10` was the best checkpoint from this
sequence: it improved disjoint broad top-1 from `0.3395` to `0.3401`, top-3
from `0.5421` to `0.5426`, kept top-5 roughly flat, and lowered policy loss.
The first direct gate attempt was interrupted when the persistent GPU
reservation was canceled, so I recreated the same broad-only epoch and blend on
a temporary H100 host. The validation metrics reproduced exactly.

## Direct Gate

```bash
CUDA_VISIBLE_DEVICES=0 uv run alpha-chess eval \
  --checkpoint experiments/policyhead192-guarded-blends-v1/checkpoints/broad_epoch3_w0.10.pt \
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
  --good-action-book data/teacher/stockfish_multipv_elo1800_65536_t005 data/teacher/stockfish_multipv_elo1800_8192_t05 \
  --bad-action-book data/teacher/policyhead192_stockfish_confirmed_blunders_broad73k_top3_v1 \
  --pgn-out reports/policyhead192_guarded_blend_broad_epoch3_w010_stockfish_gate.pgn
```

Result: `{'games': 2.0, 'score': 0.0, 'score_rate': 0.0, 'wins': 0.0,
'draws': 0.0, 'losses': 2.0}`.

PGN file mtime: `2026-05-20T13:34:42-07:00`.

The first loss followed the familiar `e4 e5 Nf3 Nc6 d4` stem but collapsed
after allowing a passed `d` pawn to queen with mate. The second loss came from a
Sicilian line where AlphaChess accepted early queenside material and then lost
to a forcing attack ending in `Qa3#`. So the broad top-k guard and small blend
step produce a better supervised candidate, but still do not solve the direct
tactical reliability problem.

## First-Blunder Mining

```bash
uv run alpha-chess stockfish-teacher \
  --pgn reports/policyhead192_guarded_blend_broad_epoch3_w010_stockfish_gate.pgn \
  --out data/teacher/guarded_blend_gate_firstblunders_v1 \
  --engine-path tools/stockfish/bin/stockfish \
  --player-name AlphaChess \
  --position-stride 1 \
  --min-value-delta 0.08 \
  --multipv 4 \
  --policy-temperature-cp 180 \
  --first-blunder-only \
  --pv-plies 4 \
  --game-line-plies 2 \
  --chunk-size 256
```

Result: `14` positions from both games.

| Game | First confirmed mistake | Stockfish target | Value delta | FEN |
| --- | --- | --- | ---: | --- |
| 1 | `Re1` | `Na3` | `0.4704` | `1rbqr1k1/p1p1bppp/5n2/3p2B1/8/2PB4/PP1Q1PPP/RN3RK1 w - - 1 12` |
| 2 | `...Nxa2+` | `...Rf8` | `0.3231` | `rnbqk2r/pp1p1pQp/4p3/4P3/1b1N4/2n5/PPP2PPP/R1B1KB1R b KQkq - 0 8` |

I reran the same miner with `--blunder-context-plies 2` because the game 1
mistake is a quiet lead-up move, not the final passed-pawn tactic:

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

Result: `18` positions from both games.

| Game | First confirmed mistake with context | Stockfish target | Value delta | FEN |
| --- | --- | --- | ---: | --- |
| 1 | `Re1` | `Na3` | `0.4887` | `1rbqr1k1/p1p1bppp/5n2/3p2B1/8/2PB4/PP1Q1PPP/RN3RK1 w - - 1 12` |
| 2 | `...Nxc3` | `...Rf8` | `0.1123` | `rnbqk2r/pp1p1ppp/4p3/3nP3/1b1N2Q1/2N5/PPP2PPP/R1B1KB1R b KQkq - 2 7` |

The context run moves game 2's first confirmed mistake earlier than the final
`...Nxa2+` material grab. The next repair should focus on these lead-up states
instead of only the final mate/promotion positions.

## Context-Repair Probe

I tried a very small policy-head repair from the guard-passing blend using the
18-position context slice mixed lightly with the broad teacher:

```bash
CUDA_VISIBLE_DEVICES=0 uv run alpha-chess train \
  --checkpoint experiments/policyhead192-guarded-blends-v1/checkpoints/broad_epoch3_w0.10.pt \
  --data data/teacher/stockfish_multipv_elo1800_65536_t005 data/teacher/guarded_blend_gate_firstblunders_context_v1 \
  --holdout-data data/teacher/stockfish_multipv_elo1800_holdout8192_t005_skip65536 \
  --out experiments/policyhead192-guarded-blend-contextrepair-v1/checkpoints/iter_0001 \
  --epochs 3 \
  --batch-size 512 \
  --lr 5e-8 \
  --weight-decay 1e-4 \
  --value-weight 0.05 \
  --bad-action-weight 0.3 \
  --bad-action-margin 1.0 \
  --data-weights 0.95 0.05 \
  --legal-policy-loss \
  --policy-head-only \
  --select-best-by holdout_policy_acc+holdout_policy_top3_acc \
  --select-best-require 'holdout_policy_acc>=0.3400' 'holdout_policy_top3_acc>=0.5420' \
  --device cuda
```

All epochs were rejected:

| Epoch | Holdout top-1 | Holdout top-3 | Holdout top-5 | Holdout policy loss | Eligible |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 | `0.3395` | `0.5383` | `0.6393` | `3.6087` | no |
| 2 | `0.3392` | `0.5382` | `0.6401` | `3.6183` | no |
| 3 | `0.3391` | `0.5385` | `0.6394` | `3.6363` | no |

Even a `5%` context slice weight immediately erased the blend's top-3 gain.
This reinforces that narrow loss repair needs either a much lower effective
weight or a different objective; otherwise it destroys the broad ranking signal
before direct play.

## Source-Repeat Cap Follow-Up

I added `--max-source-repeat` to cap how often tiny repair sources are sampled
per epoch. This prevents an 18-position slice from being replayed thousands of
times when it receives nonzero `--data-weights` mass.

The capped version used the same context repair but limited the tiny source to
at most five expected samples per position per epoch:

```bash
CUDA_VISIBLE_DEVICES=0 uv run alpha-chess train \
  --checkpoint experiments/policyhead192-guarded-blends-v1/checkpoints/broad_epoch3_w0.10.pt \
  --data data/teacher/stockfish_multipv_elo1800_65536_t005 data/teacher/guarded_blend_gate_firstblunders_context_v1 \
  --holdout-data data/teacher/stockfish_multipv_elo1800_holdout8192_t005_skip65536 \
  --out experiments/policyhead192-guarded-blend-contextrepair-capped-v1/checkpoints/iter_0001 \
  --epochs 5 \
  --batch-size 512 \
  --lr 5e-8 \
  --weight-decay 1e-4 \
  --value-weight 0.05 \
  --bad-action-weight 0.3 \
  --bad-action-margin 1.0 \
  --data-weights 0.95 0.05 \
  --max-source-repeat 5 \
  --legal-policy-loss \
  --policy-head-only \
  --select-best-by holdout_policy_acc+holdout_policy_top3_acc \
  --select-best-require 'holdout_policy_acc>=0.3400' 'holdout_policy_top3_acc>=0.5420' \
  --device cuda
```

All epochs were still rejected:

| Epoch | Holdout top-1 | Holdout top-3 | Holdout top-5 | Holdout policy loss | Eligible |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 | `0.3386` | `0.5385` | `0.6401` | `3.6263` | no |
| 2 | `0.3391` | `0.5386` | `0.6401` | `3.6176` | no |
| 3 | `0.3390` | `0.5385` | `0.6396` | `3.5897` | no |
| 4 | `0.3395` | `0.5385` | `0.6398` | `3.5906` | no |
| 5 | `0.3394` | `0.5381` | `0.6401` | `3.5855` | no |

The cap fixes the sampling mechanics, but this run still continues broad
training from the blend toward the rejected broad-only epoch, so it loses the
top-3 gain again.

I also tried a single low-LR pass over only the 18 context positions:

```bash
CUDA_VISIBLE_DEVICES=0 uv run alpha-chess train \
  --checkpoint experiments/policyhead192-guarded-blends-v1/checkpoints/broad_epoch3_w0.10.pt \
  --data data/teacher/guarded_blend_gate_firstblunders_context_v1 \
  --holdout-data data/teacher/stockfish_multipv_elo1800_holdout8192_t005_skip65536 \
  --out experiments/policyhead192-guarded-blend-contextonly-tiny-v1/checkpoints/iter_0001 \
  --epochs 1 \
  --batch-size 18 \
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

That was much less destructive but still missed the guard: top-1 `0.3395`,
top-3 `0.5420`, top-5 `0.6417`, policy loss `3.7816`. The practical read is
that first-blunder repair needs either a different objective or an explicit
interpolation step after repair; direct fine-tuning from the blend is not enough.
