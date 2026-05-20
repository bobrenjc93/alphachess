# Stockfish-Confirmed Model-Blunder Mining

Timestamp: `2026-05-20T00:22:48-07:00`

## Summary

The previous hard-negative runs penalized moves that disagreed with the teacher
target. This probe adds a sharper miner: take the model's top legal move, and
only store it as a bad action when Stockfish confirms that playing it drops the
root value by at least a configured threshold.

The mined data is a better diagnostic signal than label disagreement alone:
`2,010` confirmed blunders were found in `16,384` broad/t05 teacher positions.
However, a small policy-head-only repair reduced the confirmed-blunder loss only
slightly and still scored `0.0/2` against Stockfish.

## Code change

New CLI:

```bash
uv run alpha-chess model-blunders \
  --checkpoint CHECKPOINT \
  --data TEACHER_DIR... \
  --out OUT_DIR \
  --engine-path tools/stockfish/bin/stockfish \
  --engine-time 0.02 \
  --max-positions 16384 \
  --min-value-delta 0.08 \
  --bad-actions-per-position 1 \
  --prefer-action-labels
```

The miner writes replay NPZ files containing the teacher action as `actions`,
the model move as `bad_actions`, and a padded `value_deltas` array for the
Stockfish-confirmed drops.

Verification:

```bash
uv run pytest
```

Result: `94 passed`.

## Mined data

- Checkpoint:
  `experiments/policyhead192-hardneg-broadrepair-v1/checkpoints/iter_0001/latest.pt`
- Data:
  `data/teacher/stockfish_multipv_elo1800_8192_t05`,
  `data/teacher/stockfish_multipv_elo1800_65536_t005`
- Output: `data/teacher/policyhead192_stockfish_confirmed_blunders_v1`
- Engine: `tools/stockfish/bin/stockfish`
- Engine time: `0.02`
- Max positions: `16384`
- Min value delta: `0.08`

Summary:

| Metric | Value |
| --- | ---: |
| positions seen | `16384` |
| model-wrong positions | `4540` |
| Stockfish-confirmed blunder positions | `2010` |
| bad actions | `2010` |
| value-delta min | `0.0802` |
| value-delta mean | `0.2725` |
| value-delta max | `1.9042` |

Baseline validation of the starting checkpoint:

| Source | Policy acc | Top-3 | Top-5 | Bad-action loss |
| --- | ---: | ---: | ---: | ---: |
| confirmed blunders | `0.0000` | `0.3343` | `0.5005` | `4.0346` |
| disjoint holdout | `0.3442` | `0.5459` | `0.6464` | N/A |

## Repair run

- Experiment:
  `experiments/policyhead192-stockfish-confirmed-blunder-repair-v1/checkpoints/iter_0001`
- Start checkpoint:
  `experiments/policyhead192-hardneg-broadrepair-v1/checkpoints/iter_0001/latest.pt`
- Training data weights:
  broad65k `0.43`, t05 `0.24`, confirmed blunders `0.23`,
  fullnet loss slice `0.05`, latest loss slice `0.05`
- Epochs: `3`
- LR: `4e-6`
- Bad-action weight: `0.20`
- Selection metric: `holdout_policy_acc`
- Selected epoch: `1`

Selected checkpoint metrics:

| Source | Policy acc | Top-3 | Top-5 | Bad-action loss |
| --- | ---: | ---: | ---: | ---: |
| disjoint holdout | `0.3438` | `0.5459` | `0.6472` | N/A |
| confirmed blunders | `0.0000` | `0.3803` | `0.5070` | `3.9276` |

Direct gate:

| Check | Result | Artifact |
| --- | ---: | --- |
| Stockfish gate | `0.0/2` | `reports/policyhead192_stockfish_confirmed_blunder_repair_gate.pgn` |

## H100 full-network follow-ups

The policy-head-only repair did not move target top-1, so I used the active
H100 reservation to try two trunk-unfrozen repairs from the same parent.

I then mined the full broad/t05 teacher pool:
`73,728` positions scanned, `11,161` model-top disagreements, and `4,687`
Stockfish-confirmed value-dropping model moves written to
`data/teacher/policyhead192_stockfish_confirmed_blunders_broad73k_v1`.
A top-3 variant on the same pool found `8,001` positions and `16,268`
Stockfish-confirmed bad-action labels in
`data/teacher/policyhead192_stockfish_confirmed_blunders_broad73k_top3_v1`.

| Run | Selection | Key settings | Disjoint holdout | Confirmed blunders | Direct gate |
| --- | --- | --- | ---: | ---: | ---: |
| `experiments/policyhead192-stockfish-confirmed-fullnet-v1/checkpoints/iter_0001` | best `holdout_policy_acc`, epoch `3` | LR `1e-6`, bad-action weight `0.20`, weights `0.43/0.24/0.23/0.05/0.05` | top-1 `0.3446`, top-3 `0.5470`, top-5 `0.6483` | top-1 `0.0199`, top-3 `0.3423`, top-5 `0.5075`, bad-action loss `4.0469` | `0.0/2` (`reports/policyhead192_stockfish_confirmed_fullnet_gate.pgn`) |
| `experiments/policyhead192-stockfish-confirmed-fullnet-badmargin-v1/checkpoints/iter_0001` | best confirmed-blunder bad-action loss, epoch `5` | LR `7.5e-7`, bad-action weight `0.50`, weights `0.40/0.22/0.28/0.05/0.05` | top-1 `0.3439`, top-3 `0.5470`, top-5 `0.6476` | top-1 `0.0209`, top-3 `0.3458`, top-5 `0.5095`, bad-action loss `4.0162` | `0.0/2` (`reports/policyhead192_stockfish_confirmed_fullnet_badmargin_gate.pgn`) |
| `experiments/policyhead192-stockfish-confirmed-broad73k-fullnet-v1/checkpoints/iter_0001` | best broad73k confirmed-blunder bad-action loss, epoch `4` | LR `7.5e-7`, bad-action weight `0.45`, weights `0.38/0.22/0.30/0.05/0.05` | top-1 `0.3440`, top-3 `0.5472`, top-5 `0.6488` | top-1 `0.0365`, top-3 `0.4137`, top-5 `0.5790`, bad-action loss `3.6055` | `0.0/2` (`reports/policyhead192_stockfish_confirmed_broad73k_fullnet_gate.pgn`) |
| `experiments/policyhead192-stockfish-confirmed-broad73k-top3-fullnet-v1/checkpoints/iter_0001` | best top-3 confirmed-blunder bad-action loss, epoch `3` | LR `7.5e-7`, bad-action weight `0.55`, weights `0.36/0.20/0.34/0.05/0.05` | top-1 `0.3452`, top-3 `0.5472`, top-5 `0.6464` | top-1 `0.0346`, top-3 `0.5354`, top-5 `0.6670`, bad-action loss `2.4040` | `0.0/2` (`reports/policyhead192_broad73k_top3_fullnet_stockfish_gate.pgn`) |

## Exact bad-action book

I added an optional eval-time `--bad-action-book` path that loads mined replay
NPZs, keys positions by FEN without move counters, and suppresses exact matched
bad actions only at AlphaChess root moves. Defaults are unchanged, and the
filter falls back to the original legal move list if every root action would be
removed.

Tests: `uv run pytest` passed with `96 passed`.

| Check | Bad-action book | Direct gate |
| --- | --- | ---: |
| hard-negative broad/t05 parent | `data/teacher/policyhead192_stockfish_confirmed_blunders_broad73k_v1` | `0.0/2` (`reports/policyhead192_hardneg_broad73k_badbook_stockfish_gate.pgn`) |
| broad73k full-network repair | `data/teacher/policyhead192_stockfish_confirmed_blunders_broad73k_v1` | `0.0/2` (`reports/policyhead192_broad73k_fullnet_badbook_stockfish_gate.pgn`) |
| broad73k full-network repair plus strict root guards | `data/teacher/policyhead192_stockfish_confirmed_blunders_broad73k_v1` | `0.0/2` (`reports/policyhead192_broad73k_fullnet_badbook_strictguards_stockfish_gate.pgn`) |
| broad73k full-network repair plus strict root guards, `64` simulations | `data/teacher/policyhead192_stockfish_confirmed_blunders_broad73k_v1` | `0.0/2` (`reports/policyhead192_broad73k_fullnet_badbook_strictguards_64sims_stockfish_gate.pgn`) |
| top-3 broad73k full-network repair plus strict root guards | `data/teacher/policyhead192_stockfish_confirmed_blunders_broad73k_top3_v1` | `0.0/2` (`reports/policyhead192_broad73k_top3_fullnet_badbook_strictguards_stockfish_gate.pgn`) |

## Direct-loss PV follow-up

The latest top-3 full-network gates still failed tactically, so I generated a
fresh Stockfish teacher slice from those direct-loss PGNs:

- Sources:
  `reports/policyhead192_broad73k_top3_fullnet_stockfish_gate.pgn`,
  `reports/policyhead192_broad73k_top3_fullnet_badbook_strictguards_stockfish_gate.pgn`,
  `reports/policyhead192_broad73k_fullnet_badbook_strictguards_64sims_stockfish_gate.pgn`
- Output: `data/teacher/policyhead192_top3_directloss_pv_v1`
- Engine: `tools/stockfish/bin/stockfish`
- Engine time: `0.05`, `multipv=8`, `player_name=AlphaChess`,
  `position_stride=1`, `pv_plies=4`, `game_line_plies=2`
- Positions: `399` from `6` games
- Stockfish-confirmed played bad actions: `57`

Baseline for the top-3 full-network parent on this new direct-loss slice:
top-1 `0.2155`, top-3 `0.4085`, top-5 `0.5238`, bad-action loss `2.9992`.

| Run | Selection | Key settings | Disjoint holdout | Direct-loss slice | Direct gates |
| --- | --- | --- | ---: | ---: | ---: |
| `experiments/policyhead192-top3-directloss-fullnet-v1/checkpoints/iter_0001` | best direct-loss bad-action loss, epoch `4` | full network, LR `5e-7`, bad-action weight `0.75`, weights `0.28/0.17/0.25/0.30` | top-1 `0.3440`, top-3 `0.5474`, top-5 `0.6454` | top-1 `0.2155`, top-3 `0.4236`, top-5 `0.5263`, bad-action loss `2.9056` | plain `0.0/2` (`reports/policyhead192_top3_directloss_fullnet_stockfish_gate.pgn`); book+strict `0.0/2` (`reports/policyhead192_top3_directloss_fullnet_badbook_strictguards_stockfish_gate.pgn`) |
| `experiments/policyhead192-top3-directloss-policyhead-v1/checkpoints/iter_0001` | best direct-loss bad-action loss, epoch `6` | policy head only, LR `2e-6`, bad-action weight `1.25`, weights `0.22/0.13/0.20/0.45` | top-1 `0.3428`, top-3 `0.5455`, top-5 `0.6431` | top-1 `0.2256`, top-3 `0.4361`, top-5 `0.5439`, bad-action loss `2.7075` | plain `0.0/2` (`reports/policyhead192_top3_directloss_policyhead_stockfish_gate.pgn`); book+strict `0.0/2` (`reports/policyhead192_top3_directloss_policyhead_badbook_strictguards_stockfish_gate.pgn`) |

I also expanded the same process to every tracked Stockfish PGN available on
the H100 checkout. This produced `4,096` positions from `86` games with `780`
Stockfish-confirmed bad played actions in
`data/teacher/alpha_all_directloss_pv_v1`.
Because that run filled its cap from alphabetically early PGNs, I then repeated
the mining from the `80` most recent local PGN mtimes. That recency-biased set
produced `4,096` positions from `93` games with `772` bad played actions in
`data/teacher/alpha_recent80_directloss_pv_v1`.
After adding `stockfish-teacher --blunder-context-plies`, I repeated the
recency-biased run with two lead-up AlphaChess decision positions attached to
each confirmed blunder. That context set produced `4,096` positions from `70`
games with `550` bad played actions in
`data/teacher/alpha_recent80_directloss_context_v1`.

| Run | Selection | Key settings | Disjoint holdout | All-history direct-loss slice | Latest direct-loss slice | Direct gates |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `experiments/policyhead192-all-directloss-fullnet-v1/checkpoints/iter_0001` | best all-history direct-loss bad-action loss, epoch `4` | full network, LR `7.5e-7`, bad-action weight `0.85`, weights `0.25/0.15/0.25/0.35` | top-1 `0.3430`, top-3 `0.5436`, top-5 `0.6437` | top-1 `0.2925`, top-3 `0.5066`, top-5 `0.6299`, bad-action loss `1.7075` | top-1 `0.2130`, top-3 `0.4135`, top-5 `0.5138`, bad-action loss `2.9852` | plain `0.0/2` (`reports/policyhead192_all_directloss_fullnet_stockfish_gate.pgn`); book+strict `0.0/2` (`reports/policyhead192_all_directloss_fullnet_badbook_strictguards_stockfish_gate.pgn`) |
| `experiments/policyhead192-recent-directloss-fullnet-v1/checkpoints/iter_0001` | best recent direct-loss bad-action loss, epoch `4` | full network, LR `7.5e-7`, bad-action weight `0.85`, weights `0.25/0.15/0.25/0.35` | top-1 `0.3420`, top-3 `0.5424`, top-5 `0.6439` | top-1 `0.2451`, top-3 `0.4661`, top-5 `0.5854`, bad-action loss `2.6682` | top-1 `0.2130`, top-3 `0.4160`, top-5 `0.5163`, bad-action loss `2.9265` | plain `0.0/2` (`reports/policyhead192_recent_directloss_fullnet_stockfish_gate.pgn`); book+strict `0.0/2` (`reports/policyhead192_recent_directloss_fullnet_badbook_strictguards_stockfish_gate.pgn`) |
| `experiments/policyhead192-context-directloss-fullnet-v1/checkpoints/iter_0001` | best context direct-loss bad-action loss, epoch `4` | full network, LR `7.5e-7`, bad-action weight `0.85`, weights `0.25/0.15/0.25/0.35`, `blunder_context_plies=2` | top-1 `0.3420`, top-3 `0.5444`, top-5 `0.6438` | context slice top-1 `0.2761`, top-3 `0.5015`, top-5 `0.6157`, bad-action loss `2.8421`; recent80 slice top-1 `0.2449`, top-3 `0.4675`, top-5 `0.5815`, bad-action loss `2.6712` | top-1 `0.2105`, top-3 `0.4135`, top-5 `0.5163`, bad-action loss `2.9300` | plain `0.0/2` (`reports/policyhead192_context_directloss_fullnet_stockfish_gate.pgn`); book+strict `0.0/2` (`reports/policyhead192_context_directloss_fullnet_badbook_strictguards_stockfish_gate.pgn`) |

## Read

Stockfish-confirmed model-blunder mining is useful because it separates true
value-dropping model choices from harmless label disagreements. The first repair
attempt shows the current policy-head-only recipe is too weak: it moves top-k
and bad-action loss slightly, but target top-1 remains `0.0` on the mined set
and direct play is unchanged.

The H100 full-network follow-ups did get confirmed-blunder target top-1 off
zero, but only to about `2%`. The bad-margin selector also improved the mined
bad-action margin relative to the parent, while slightly regressing the disjoint
holdout. Both still scored `0.0/2` against Stockfish, so this signal is useful
diagnostically but is not yet strong enough as a small replay slice.

The broad73k pass more than doubled the confirmed-blunder slice and moved top-1
further, to `3.65%`, while keeping the broad holdout roughly flat. It still did
not improve direct play, and the margin loss on the broad slice did not beat the
parent, so the next repair needs either a different objective or direct
move-selection use of this signal rather than another small supervised replay.
The top-3 variant gave the best disjoint holdout top-1 so far (`0.3452`) and
substantially better mined top-3/top-5 ranking, but it also failed direct play.
The first exact-position move-selection use did not cover enough of the direct
loss lines to change the gate. Combining the book with stricter mate,
material, and king-safety root filters also failed, including at `64` MCTS
simulations on the H100.
Fresh direct-loss PV replay from those failures improved the new direct-loss
diagnostics, especially with an aggressive policy-head-only repair, but it
regressed the disjoint broad holdout and still scored `0.0/2` in both plain and
book-plus-strict direct Stockfish gates. The current blocker is therefore not
solved by a small replay of the latest failure PGNs.
Scaling that replay to all tracked Stockfish losses fit the broader historical
failure slice much more strongly, but it did not improve the latest-loss slice
and still scored `0.0/2`. Historical direct-loss replay is therefore not enough
unless the sampling and objective are made more current-position specific.
Recency-biased sampling improved the current slice slightly more than the
alphabetic all-history run, but still regressed broad holdout and did not move
direct play. The next useful direction is likely a stronger search/labeling
change around the forcing tactical sequences, not more small supervised replay.
Backfilling lead-up context positions made the replay data less narrowly tied to
only the final blunder, but it still did not transfer to direct Stockfish play.
The failure is therefore not just missing the previous two AlphaChess decisions
around each logged mistake; the policy/search stack still needs stronger
tactical credit assignment or a materially better search target.
