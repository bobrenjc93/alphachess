# Opening-Loss Repair

Timestamp: `2026-05-20T08:39:07-07:00`

## Summary

The latest direct gates still lose through repeated early opening/tactical
families. I added score filtering to `stockfish-teacher` so mined data can
target only AlphaChess losses:

```bash
uv run alpha-chess stockfish-teacher \
  --player-name AlphaChess \
  --player-score-max 0.0
```

Verification:

```bash
uv run pytest tests/test_stockfish_teacher.py  # 16 passed
python3 -m compileall src/alpha_chess/stockfish_teacher.py src/alpha_chess/__main__.py
```

A local PGN scan found `236` Stockfish PGNs containing `458` games and `451`
AlphaChess losses. The most common 8-ply loss prefix occurred `57` times:

```text
e2e4 e7e5 g1f3 b8c6 d2d4 e5d4 f3d4 g8f6
```

## Data

| Dataset | Sampling | Positions | Bad actions | Read |
| --- | --- | ---: | ---: | --- |
| `data/teacher/alpha_stockfish_loss_opening_context_v1` | all Stockfish PGNs, AlphaChess score `0.0`, `max_ply=20`, first blunder only, `blunder_context_plies=8` | `3,516` from `444` loss games | `424` | Wider lead-up context around the first early loss blunder. |
| `data/teacher/alpha_recent80_stockfish_loss_opening_allblunders_v1` | `80` most recent Stockfish PGNs by mtime, AlphaChess score `0.0`, `max_ply=24`, all early confirmed blunders | `4,096` from `141` recent loss games | `763` | Current-position-specific opening/tactical bad-action labels. |
| `data/teacher/alpha_recent60_opening_legalbad_v1` | `60` most recent Stockfish PGNs by mtime, AlphaChess score `0.0`, `max_ply=18`, all legal root moves scored by Stockfish, up to `8` bad actions at value drop `>=0.12` | `1,147` from `123` recent loss games | `8,705` | Dense per-position bad-action labels, averaging `7.6` bad legal moves per sampled opening position. |
| `data/teacher/alpha_recent40_opening_legalvalue_v1` | `40` most recent Stockfish PGNs by mtime, AlphaChess score `0.0`, `max_ply=18`, all legal root moves scored by Stockfish, dense legal-move value policy at temperature `0.25`, up to `8` bad actions at value drop `>=0.12` | `760` from `80` recent loss games | `5,656` | Dense value-policy targets plus bad-action labels for the latest opening failures. |
| `data/teacher/alpha_sacguard_directloss_legalvalue_v1` | sac-guard direct-loss PGN, AlphaChess score `0.0`, `max_ply=70`, dense legal-move value policy at temperature `0.25`, up to `8` bad actions at value drop `>=0.10` | `52` | `303` | Very small focused replay slice for the losses after the speculative checking-capture guard. |
| `data/teacher/alpha_guardfallback_directloss_legalvalue_v1` | guard-fallback direct-loss PGN, AlphaChess score `0.0`, `max_ply=80`, dense legal-move value policy at temperature `0.25`, up to `8` bad actions at value drop `>=0.10` | `59` | `307` | Focused replay slice for the losses after the independent root-guard fixes. |
| `data/teacher/alpha_materialfallback_directloss_legalvalue_v1` | material-fallback direct-loss PGN, AlphaChess score `0.0`, `max_ply=80`, dense legal-move value policy at temperature `0.25`, up to `8` bad actions at value drop `>=0.10` | `70` | `274` | Focused replay slice for the losses after the all-empty root-guard fallback fix. |
| `data/teacher/alpha_recent80_fullgame_legalvalue_v1` | `80` most recent Stockfish PGNs from git history, AlphaChess score `0.0`, `position_stride=2`, `max_ply=80`, all legal root moves scored by Stockfish, dense legal-move value policy at temperature `0.25`, up to `8` bad actions at value drop `>=0.10` | `2,048` from `75` loss games | `9,497` | Broader full-game legal-value and bad-action replay covering middlegame failures, not only the opening root. |

## Runs

| Run | Selection | Key settings | Disjoint holdout | Opening/loss slices | Direct gates |
| --- | --- | --- | ---: | ---: | ---: |
| `experiments/policyhead192-openingloss-context-fullnet-v1/checkpoints/iter_0001` | best opening-context bad-action loss, epoch `4` | full network, LR `7.5e-7`, bad-action weight `0.90`, weights `0.20/0.16/0.24/0.40` | top-1 `0.3356`, top-3 `0.5344`, top-5 `0.6389` | opening context top-1 `0.5796`, top-3 `0.8513`, bad-action loss `0.9588`; top-3 direct-loss bad-action loss `2.5670`; recent80 direct-loss bad-action loss `2.6469` | plain `0.0/2` (`reports/policyhead192_openingloss_context_fullnet_stockfish_gate.pgn`); book+strict `0.0/2` (`reports/policyhead192_openingloss_context_fullnet_badbook_strictguards_stockfish_gate.pgn`) |
| `experiments/policyhead192-recent-opening-allblunders-policyhead-v1/checkpoints/iter_0001` | best recent opening all-blunders bad-action loss, epoch `4` | policy head only, LR `2e-6`, bad-action weight `1.0`, weights `0.20/0.14/0.20/0.46` | top-1 `0.3429`, top-3 `0.5414`, top-5 `0.6403` | recent opening all-blunders top-1 `0.3264`, top-3 `0.5496`, top-5 `0.6909`, bad-action loss `2.1351`; recent80 direct-loss bad-action loss `2.5558`; top-3 direct-loss bad-action loss `2.6727` | plain `0.0/2` (`reports/policyhead192_recent_opening_allblunders_policyhead_stockfish_gate.pgn`); book+strict `0.0/2` (`reports/policyhead192_recent_opening_allblunders_policyhead_badbook_strictguards_stockfish_gate.pgn`) |
| `experiments/policyhead192-recent-opening-allblunders-fullnet-v1/checkpoints/iter_0001` | best recent opening all-blunders bad-action loss | full network, LR `5e-7`, bad-action weight `1.0`, weights `0.18/0.12/0.20/0.50` | top-1 `0.3370`, top-3 `0.5383`, top-5 `0.6384` | recent opening all-blunders top-1 `0.3149`, top-3 `0.5571`, top-5 `0.6765`, bad-action loss `2.0906`; recent80 direct-loss top-1 `0.2434`, top-3 `0.4644`, top-5 `0.5750`, bad-action loss `2.5354`; top-3 direct-loss top-1 `0.2130`, top-3 `0.4060`, top-5 `0.5138`, bad-action loss `2.6586` | plain `0.0/2` (`reports/policyhead192_recent_opening_allblunders_fullnet_stockfish_gate.pgn`); book+strict `0.0/2` (`reports/policyhead192_recent_opening_allblunders_fullnet_badbook_strictguards_stockfish_gate.pgn`) |
| `experiments/policyhead192-recent-opening-legalbad-policyhead-v1/checkpoints/iter_0001` | best dense legal-bad-action loss | policy head only from top-3 fullnet parent, LR `1.5e-6`, bad-action weight `1.2`, weights `0.16/0.12/0.18/0.18/0.36` | top-1 `0.3361`, top-3 `0.5313`, top-5 `0.6359` | dense legal-bad top-1 `0.4987`, top-3 `0.7489`, top-5 `0.8509`, bad-action loss `0.1769`; recent all-blunders bad-action loss `1.9832`; recent80 direct-loss bad-action loss `2.3248`; top-3 direct-loss bad-action loss `2.4977` | plain `0.0/2` (`reports/policyhead192_recent_opening_legalbad_policyhead_stockfish_gate.pgn`); book+strict `0.0/2` (`reports/policyhead192_recent_opening_legalbad_policyhead_badbook_strictguards_stockfish_gate.pgn`) |
| `experiments/policyhead192-recent-opening-legalvalue-policyhead-v1/checkpoints/iter_0001` | best dense legal-value policy loss | policy head only from top-3 fullnet parent, LR `1.2e-6`, bad-action weight `0.8`, weights `0.20/0.14/0.18/0.34/0.14` | top-1 `0.3356`, top-3 `0.5288`, top-5 `0.6345` | legal-value top-1 `0.4908`, top-3 `0.7447`, top-5 `0.8461`, bad-action loss `0.1980`; dense legal-bad bad-action loss `0.1874`; recent all-blunders bad-action loss `2.0366`; recent80 direct-loss bad-action loss `2.3648` | plain `0.0/2` (`reports/policyhead192_recent_opening_legalvalue_policyhead_stockfish_gate.pgn`); book+strict `0.0/2` (`reports/policyhead192_recent_opening_legalvalue_policyhead_badbook_strictguards_stockfish_gate.pgn`) |
| `experiments/policyhead192-recent-opening-legalvalue-policyhead-v1/checkpoints/iter_0001` | capture-starting mate-search guard | same checkpoint, same legal-value bad-action book, strict guards, `root_mate_search_plies=5` with mate recursion over checks plus high-priority captures/promotions | N/A | root regression fixed: the latest gate's `...axb3 Kc1 Ne2+ Kb1 Rd1#` family is detected; a depth-7 root check prunes the earlier `Ne3` move but is too slow for full-game gates without more pruning | book+strict `0.0/2` (`reports/policyhead192_recent_opening_legalvalue_policyhead_badbook_capturemate5_stockfish_gate.pgn`) |
| `experiments/policyhead192-recent-opening-legalvalue-policyhead-v1/checkpoints/iter_0001` | exact Stockfish good-action book | same checkpoint, legal-value good-action book plus bad-action book, strict guards | N/A | restored the exact teacher opening `e4 e5 Nf3 Nc6 d4`, proving the book can override overzealous root heuristics at known positions | good+bad book strict `0.0/2` (`reports/policyhead192_recent_opening_legalvalue_policyhead_goodbadbook_strict_stockfish_gate.pgn`) |
| `experiments/policyhead192-recent-opening-legalvalue-policyhead-v1/checkpoints/iter_0001` | broad exact teacher books | same checkpoint, broad Stockfish good-action book plus recent loss/legal-value/top-3 exact books and strict guards | N/A | expanded exact-position coverage changed the mid-opening but still left uncovered tactical collapses | broad-good+bad book strict `0.0/2` (`reports/policyhead192_recent_opening_legalvalue_policyhead_broadgoodbooks_strict_stockfish_gate.pgn`) |
| `experiments/policyhead192-recent-opening-legalvalue-policyhead-v1/checkpoints/iter_0001` | speculative checking-capture guard | same checkpoint and broad exact books, plus near-best material fallback for checking captures and a king-recapture penalty for speculative checking captures | N/A | avoided the previous `Bxh7+` line, but the replacement games still lost tactically | broad-good+bad book strict `0.0/2` (`reports/policyhead192_recent_opening_legalvalue_policyhead_broadgoodbooks_sacguard_stockfish_gate.pgn`) |
| `experiments/policyhead192-sacguard-directloss-policyhead-v1/checkpoints/iter_0001` | best sac-guard bad-action loss, epoch `5` | policy head only from legal-value parent, LR `1.2e-6`, bad-action weight `1.0`, weights `0.20/0.25/0.55` over broad/legal-value/sac-guard slices | broad holdout top-1 `0.3336`, top-3 `0.5313`, top-5 `0.6329` | sac-guard slice top-1 `0.3654`, top-3 `0.6154`, top-5 `0.7308`, bad-action loss `0.3012` vs parent `0.3309` | broad-good+bad book strict `0.0/2` (`reports/policyhead192_sacguard_directloss_policyhead_broadgoodbooks_stockfish_gate.pgn`) |
| `experiments/policyhead192-sacguard-directloss-policyhead-v1/checkpoints/iter_0001` | speculative-capture veto before root guards | same checkpoint and books, but king-recapturable non-pawn checking captures are removed before king/material guards can collapse the root | N/A | removed the repeated `Bxh7+` family; first follow-up still lost centrally, and the second exposed that material fallback could starve king-safety on `...Kd7` | broad-good+bad book strict `0.0/2` (`reports/policyhead192_sacguard_directloss_policyhead_broadgoodbooks_sacfilter_stockfish_gate.pgn`) |
| `experiments/policyhead192-sacguard-directloss-policyhead-v1/checkpoints/iter_0001` | king-safety before material fallback | same checkpoint and books, with king-safety run before material fallback | N/A | removed the `...Kd7` king-walk failure, but king-safety then collapsed one white root to `Qxh7+` before material could veto it | broad-good+bad book strict `0.0/2` (`reports/policyhead192_sacguard_directloss_policyhead_broadgoodbooks_kingfirst_stockfish_gate.pgn`) |
| `experiments/policyhead192-sacguard-directloss-policyhead-v1/checkpoints/iter_0001` | speculative-capture veto plus king-first guard order | same checkpoint and books, with speculative checking captures vetoed before both guards and king-safety run before material fallback | N/A | removed the `Bxh7+`, `Qxh7+`, and `...Kd7` root-filter failures, but the latest games still lost through nearby opening tactics | broad-good+bad book strict `0.0/2` (`reports/policyhead192_sacguard_directloss_policyhead_broadgoodbooks_sacfilter2_stockfish_gate.pgn`) |
| `experiments/policyhead192-sacguard-directloss-policyhead-v1/checkpoints/iter_0001` | independent material and king-safety safe sets | same checkpoint and books, with material and king-safety each evaluated on the pre-fallback root set before combining safe actions | N/A | removed the latest `...Nxf2` singleton-guard failure; remaining games lost through broader tactical exchanges and promotion pressure | broad-good+bad book strict `0.0/2` (`reports/policyhead192_sacguard_directloss_policyhead_broadgoodbooks_independentguards_stockfish_gate.pgn`) |
| `experiments/policyhead192-sacguard-directloss-policyhead-v1/checkpoints/iter_0001` | material fallback vetoes disjoint king-safe sacrifices | same checkpoint and books, with material fallback preferred when material has no threshold-safe move and the fallback is not only king moves | N/A | removed the `...Rxg4+` disjoint-safe-set exchange sacrifice; remaining games lost through broader attacking and promotion lines | broad-good+bad book strict `0.0/2` (`reports/policyhead192_sacguard_directloss_policyhead_broadgoodbooks_guardfallback_stockfish_gate.pgn`) |
| `experiments/policyhead192-guardfallback-directloss-policyhead-v1/checkpoints/iter_0001` | best guard-fallback bad-action loss, epoch `3` | policy head only from sac-guard parent, LR `1.0e-6`, bad-action weight `1.0`, weights `0.18/0.22/0.22/0.38` over broad/legal-value/sac-guard/guard-fallback slices | broad holdout top-1 `0.3324`, top-3 `0.5317`, top-5 `0.6334` | guard-fallback slice top-1 `0.4068`, top-3 `0.5085`, top-5 `0.6441`, bad-action loss `0.6428` vs parent `0.6754` | broad-good+bad book strict `0.0/2` (`reports/policyhead192_guardfallback_directloss_policyhead_broadgoodbooks_stockfish_gate.pgn`) |
| `experiments/policyhead192-guardfallback-directloss-policyhead-v1/checkpoints/iter_0001` | material fallback preferred when both guards find no safe move | same checkpoint and books, with material fallback preferred over king-safety fallback when both threshold-safe sets are empty and the material fallback is not only king moves | N/A | removed the follow-up all-empty-fallback `...Nxf2` sacrifice; remaining games lost through broader material, king-safety, and passed-pawn failures | broad-good+bad book strict `0.0/2` (`reports/policyhead192_guardfallback_directloss_policyhead_broadgoodbooks_materialfallback_stockfish_gate.pgn`) |
| `experiments/policyhead192-materialfallback-directloss-policyhead-v1/checkpoints/iter_0001` | best material-fallback bad-action loss, epoch `5` | policy head only from guard-fallback parent, LR `5e-7`, bad-action weight `0.6`, weights `0.34/0.22/0.14/0.14/0.16` over broad/legal-value/sac-guard/guard-fallback/material-fallback slices | broad holdout top-1 `0.3386`, top-3 `0.5360`, top-5 `0.6360` vs parent top-1 `0.3383`, top-3 `0.5365`, top-5 `0.6367` | material-fallback slice top-1 `0.3857`, top-3 `0.5857`, top-5 `0.7143`, bad-action loss `0.3811` vs parent `0.3893` | broad-good+bad book strict `0.0/2` (`reports/policyhead192_materialfallback_directloss_policyhead_broadgoodbooks_stockfish_gate.pgn`) |
| `experiments/policyhead192-recent80-fullgame-legalvalue-policyhead-v1/checkpoints/iter_0001` | best recent80 full-game bad-action loss, epoch `5` | policy head only from material-fallback parent, LR `7.5e-7`, bad-action weight `0.7`, weights `0.40/0.20/0.40` over broad/opening/full-game slices | broad holdout top-1 `0.3387`, top-3 `0.5353`, top-5 `0.6373` vs parent top-1 `0.3386`, top-3 `0.5360`, top-5 `0.6360` | recent80 full-game top-1 `0.3638`, top-3 `0.5952`, top-5 `0.7095`, bad-action loss `0.3806` vs parent top-1 `0.3652`, top-3 `0.5806`, top-5 `0.7041`, bad-action loss `0.3908` | broad-good+bad book strict `0.0/2` (`reports/policyhead192_recent80_fullgame_legalvalue_policyhead_broadgoodbooks_stockfish_gate.pgn`) |
| `experiments/policyhead192-recent80-fullgame-legalvalue-policyhead-v1/checkpoints/iter_0001` | 64-simulation direct search check | same checkpoint and books, same guards, `simulations=64` instead of `16` | N/A | deeper search changed both games but still found losing middlegame lines | 64-sim broad-good+bad book strict `0.0/2` (`reports/policyhead192_recent80_fullgame_legalvalue_policyhead_broadgoodbooks_64sims_stockfish_gate.pgn`) |

## Read

Loss-only opening mining improves the targeted replay metrics but still does
not transfer to direct Stockfish play. The exact bad-action books changed the
opening choices, which means the mined labels do affect move selection, but
Stockfish still finds adjacent tactical collapses. The current blocker is not
just one repeated bad move in the common `e4 e5 Nf3 Nc6 d4` stem; the model and
search need stronger tactical credit assignment across nearby opening families.
The full-network all-blunders follow-up reduced the targeted all-blunders
bad-action loss slightly versus policy-head-only tuning, but it worsened the
broad holdout and still did not transfer to direct play. The exact-book gate
again changed openings, then failed through nearby tactical collapses.
The dense all-legal bad-action sweep is a stronger local signal than played-move
replay: it moved recent direct-loss bad-action loss from `2.7664` on the parent
to `2.3248`. That still did not transfer to direct play, and the disjoint
holdout top-1 slipped from `0.3395` to `0.3361`, so the model is still learning
local avoidance without reliable tactical generalization.
The legal-value policy target improved its own policy loss from `6.5092` on the
parent to `5.0559` and kept recent direct-loss bad-action loss better than the
parent (`2.3648` vs `2.7708`), but it regressed broad holdout top-1 to `0.3356`
and still lost both plain and strict direct gates. Dense root-value labels are
useful diagnostics, but policy-head replay alone is not enough.
The root mate guard now follows checks plus high-priority captures/promotions,
so it recognizes forced mates whose first attacking move is not a check. That
fixes a concrete failure family from the latest gate, but the practical
depth-5 book-plus-strict Stockfish check still lost `0.0/2`. A depth-7 root
check catches the earlier losing `Ne3` position from the same game, but full
direct gates at that depth are too slow without a separate search budget or
more selective candidate pruning.
Exact good-action books are now available at evaluation time and can force the
Stockfish teacher's best exact-position move before material/king-safety
heuristics run. On the legal-value opening book this restored the common
teacher line, but both direct games still failed after the book stopped
covering the position. The current gap is therefore no longer just the root
heuristics rejecting known-good openings; the policy/search stack still needs
general tactical reliability after the exact teacher table ends.
Combining broader exact books from the 65k Stockfish teacher, recent loss
openings, legal-value data, and top-3 confirmed blunders also failed `0.0/2`.
That makes exact-position table coverage a diagnostic aid, not a path to the
current direct gate by itself.
The root material guard previously forced the `Bxh7+` speculative sacrifice
because every root move scored below the strict material threshold and the
fallback kept only the single highest material score. I changed that fallback to
keep a small near-best band only for checking-capture sacrifices and added a
penalty when the opponent king can immediately recapture the attacker. This
fixes the concrete `Bxh7+` root-filter failure, but the direct gate still lost
both games through other tactics.
The follow-up sac-guard replay improved the tiny targeted slice, reducing
bad-action loss from `0.3309` to `0.3012` and lifting top-3 from `0.5769` to
`0.6154`, but broad holdout top-1 slipped from `0.3356` to `0.3336`. The direct
gate remained `0.0/2` and even found a nearby `Bxh7+` sacrifice from a different
position, so exact replay of the latest loss is still too local to solve the
opening-tactical failure family.
The root-filter follow-up fixed concrete guard interactions rather than model
strength. Moving the speculative checking-capture veto before the guard stack
removed the repeated bishop sacrifice, but the first comparison still lost and
exposed a `...Kd7` king-walk line. Running king-safety before material fallback
fixed that singleton-fallback failure, but then king-safety alone collapsed a
different root to `Qxh7+`. The final ordering vetoes king-recapturable checking
captures before both guards and then applies king-safety before material. That
fixes the observed `Bxh7+`, `Qxh7+`, and `...Kd7` root-filter failures, but the
comparable direct gate still scored `0.0/2`, so the remaining opening collapses
are broader than this guard-order bug.
The next gate showed the same sequencing problem from the other direction:
king-safety collapsed the root to a non-checking `...Nxf2` sacrifice before
material could reject it. I changed the combined guard path so material and
king-safety each produce safe actions from the same pre-fallback root set, then
intersect or fall back to whichever guard has real safe actions. That removed
the `...Nxf2` failure, but the comparable direct gate remained `0.0/2`; the
losses shifted to broader tactical exchanges and passed-pawn/promotion pressure.
The following comparison showed that returning the union of disjoint material
and king-safety safe sets was still too permissive: a king-safe but
material-losing `...Rxg4+` survived. I changed that disagreement case to prefer
material's fallback unless the fallback is only king moves, preserving the
`...Kd7` fix while vetoing the exchange sacrifice. The direct gate still scored
`0.0/2`; the new losses are broader attacking and promotion failures, not the
specific singleton-root guard bugs fixed here.
Mining those latest losses produced a 59-position dense legal-value slice with
307 bad-action labels. A policy-head-only repair lowered that slice's
bad-action loss from `0.6754` to `0.6428`, but it also regressed the disjoint
broad holdout top-1 from `0.3336` to `0.3324` and the target slice top-1 from
`0.4237` to `0.4068`. The direct gate still scored `0.0/2`, so this is another
local replay fit without practical transfer.
The next guard check exposed one remaining fallback tie case: when both
threshold-safe sets were empty, the old sequential path could still return
king-safety's fallback sacrifice. Preferring the material fallback in that
all-empty case removed the repeated `...Nxf2` family from the latest direct
gate, while still allowing king-only material fallbacks to preserve the earlier
`...Kd7` fix. The comparable Stockfish gate remained `0.0/2`; the new games
shifted to broader exchange, king-safety, and passed-pawn failures.
Mining the material-fallback losses produced a 70-position dense legal-value
slice with 274 bad-action labels. A deliberately lower-pressure policy-head
repair improved that slice's bad-action loss from `0.3893` to `0.3811` and kept
the broad holdout essentially flat-to-slightly-up (`0.3383` to `0.3386`
top-1). That still did not transfer to direct play: the next gate scored
`0.0/2`, with losses moving to new `Rxb2`, passed-pawn, and king-attack
families rather than the exact `...Nxf2` root fallback.
The broader full-game legal-value replay is a better diagnostic slice than the
single-gate repairs: it mined 2,048 positions and 9,497 bad-action labels from
75 recent AlphaChess losses through ply 80. Policy-head tuning on that slice
improved bad-action loss (`0.3908` to `0.3806`) and lifted top-3/top-5 while
leaving broad holdout top-1 flat. The direct gate still scored `0.0/2`, so the
current bottleneck is not just lack of exact middlegame labels; the policy and
search still fail to convert those local labels into robust direct play.
Raising that checkpoint's direct gate from 16 to 64 MCTS simulations also
scored `0.0/2`. The game lines changed, but both still lost tactically, which
argues against this candidate being one small search-budget increase away from
the current Stockfish gate.
