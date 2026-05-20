# Opening-Loss Repair

Timestamp: `2026-05-20T06:09:52-07:00`

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
