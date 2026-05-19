# MCTS Deterministic Tie-Break

Date: 2026-05-19

## Change

Deterministic `SearchResult` move selection now breaks equal visit-count ties by
root child prior instead of by action id. If every root child has zero visits,
temperature-zero policy selection also falls back to the highest-prior root
child instead of returning an empty policy.

The previous behavior was arbitrary for deterministic play: `np.argmax` on the
visit array selected the lowest action index among tied moves.

## Reproduction

Latest rejected checkpoint:

`experiments/focus-openingmultipv12-pvrecent-low-vw025-material015/checkpoints/iter_0002/latest.pt`

Position from the 64-simulation Stockfish loss after:

`1. e4 e5 2. Nf3 Nc6 3. d4 exd4 4. Nxd4 Nf6 5. Nxc6 bxc6 6. e5 Qe7`

FEN:

`r1b1kb1r/p1ppqppp/2p2n2/4P3/8/8/PPP2PPP/RNBQKB1R w KQkq - 1 7`

Before this change, `7.Nc3` and `7.Qe2` both received `12` visits at
64 simulations, and deterministic selection chose `7.Nc3` because its action id
was lower. That game continued with `7...Qxe5+` and ended in `13...Qh3#`.

After the change, the same search selects `7.Qe2`, because its root prior is
`0.1781` versus `0.1353` for `7.Nc3`.

## Direct Smokes

The selector fix is not sufficient to recover the Stockfish gate:

| Config | Score | PGN |
| --- | ---: | --- |
| patched selector, default root guards | `0.0/2` | `reports/focus_openingmultipv12_iter2_tiebreak_stockfish_16sims.pgn` |
| existing root material guard, `plies=2`, `max_loss_cp=250` | `0.0/2` | `reports/focus_openingmultipv12_iter2_rootmaterial250_stockfish_16sims.pgn` |

## Tests

`uv run pytest tests/test_mcts.py`

Result: `14 passed`

## Conclusion

Keep the fix. It removes an arbitrary deterministic-play artifact and makes
zero-simulation deterministic selection usable, but the direct-play failure mode
remains deeper than root visit tie-breaking.
