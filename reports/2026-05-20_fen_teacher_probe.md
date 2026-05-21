# FEN Teacher Probe

Timestamp: `2026-05-20T17:25:52-07:00`

## Summary

I added direct FEN input to `stockfish-teacher` so targeted failure positions
can be labeled without reconstructing them through PGNs. This is meant for the
latest failure pattern where exact PGN-line books fix one move and the model
immediately fails in an adjacent out-of-book branch.

The new input file is:

- `reports/2026-05-20_latest_failure_fens.txt`
- 6 recent first-blunder FENs from the material-depth-4, top-2 exact-book, and
  latest-source repair gates

The generator now accepts one FEN per non-comment line:

```bash
uv run alpha-chess stockfish-teacher \
  --fen-file reports/2026-05-20_latest_failure_fens.txt \
  --out data/teacher/policyhead192_latest_failure_fens_legalvalue_t20_v1 \
  --engine-path tools/stockfish/bin/stockfish \
  --engine-time 0.2 \
  --multipv 4 \
  --policy-temperature-cp 180 \
  --legal-bad-actions-per-position 8 \
  --legal-bad-action-min-delta 0.08 \
  --legal-value-policy-temperature 0.2 \
  --max-positions 64 \
  --chunk-size 64
```

Result:

- `6` positions
- `48` bad-action labels
- `226` legal candidates evaluated
- output: `data/teacher/policyhead192_latest_failure_fens_legalvalue_t20_v1`

## Label Stability

I first generated the same FEN source at `engine_time=0.05`. It worked
mechanically, but the labels were too noisy for use as an exact book: several
best moves disagreed with the earlier first-blunder mining targets. Regenerating
at `engine_time=0.2` was more stable on the newest failures:

| Row | FEN source | t20 best action | t20 top policy moves | Read |
| --- | --- | --- | --- | --- |
| 0 | mat4 White `e5`/`exd5` | `e5` | `e5`, `Nc3`, `exd5`, `Nd2`, `Qe2` | still conflicts with earlier first-blunder target, so not exact-book safe |
| 1 | mat4 Black `...Na5`/`...Nxe5` | `Nxe5` | `Nxe5`, `Bb7`, `Na5`, `Nb8`, `Na7` | filters the played `...Na5` at top-2 |
| 2 | top-2 book White `f3`/`b3` | `f4` | `f4`, `Be3`, `b3`, `Nd2`, `Nc3` | conflicts with earlier target; use only as dense policy signal |
| 3 | top-2 book Black `...Kh8`/`...c5` | `c5` | `c5`, `h6`, `c6`, `Re8`, `Be7` | filters the played `...Kh8` at top-2 |
| 4 | repair White `Bc4`/`Bxe4` | `Bxe4` | `Bxe4`, `O-O`, `Bg5`, `Bd2`, `Ba6` | matches latest target and filters `Bc4` |
| 5 | repair Black `...Bd5`/`...h6` | `Bc7` | `Bc7`, `h6`, `Qd7`, `Bb8`, `Qc7` | `h6` is near the top, but best move changed at higher time |

The practical conclusion is that FEN input is useful infrastructure, but a tiny
six-position source should not be used as an eval-time exact book. It is better
used to create dense legal-value supervision or to seed a broader branch dataset
with enough engine time to avoid target instability.

## Baseline Validation

Validation of the current opening16k/stability `75%` blend on the t20 FEN
source, using dense legal policies:

| Slice | Top-1 | Top-3 | Top-5 | Policy loss | Bad-action loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| broad holdout, soft labels | `0.3398` | `0.5383` | `0.6395` | `3.6053` | `0.0000` |
| latest failure FEN legal-value t20 | `0.3333` | `0.5000` | `0.6667` | `5.2709` | `0.3167` |

This source is not enough by itself to justify a direct gate. The next useful
step is to expand from these FENs into a larger branch set rather than replaying
the same six positions.

## Verification

- `uv run pytest tests/test_stockfish_teacher.py -q`: `17 passed`
- `python3 -m compileall -q src/alpha_chess`: passed
- `uv run pytest -q`: `123 passed in 94.11s`
