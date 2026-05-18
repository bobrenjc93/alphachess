# Material Value Prior

Date: 2026-05-18

## Change

Added an opt-in material value blend for neural evaluators:

```text
value = (1 - material_value_weight) * neural_value
      + material_value_weight * material_value
```

The material value is from the side-to-move perspective and is squashed with
`tanh(material_cp / 1200)`. Default weight is `0.0`, so baseline behavior is
unchanged unless `--material-value-weight` is supplied.

Wired through:

- `alpha-chess self-play`
- `alpha-chess eval`
- `alpha-chess iterate`
- UCI config construction
- `scripts/submit_gpu_iteration.sh` as `MATERIAL_VALUE_WEIGHT`

## Verification

```text
uv run pytest
36 passed
```

CLI help exposes `--material-value-weight` for self-play, eval, and iterate.

## Evaluation

Checkpoint: `experiments/focus-forced-mate-strict/checkpoints/iter_0001/latest.pt`

Material weight: `0.15`

Uniform opponent:

```text
games=4
score=4.0
wins=4
draws=0
losses=0
```

Stockfish smoke at 16 simulations:

```text
games=2
score=0.5
wins=0
draws=1
losses=1
pgn=reports/material015_forced_mate_strict_vs_stockfish_16sims.pgn
```

Stockfish smoke at 64 simulations:

```text
games=1
score=0.0
wins=0
draws=0
losses=1
pgn=reports/material015_forced_mate_strict_s64_vs_stockfish.pgn
```

## Conclusion

The material prior produced the first nonzero Stockfish smoke result in this
sequence while keeping uniform play intact. It is not enough for the higher
search Stockfish gate, but it is a useful inference and self-play option for
reducing obvious material collapse.
