# Tactical Root MCTS Filter

Date: 2026-05-18

## Change

Added a root-only tactical filter to MCTS:

- if a legal root move gives checkmate, restrict the root to mating moves
- otherwise, when safe alternatives exist, prune root moves that allow the
  opponent checkmate in one

This is a rules-based search guard, not a model change. It is meant to prevent
single-ply tactical collapses during action selection and self-play target
generation.

## Verification

Unit tests:

```text
uv run pytest
33 passed
```

Coverage added:

- MCTS selects `...Qh4#` in the Fool's mate position.
- MCTS prunes `g2-g4` after `1. f3 e5` because it allows `...Qh4#`.

## Evaluation

Checkpoint: `experiments/focus-selfplay-replay-lowlr2/checkpoints/iter_0002/latest.pt`

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
score=0.0
wins=0
draws=0
losses=2
pgn=reports/tactical_root_iter2_vs_stockfish_16sims.pgn
```

Stockfish smoke at 64 simulations:

```text
games=1
score=0.0
wins=0
draws=0
losses=1
pgn=reports/tactical_root_iter2_s64_vs_stockfish.pgn
```

## Conclusion

The filter improves basic tactical hygiene and restored the uniform smoke for
iteration 2, but it does not solve the deeper Stockfish tactical failures. The
next search-side improvement needs multi-ply tactical awareness or stronger
search/evaluation, not only mate-in-one pruning.

## Forced-Mate Extension

Extended the root filter with a small forced-mate search over checking lines:

- default root mate search depth: 3 plies
- still root-only
- detects immediate mating moves
- prunes root moves that allow the opponent a forced mate in the search window
  when alternatives exist

Verification:

```text
uv run pytest
35 passed
```

Runtime smoke:

```text
checkpoint=experiments/focus-tactical-strict/checkpoints/iter_0001/latest.pt
opponent=uniform
games=1
simulations=16
score=1.0/1
```

Stockfish smoke:

```text
checkpoint=experiments/focus-tactical-strict/checkpoints/iter_0001/latest.pt
opponent=stockfish
games=2
simulations=16
score=0.0/2
pgn=reports/forced_mate_root_strict_vs_stockfish_16sims.pgn
```

The extension is useful search hygiene but still does not cover the tactical
depth needed to survive Stockfish.
