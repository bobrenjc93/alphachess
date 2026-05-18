# Expert-Initialized Self-Play Smoke

## Starting Point

- Initial checkpoint: `checkpoints/expert_lichess_10k/latest.pt`
- Run directory: `experiments/expert_selfplay_smoke`
- GPU: 1x L4 via `gpu-dev`

## Iteration Settings

- Iterations: 1
- Self-play games: 4
- MCTS simulations per move: 16
- Max plies: 120
- Training epochs: 1
- Batch size: 64
- Model: 128 channels, 6 residual blocks

## Promotion Gate

Candidate checkpoint:

```text
experiments/expert_selfplay_smoke/checkpoints/iter_0001/latest.pt
```

Evaluation against the starting expert checkpoint:

```text
games=4
score=2.0
score_rate=0.5
wins=0
draws=4
losses=0
promoted=true
```

This is only a smoke-scale RL iteration. It verifies that checkpoint-initialized self-play, candidate training, incumbent evaluation, and promotion all work end-to-end on GPU. It is not evidence of superhuman strength.
