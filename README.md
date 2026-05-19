# AlphaChess

AlphaChess is a chess-native reproduction of the core AutoGo idea: use cheap game simulation, Monte Carlo Tree Search, and a policy/value network to automate the improvement loop.

The repo starts with a complete AlphaZero-style baseline:

- `python-chess` legal move generation and terminal adjudication.
- 64 x 73 AlphaZero chess action encoding with side-to-move orientation.
- Residual policy/value network in PyTorch.
- PUCT MCTS with root Dirichlet exploration for self-play.
- NPZ replay data, supervised policy/value training, and checkpointed evaluation.
- `gpu-dev submit` helper for running the loop on reserved GPUs.

This is not yet a superhuman model. It is the training and evaluation scaffold needed to iterate toward one.

## Setup

```bash
uv sync
uv run pytest
```

## CPU Smoke Loop

```bash
uv run alpha-chess self-play --games 2 --simulations 8 --out data/selfplay/smoke
uv run alpha-chess train --data data/selfplay/smoke --out checkpoints/smoke --epochs 1 --batch-size 8
uv run alpha-chess eval --checkpoint checkpoints/smoke/latest.pt --games 2 --simulations 8
```

When a UCI engine is installed, evaluate directly against it:

```bash
uv run alpha-chess eval --checkpoint checkpoints/run/latest.pt --opponent stockfish --engine-path stockfish --engine-time 0.05
```

Install a local ignored Stockfish binary:

```bash
scripts/install_stockfish.sh
uv run alpha-chess eval --checkpoint checkpoints/run/latest.pt --opponent stockfish --engine-path tools/stockfish/bin/stockfish --engine-time 0.05
uv run alpha-chess eval --checkpoint checkpoints/run/latest.pt --opponent stockfish --engine-path tools/stockfish/bin/stockfish --pgn-out reports/eval_games.pgn
```

Serve a checkpoint as a UCI engine:

```bash
uv run alpha-chess uci --checkpoint checkpoints/run/latest.pt --simulations 128
```

## Expert Bootstrap

Convert PGN games into sparse expert action/value targets:

```bash
uv run alpha-chess import-pgn --pgn games.pgn --out data/expert --max-games 10000
uv run alpha-chess import-pgn --pgn lichess_elite.pgn.zst --out data/expert/elite --max-games 100000
uv run alpha-chess import-pgn --pgn lichess_elite.pgn.zst --out data/expert/elite_2200 --min-elo 2200 --max-imported-games 10000
uv run alpha-chess import-pgn --pgn lichess_elite.pgn.zst --out data/expert/elite_rapid --min-elo 2000 --min-initial-seconds 180
uv run alpha-chess train --data data/expert --out checkpoints/expert --epochs 4
uv run alpha-chess train --data data/expert/elite data/teacher/tactics --out checkpoints/mixed --epochs 2
uv run alpha-chess train --data data/expert/elite data/teacher/tactics data/puzzles/mate --data-weights 0.7 0.2 0.1 --out checkpoints/mixed
uv run alpha-chess train --data data/expert/elite --out checkpoints/expert_legal --legal-policy-loss
uv run alpha-chess validate --checkpoint checkpoints/mixed/latest.pt --data data/expert/elite data/teacher/tactics data/puzzles/mate --legal-policy-loss
```

GPU pretraining from an imported expert dataset:

```bash
DATA_DIR=data/expert/lichess_2013_01_10k OUT_DIR=checkpoints/expert_10k scripts/submit_gpu_expert_train.sh
CHECKPOINT=checkpoints/expert_10k/latest.pt OUT_DIR=checkpoints/expert_10k_e2 scripts/submit_gpu_expert_train.sh
DATA_DIR="data/expert/elite data/teacher/tactics data/puzzles/mate" DATA_WEIGHTS="0.7 0.2 0.1" OUT_DIR=checkpoints/mixed scripts/submit_gpu_expert_train.sh
DATA_DIR=data/expert/elite LEGAL_POLICY_LOSS=1 OUT_DIR=checkpoints/expert_legal scripts/submit_gpu_expert_train.sh
```

Generate Stockfish teacher labels from selected PGN positions:

```bash
uv run alpha-chess stockfish-teacher \
  --pgn games.pgn.zst \
  --out data/teacher/stockfish_sample \
  --engine-path tools/stockfish/bin/stockfish \
  --max-positions 1024 \
  --min-elo 1800 \
  --min-initial-seconds 180 \
  --min-value-delta 0.25 \
  --multipv 4 \
  --policy-temperature-cp 200

uv run alpha-chess stockfish-teacher \
  --pgn reports/failed_eval_games_a.pgn reports/failed_eval_games_b.pgn \
  --out data/teacher/alpha_loss_blunders \
  --engine-path tools/stockfish/bin/stockfish \
  --player-name AlphaChess \
  --position-stride 1 \
  --min-value-delta 0.20
```

For loss-PGN repair data, include Stockfish PV continuations and the actual
game-line states Alpha entered after the sampled mistake:

```bash
uv run alpha-chess stockfish-teacher \
  --pgn reports/failed_eval_games_a.pgn reports/failed_eval_games_b.pgn \
  --out data/teacher/alpha_loss_lines \
  --engine-path tools/stockfish/bin/stockfish \
  --player-name AlphaChess \
  --position-stride 1 \
  --min-value-delta 0.08 \
  --multipv 4 \
  --policy-temperature-cp 200 \
  --pv-plies 4 \
  --game-line-plies 2
```

When `stockfish-teacher` is run with `--min-value-delta`, it also stores the
played PGN move as `bad_actions` when that move differs from Stockfish's target
move. Training can use those negative labels with a margin loss:

```bash
uv run alpha-chess train \
  --checkpoint checkpoints/current/latest.pt \
  --data data/teacher/stockfish_sample data/teacher/alpha_loss_blunders \
  --data-weights 0.8 0.2 \
  --legal-policy-loss \
  --bad-action-weight 0.25 \
  --bad-action-margin 1.0 \
  --out checkpoints/bad_action_repair
```

Training can also double FEN-backed replay with exact color-mirror symmetry:

```bash
uv run alpha-chess train \
  --checkpoint checkpoints/current/latest.pt \
  --data data/teacher/stockfish_sample data/puzzles/mate_1200_2200 \
  --legal-policy-loss \
  --color-mirror-augmentation \
  --out checkpoints/mirror_augmented
```

Import Lichess puzzle CSV tactics:

```bash
uv run alpha-chess import-puzzles \
  --puzzles lichess_db_puzzle.csv.zst \
  --out data/puzzles/mate_1200_2200 \
  --theme mate \
  --min-rating 1200 \
  --max-rating 2200 \
  --max-positions 100000
```

## GPU Training

```bash
GAMES=64 SIMULATIONS=64 EPOCHS=4 scripts/submit_gpu_training.sh
```

The script reserves a GPU with `gpu-dev submit`, syncs this repository to the worker, runs self-play, trains a checkpoint, and syncs results back into `data/` and `checkpoints/`.

For repeated AlphaZero-style improvement with promotion gating:

```bash
uv run alpha-chess iterate --run-dir experiments/run1 --iterations 4 --games 64 --simulations 64
```

Start self-play from an expert bootstrap checkpoint:

```bash
CHECKPOINT=checkpoints/expert_lichess_10k/latest.pt ITERATIONS=1 GAMES=32 LEGAL_POLICY_LOSS=1 scripts/submit_gpu_iteration.sh
```

Keep fixed teacher data in the iteration training mix:

```bash
CHECKPOINT=checkpoints/legal_multipv4096_focus_ft/latest.pt \
REPLAY_DATA="data/teacher/stockfish_multipv_elo1800_4096 data/puzzles/all_1200_2400_50k" \
REPLAY_WEIGHTS="0.45 0.10" \
SELF_PLAY_WEIGHT=0.45 \
PROMOTION_SCORE=0.55 \
MATERIAL_VALUE_WEIGHT=0.15 \
LEGAL_POLICY_LOSS=1 \
ITERATIONS=1 GAMES=32 scripts/submit_gpu_iteration.sh
```

Require candidates that pass the parent match to also score at least 50% in a
direct Stockfish smoke before promotion:

```bash
CHECKPOINT=experiments/current-best/checkpoints/iter_0001/latest.pt \
STOCKFISH_GATE_GAMES=2 \
STOCKFISH_GATE_SIMULATIONS=16 \
STOCKFISH_GATE_MIN_SCORE=0.50 \
STOCKFISH_GATE_ENGINE_PATH=tools/stockfish/bin/stockfish \
ITERATIONS=1 GAMES=32 scripts/submit_gpu_iteration.sh
```

## Design Notes

AutoGo’s useful pattern is preserved: keep the game implementation deterministic, make MCTS consume a small state/evaluator interface, store each self-play position with the improved visit-count policy, and make experiments reproducible from plain scripts. Chess-specific complexity is kept behind `python-chess` so castling, en passant, promotions, repetition, and draw claims stay correct.
