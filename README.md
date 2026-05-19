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

## Progress Tracker

Last updated: `2026-05-19T16:00:27-07:00`.

This repo does not yet have a calibrated Elo. The direct Stockfish gates are
small, usually 2-4 games, so a formal Elo would be misleading. The table below
uses the closest honest equivalent:

- **Parent/internal score**: candidate score against the previous checkpoint or
  a weaker smoke opponent.
- **Direct Stockfish score**: score against local Stockfish at
  `engine_time=0.05`, usually with 16 MCTS simulations and
  `material_value_weight=0.15`.
- **Elo proxy**: `400 * log10(score_rate / (1 - score_rate))` against that
  exact Stockfish gate. It is only shown for nonzero direct scores; a 0% score
  has no finite point estimate.

![Capability progress: best direct Stockfish smoke-gate score rate over time](reports/capability_progress.svg)

The graph tracks the best 4-game-or-larger direct Stockfish smoke-gate score
rate seen so far, using the actual timestamp of the first result that reached
that level. The single `0.5/2` hard-negative smoke is not included in the line
because its 4-game confirmation was `0.0/4`.

Timestamps are real `git log --date=iso-strict` commit times unless marked as a
PGN file mtime or report timestamp, where the result was generated after the
latest committed report.

| Timestamp | Milestone | Parent/internal score | Direct Stockfish score | Strength read |
| --- | --- | ---: | ---: | --- |
| `2026-05-18T08:15:09-07:00` | Baseline AlphaZero pipeline bootstrapped (`e5eb2d8`). | N/A | N/A | Infrastructure only. |
| `2026-05-18T09:14:04-07:00` | Lichess 10k expert bootstrap report (`45af7b1`, `reports/2026-05-18_expert_lichess_10k.md`). | `3.0/4` vs uniform | `0.0/2` | Stockfish path working; model clearly weak. |
| `2026-05-18T13:18:35-07:00` | Focused Stockfish MultiPV 4096 training (`0baf312`, `reports/2026-05-18_legal_multipv4096_focus.md`). | `4.0/4` vs uniform | `0.0/2`, plus `0.0/1` at higher search | Better teacher accuracy, no direct strength. |
| `2026-05-18T14:35:58-07:00` | Low-LR replay iteration promoted internally (`8fbf733`). | `3.0/4` vs base | `0.0/2` | First useful internal promotion, still loses directly. |
| `2026-05-18T18:22:01-07:00` | Puzzle-line qvalue branch (`4f1be9b`, `reports/2026-05-18_focus_puzzlelines20_vw025_material015.md`). | `6.0/8` vs base | `0.0/2`, plus 64/128-sim losses | Best fixed teacher diagnostics at that point. |
| `2026-05-18T19:59:39-07:00` | Qvalue and poisoned-capture replay (`a6021f9`, `reports/2026-05-18_qvalue_and_poisoned_replay.md`). | qvalue `8.0/8`; poisoned branch `6.0/8` vs qvalue | direct gates still failed | Strong internal qvalue parent established. |
| `2026-05-19T01:27:25-07:00` | Stockfish promotion gate documented (`d78efe0`). | N/A | gate added before promotion | Process improvement: candidates must survive direct play. |
| `2026-05-19T03:39:01-07:00` | Tree-reuse self-play PV-recent probe (`7b019cf`). | `6.0/8` vs parent | `0.0/2` | Internal wins did not transfer to Stockfish. |
| `2026-05-19T10:31:47-07:00` | Policy-head-only broad replay (`c3f1fd7`, `reports/2026-05-19_policy_head_only_broad.md`). | `8.0/16`, all draws vs qvalue | `0.5/4` | First direct Stockfish draw; tiny-sample Elo proxy about `-338` vs this gate. |
| `2026-05-19T12:11:13-07:00` | Hard-label policy-head run (`4e02a7f`, `reports/2026-05-19_policy_head_only_hardlabels.md`). | `6.0/8` vs qvalue | `0.0/4` | Better hard-label diagnostics, no direct score. |
| `2026-05-19T12:27:41-07:00` | Value-head-only calibration (`0c64101`, `reports/2026-05-19_value_head_only_hardlabels.md`). | `6.0/8` vs hard-label parent | `0.0/4` | Value fit improved, direct tactical failures remained. |
| `2026-05-19T12:40:49-07:00` | Leaf-material MCTS value blend (`7ba3097`, `reports/2026-05-19_leaf_material_mcts_probe.md`). | N/A | all tested variants `0.0/2`; broad 64-sim check `0.0/2` | Search-time material fallback did not recover the gate. |
| `2026-05-19T12:52:53-07:00` | Latest-loss replay repair (`8689a40`, `reports/2026-05-19_leaf_loss_replay_repair.md`). | `2.0/8` vs hard-label parent | `0.0/2` | Narrow loss replay overfit/regressed. |
| `2026-05-19T12:53:53-07:00` PGN file mtime | Root-mate-depth-5 inference checks (`reports/2026-05-19_rootmate5_and_16k_refresh.md`). | N/A | hard-label `0.0/2`; broad `0.0/2` | Deeper root mate filter alone was not enough. |
| `2026-05-19T13:02:42-07:00` PGN file mtime | 16k Stockfish plus latest-loss policy refresh (`reports/2026-05-19_rootmate5_and_16k_refresh.md`). | `2.0/8` vs qvalue | `0.0/2` | Broader supervised refresh also regressed. |
| `2026-05-19T13:17:29-07:00` PGN file mtime | Full-network hard-label 16k/latest-loss refresh (`reports/2026-05-19_fullhard_16k_leafloss.md`). | `2.0/8` vs qvalue | `0.0/2` | Full-network low-LR tuning also regressed. |
| `2026-05-19T13:26:47-07:00` | Root king-safety MCTS filter (`cef62fb`, `reports/2026-05-19_root_king_safety_filter.md`). | N/A | all tested variants `0.0/2` | Shallow static king-safety pruning did not recover the gate. |
| `2026-05-19T13:29:14-07:00` PGN file mtime | Policy-head broad 256-simulation follow-up (`reports/policyhead_broad_qvalue_stockfish_256sims.pgn`). | N/A | `0.0/2` | Deeper search did not reproduce the earlier Stockfish draw. |
| `2026-05-19T13:33:22-07:00` report timestamp | Policy top-k validation diagnostics (`reports/2026-05-19_policy_topk_diagnostics.md`). | N/A | N/A | Loss positions have target in top-5 about 68% of the time, so failures are partly ranking/search calibration. |
| `2026-05-19T13:36:12-07:00` report timestamp | Higher-exploration policy probe (`reports/2026-05-19_policy_exploration_probe.md`). | N/A | all tested variants `0.0/2` | Softer priors and larger `c_puct` did not recover direct play. |
| `2026-05-19T13:37:44-07:00` report timestamp | Policy-only direct probe (`reports/2026-05-19_policy_only_probe.md`). | N/A | both tested variants `0.0/2` | Search is not merely overriding a safe policy; policy-only play still blunders. |
| `2026-05-19T13:57:45-07:00` report timestamp | Hard-negative policy repair (`reports/2026-05-19_hard_negative_repair.md`). | `2.0/8` vs policy-head 16k parent | first smoke `0.5/2`; confirmation `0.0/4` | Mined top-wrong moves reduced bad-action loss but hurt top-k and did not confirm direct strength. |
| `2026-05-19T14:10:09-07:00` report timestamp | Lower-pressure hard-negative repair (`reports/2026-05-19_hard_negative_repair.md`). | `6.0/8` vs policy-head 16k parent | `0.0/2` | Lower margin pressure preserved parent strength but still failed the direct gate. |
| `2026-05-19T14:26:11-07:00` report timestamp | Mixed hard-negative repair (`reports/2026-05-19_hard_negative_repair.md`). | `8.0/8` vs policy-head 16k parent | gate `0.0/2`; confirmation `0.0/4` | Mixing broad labels preserved top-k better and dominated the parent, but still failed direct Stockfish. |
| `2026-05-19T14:30:59-07:00` report timestamp | Mixed hard-negative root-filter checks (`reports/2026-05-19_hard_negative_repair.md`). | N/A | root-material and root-king variants `0.0/2` | Existing tactical filters did not rescue the internally strong mixed checkpoint. |
| `2026-05-19T14:37:25-07:00` report timestamp | King-shelter safety filter (`reports/2026-05-19_king_shelter_filter.md`). | N/A | tested variants `0.0/2` | Pawn-shelter heuristic catches one failure pattern but still did not recover direct play. |
| `2026-05-19T14:51:35-07:00` report timestamp | Soft broad-policy hard-negative mix (`reports/2026-05-19_hard_negative_repair.md`). | `6.0/8` vs policy-head 16k parent | `0.0/2` | Keeping broad MultiPV labels soft did not materially improve hard-target top-k or direct play. |
| `2026-05-19T15:34:00-07:00` report timestamp | Direct loss-blunder replay repairs (`reports/2026-05-19_hard_negative_repair.md`). | balanced repair `4.0/8`; direct-loss mix `8.0/8` vs soft-mix parent | both `0.0/2` | A 100-position direct-loss replay improved top-3/top-5 and bad-action loss slightly, but target top-1 stayed `0.12` and direct play still failed. |
| `2026-05-19T15:48:50-07:00` report timestamp | Full-network loss-blunder repair (`reports/2026-05-19_hard_negative_repair.md`). | `4.0/8` vs direct-loss mix parent | `0.0/2` | Unfreezing the trunk reduced targeted bad-action loss and nudged target top-1 to `0.13`, but broad teacher accuracy regressed and direct play still failed. |
| `2026-05-19T16:00:27-07:00` report timestamp | Root material worst-depth guard (`reports/2026-05-19_hard_negative_repair.md`). | N/A | guarded variants all `0.0/2` | Fixed a non-monotonic material-pruning issue, but the combined root guards still did not recover direct play. |

Current practical status:

- Best direct result so far: `0.5/4` against the local Stockfish smoke gate
  from the policy-head broad run. It did not reproduce in follow-up 64-sim or
  256-sim probes, or inference-sweep probes.
- Best working parent for future experiments: the qvalue puzzle-line checkpoint
  at `experiments/focus-qvalue-puzzlelines20-vw025-material015/checkpoints/iter_0001/latest.pt`.
- Latest diagnostics show both ranking and policy-confidence failures: many
  loss positions have the Stockfish target in the policy top-5, while recent
  direct losses also include high-confidence policy top-1 blunders. A focused
  direct-loss replay improved top-3/top-5 and bad-action loss slightly, but did
  not move target top-1 or the direct Stockfish gate.
- Policy-only direct play also loses tactically, so the policy head still needs
  stronger direct-play reliability before MCTS can amplify it.
- Hard-negative mining can produce direct draws, but the first weight tried
  over-penalized wrong top moves and regressed the parent match; lower pressure
  preserved parent strength but still failed direct Stockfish.
- Mixing hard negatives with broad labels can dominate the internal parent, but
  that internal gain still does not transfer to the direct Stockfish gate.
- The optional king-shelter filter catches a recurring pawn-shield blunder
  pattern, but direct losses remain broader than that heuristic.
- Softening the broad Stockfish source did not fix the gap between internal
  parent wins and direct Stockfish play.
- Direct-loss blunder replay can also dominate the internal parent, but the
  high-confidence blunders need a stronger signal than a small policy-head-only
  fine-tune.
- Low-LR full-network fine-tuning moved the targeted margin loss more than
  policy-head-only tuning, but it regressed broad teacher accuracy and still
  failed Stockfish.
- The root material filter is now more conservative across search depths, but
  guarded direct checks still fail, so the current blocker is broader than one
  tactical root-pruning horizon issue.
- No checkpoint has passed the direct Stockfish promotion gate. This is not a
  superhuman model yet.

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

To repair policy ranking directly, mine the checkpoint's current top legal
wrong move on Stockfish-labeled replay and use that as `bad_actions`:

```bash
uv run alpha-chess hard-negatives \
  --checkpoint checkpoints/current/latest.pt \
  --data data/teacher/stockfish_sample \
  --out data/teacher/current_hard_negatives \
  --prefer-action-labels
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

When self-play policies are too sparse but the game outcomes are still useful
for value learning, keep self-play in the sample mix while disabling its policy
loss:

```bash
CHECKPOINT=experiments/current-best/checkpoints/iter_0001/latest.pt \
REPLAY_DATA="data/teacher/stockfish_multipv_elo1800_4096 data/puzzles/lines_1200_2400_100k" \
SELF_PLAY_WEIGHT=0.10 \
SELF_PLAY_POLICY_WEIGHT=0.0 \
REPLAY_WEIGHTS="0.60 0.30" \
LEGAL_POLICY_LOSS=1 \
ITERATIONS=1 GAMES=64 scripts/submit_gpu_iteration.sh
```

For policy repair runs that should not perturb the shared trunk or value head,
train only the policy head:

```bash
CHECKPOINT=experiments/current-best/checkpoints/iter_0001/latest.pt \
REPLAY_DATA="data/teacher/stockfish_multipv_elo1800_4096 data/puzzles/lines_1200_2400_100k" \
REPLAY_WEIGHTS="0.70 0.30" \
POLICY_HEAD_ONLY=1 \
VALUE_WEIGHT=0.0 \
LEGAL_POLICY_LOSS=1 \
ITERATIONS=1 GAMES=0 scripts/submit_gpu_iteration.sh
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

Use multiple evaluation workers when parent or Stockfish gates are taking too
long:

```bash
EVAL_WORKERS=4 EVAL_GAMES=16 STOCKFISH_GATE_GAMES=4 scripts/submit_gpu_iteration.sh
```

## Design Notes

AutoGo’s useful pattern is preserved: keep the game implementation deterministic, make MCTS consume a small state/evaluator interface, store each self-play position with the improved visit-count policy, and make experiments reproducible from plain scripts. Chess-specific complexity is kept behind `python-chess` so castling, en passant, promotions, repetition, and draw claims stay correct.
