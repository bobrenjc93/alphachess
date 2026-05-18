# Low-LR Replay-Mixed Self-Play Iteration

Date: 2026-05-18

## GPU Iteration

- Base checkpoint: `checkpoints/legal_multipv4096_focus_ft/latest.pt`
- Candidate checkpoint: `experiments/focus-selfplay-replay-lowlr2/checkpoints/iter_0001/latest.pt`
- Runner: `gpu-dev submit`, reservation `09479966`, 1x L4
- Self-play: 16 games, 32 simulations, 160 max plies
- Training: 1 epoch, batch size 128, learning rate `0.0001`
- Loss: `--legal-policy-loss`
- Data weights:
  - accumulated self-play total weight `0.10`
  - `data/teacher/stockfish_multipv_elo1800_4096`, weight `0.60`
  - `data/puzzles/all_1200_2400_50k`, weight `0.10`
  - `data/teacher/alpha_loss_reports`, weight `0.20`

Promotion gate against the base checkpoint:

```text
score=3.0/4
wins=2
draws=2
losses=0
score_rate=0.75
promoted=true
```

Final checkpoint training metrics:

```text
loss=1.6874
policy_loss=1.5971
policy_acc=0.6032
value_loss=0.0903
epoch_loss=1.8125
val_loss=2.7626
val_policy_loss=2.5254
val_policy_acc=0.3475
val_value_loss=0.2372
```

Training split source metrics:

```text
val_source_0_policy_acc=0.6635  # self-play, 211 examples
val_source_1_policy_acc=0.4147  # Stockfish MultiPV 4096, 434 examples
val_source_2_policy_acc=0.3278  # all puzzles 50k, 4969 examples
val_source_3_policy_acc=1.0000  # AlphaChess loss replay, 3 examples
```

## Diagnostic Validation

```bash
uv run alpha-chess validate \
  --checkpoint experiments/focus-selfplay-replay-lowlr2/checkpoints/iter_0001/latest.pt \
  --data data/teacher/stockfish_multipv_elo1800_4096 data/puzzles/all_1200_2400_50k data/teacher/alpha_loss_reports \
  --batch-size 256 \
  --legal-policy-loss \
  --device cpu
```

Result:

```text
val_policy_acc=0.3384
val_source_0_policy_acc=0.4829  # Stockfish MultiPV 4096
val_source_1_policy_acc=0.3263  # all puzzles 50k
val_source_2_policy_acc=1.0000  # AlphaChess loss replay
```

Same-data baseline from `checkpoints/legal_multipv4096_focus_ft/latest.pt`:

```text
val_policy_acc=0.3267
val_source_0_policy_acc=0.4448  # Stockfish MultiPV 4096
val_source_1_policy_acc=0.3167  # all puzzles 50k
val_source_2_policy_acc=1.0000  # AlphaChess loss replay
```

## Evaluation

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
pgn=reports/focus_selfplay_replay_lowlr2_vs_stockfish_16sims.pgn
```

Stockfish smoke at 64 simulations:

```text
games=1
score=0.0
wins=0
draws=0
losses=1
pgn=reports/focus_selfplay_replay_lowlr2_s64_vs_stockfish.pgn
```

The 64-simulation loss dropped material in the opening after `4. Bc4 Nxe4`
and was mated on move 23.

## Iteration 2

Continued the same league for one more iteration from the promoted iteration-1
checkpoint.

- Candidate checkpoint: `experiments/focus-selfplay-replay-lowlr2/checkpoints/iter_0002/latest.pt`
- Runner: `gpu-dev submit`, reservation `a5de3fe8`, 1x L4
- Self-play: 16 additional games, 32 simulations, 160 max plies
- Training: 1 epoch, batch size 128, learning rate `0.0001`

Promotion gate against iteration 1:

```text
score=2.0/4
wins=0
draws=4
losses=0
score_rate=0.50
promoted=true
```

Final checkpoint training metrics:

```text
loss=1.6504
policy_loss=1.6025
policy_acc=0.5663
value_loss=0.0479
epoch_loss=1.6527
val_loss=2.7057
val_policy_loss=2.4977
val_policy_acc=0.3612
val_value_loss=0.2080
```

Diagnostic validation:

```text
val_policy_acc=0.3492
val_source_0_policy_acc=0.5371  # Stockfish MultiPV 4096
val_source_1_policy_acc=0.3335  # all puzzles 50k
val_source_2_policy_acc=1.0000  # AlphaChess loss replay
```

Evaluation:

```text
uniform: score=3.0/4, wins=2, draws=2, losses=0
stockfish 16 sims: score=0.0/2, wins=0, draws=0, losses=2
stockfish 64 sims: score=0.0/1, wins=0, draws=0, losses=1
```

PGNs:

```text
reports/focus_selfplay_replay_lowlr2_iter2_vs_stockfish_16sims.pgn
reports/focus_selfplay_replay_lowlr2_iter2_s64_vs_stockfish.pgn
```

Iteration 2 improved the fixed Stockfish MultiPV diagnostic again but did not
improve practical play against Stockfish, and it became less decisive against
the uniform baseline.

## Iteration 3: Replay-Only Loss Refresh

Generated a v2 AlphaChess loss-replay set from all current Stockfish failure
PGNs:

```bash
uv run alpha-chess stockfish-teacher \
  --pgn reports/*stockfish*.pgn \
  --out data/teacher/alpha_loss_reports_v2 \
  --engine-path tools/stockfish/bin/stockfish \
  --engine-time 0.01 \
  --max-positions 256 \
  --player-name AlphaChess \
  --min-value-delta 0.15 \
  --multipv 4 \
  --policy-temperature-cp 200 \
  --position-stride 1 \
  --chunk-size 64
```

Result: 61 correction positions across 28 games, with mean policy support of
about 3.95 legal moves.

Then continued the same league with no new self-play:

- Candidate checkpoint: `experiments/focus-selfplay-replay-lowlr2/checkpoints/iter_0003/latest.pt`
- Runner: `gpu-dev submit`, reservation `bb37bf08`, 1x L4
- Self-play: 0 games
- Training: 1 epoch, batch size 128, learning rate `0.00005`
- Data weights:
  - accumulated self-play total weight `0.0`
  - `data/teacher/stockfish_multipv_elo1800_4096`, weight `0.55`
  - `data/puzzles/all_1200_2400_50k`, weight `0.10`
  - `data/teacher/alpha_loss_reports_v2`, weight `0.35`

Promotion gate against iteration 2:

```text
score=1.0/4
wins=0
draws=2
losses=2
score_rate=0.25
promoted=false
```

Diagnostic validation:

```text
val_policy_acc=0.3445
val_source_0_policy_acc=0.5369  # Stockfish MultiPV 4096
val_source_1_policy_acc=0.3282  # all puzzles 50k
val_source_2_policy_acc=0.7869  # AlphaChess loss replay v2
```

This fit the new loss-replay data better, but it did not improve head-to-head
play and should not replace iteration 2.

## Iteration 4: Tactical-Filtered Self-Play

After adding root tactical filtering in MCTS, continued the league from the
iteration-2 best checkpoint. Empty self-play dirs from replay-only iteration 3
were filtered out before weighted training.

- Candidate checkpoint: `experiments/focus-selfplay-replay-lowlr2/checkpoints/iter_0004/latest.pt`
- Runner: `gpu-dev submit`, reservation `117adf64`, 1x L4
- Self-play: 16 games, 32 simulations, 160 max plies
- Training: 1 epoch, batch size 128, learning rate `0.00005`
- Data weights:
  - accumulated self-play total weight `0.10`
  - `data/teacher/stockfish_multipv_elo1800_4096`, weight `0.55`
  - `data/puzzles/all_1200_2400_50k`, weight `0.10`
  - `data/teacher/alpha_loss_reports_v2`, weight `0.25`

Promotion gate against iteration 2:

```text
score=2.0/4
wins=0
draws=4
losses=0
score_rate=0.50
promoted=true
```

Diagnostic validation:

```text
val_policy_acc=0.3486
val_source_0_policy_acc=0.5349  # Stockfish MultiPV 4096
val_source_1_policy_acc=0.3328  # all puzzles 50k
val_source_2_policy_acc=0.7541  # AlphaChess loss replay v2
```

Evaluation:

```text
uniform: score=4.0/4, wins=4, draws=0, losses=0
stockfish 16 sims: score=0.0/2, wins=0, draws=0, losses=2
stockfish 64 sims: score=0.0/1, wins=0, draws=0, losses=1
```

PGNs:

```text
reports/tactical_iter4_vs_stockfish_16sims.pgn
reports/tactical_iter4_s64_vs_stockfish.pgn
```

Iteration 4 is the current league best and restored the uniform gate, but it
still fails every Stockfish gate.

## Conclusion

Replay-mixed low-learning-rate iteration is now improving the fixed Stockfish
MultiPV diagnostic and can promote within the local checkpoint league, but it is
still far below the Stockfish gate. Iteration 4 is the current internal league
best; iteration 2 remains marginally stronger on the 4k MultiPV diagnostic;
iteration 3 is a rejected replay-only fine-tune. None is a solved model.
