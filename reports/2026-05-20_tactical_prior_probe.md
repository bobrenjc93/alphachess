# Root Tactical Prior Probe

Timestamp: `2026-05-20T20:27:33-07:00`

## Summary

I added an opt-in root tactical prior blend for MCTS:

- `--root-tactical-prior-weight`
- `--root-tactical-prior-temperature-cp`

The hard root guards still decide which moves are legal for the search, but the
new prior can cheaply reweight the remaining root moves using one-ply
material-quiescence plus immediate king-safety scoring. This is meant to help
positions where the Stockfish move is near the policy head's top choices but
not ranked first.

The first full depth-4 material/king-safety gate with this prior was stopped
after more than seven minutes without a completed PGN, so the prior now stays
cheap and does not reuse the full guard depth for pre-search scoring.

## Direct Smoke

I ran a comparable two-game Stockfish smoke with shallower guards so the new
prior could be tested without an impractical CPU cost:

```bash
CUDA_VISIBLE_DEVICES=1 uv run alpha-chess eval \
  --checkpoint experiments/policyhead192-distill-anchor-v1/checkpoints/broad_opening16k_stability198_probcontext_epoch2_blend_0.25.pt \
  --opponent stockfish \
  --engine-path tools/stockfish/bin/stockfish \
  --engine-time 0.05 \
  --games 2 \
  --simulations 16 \
  --device cuda \
  --material-value-weight 0.15 \
  --root-mate-search-plies 5 \
  --root-material-search-plies 2 \
  --root-material-max-loss-cp 100 \
  --root-king-safety-search-plies 1 \
  --root-king-safety-max-loss-cp 100 \
  --root-tactical-prior-weight 0.35 \
  --root-tactical-prior-temperature-cp 200 \
  --good-action-book data/teacher/stockfish_multipv_elo1800_65536_t005 data/teacher/stockfish_multipv_elo1800_8192_t05 data/teacher/stockfish_opening_elo1800_16384_t05_ply24_v1 data/teacher/policyhead192_opening_stability_firstblunders_context_t05_v1 data/teacher/policyhead192_stockfish_confirmed_blunders_broad73k_top3_v1 data/teacher/policyhead192_mat4_latest_fullgame_context_t05_v1 data/teacher/policyhead192_modelblunder_prob_bw1_firstblunders_context_t05_v1 data/teacher/policyhead192_materialopportunity_firstblunders_context_t05_v1 \
  --good-action-book-top-k 2 \
  --bad-action-book data/teacher/policyhead192_stockfish_confirmed_blunders_broad73k_top3_v1 \
  --pgn-out reports/policyhead192_stability198_tacticalprior035_shallowguards_stockfish_gate.pgn
```

Result: `0.0/2`.

The games changed substantially but still lost tactically. As White,
AlphaChess reached an exchange-heavy queenless position and was mated after
Stockfish's rook invasion. As Black, it entered a Ruy Lopez structure and lost
to a kingside attack ending in `Qxf8#`. The prior is useful infrastructure for
calibrating root priors, but the first setting is not a promotion candidate.

## Verification

- `python3 -m compileall -q src/alpha_chess`: passed
- `git diff --check`: passed
- SVG XML parse for `reports/capability_progress.svg`: passed
- `uv run pytest tests/test_mcts.py::test_mcts_root_tactical_prior_can_override_policy_prior tests/test_mcts.py::test_mcts_rejects_invalid_root_tactical_prior_settings tests/test_uci.py::test_parse_root_material_options tests/test_iteration.py::test_iteration_uses_checkpoint_self_play_workers tests/test_iteration.py::test_iteration_stockfish_gate_can_block_promotion -q`: `5 passed in 1.29s`
- `uv run pytest tests/test_mcts.py tests/test_uci.py tests/test_iteration.py tests/test_evaluate.py tests/test_cli.py tests/test_self_play.py -q`: `63 passed in 158.53s`
- `uv run pytest -q`: `133 passed in 168.98s`
