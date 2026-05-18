"""AlphaZero-style self-play / train / promote iteration driver."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from alpha_chess.evaluate import evaluate_checkpoint
from alpha_chess.evaluator import UniformEvaluator, load_evaluator
from alpha_chess.self_play import SelfPlayConfig, generate_self_play
from alpha_chess.train import TrainConfig, train


@dataclass
class IterationConfig:
    run_dir: str = "experiments/local"
    iterations: int = 1
    checkpoint: str | None = None
    games: int = 16
    simulations: int = 64
    max_plies: int = 512
    temperature_moves: int = 20
    epochs: int = 2
    batch_size: int = 128
    channels: int = 128
    blocks: int = 6
    lr: float = 1e-3
    promotion_score: float = 0.50
    eval_games: int = 8
    eval_simulations: int = 64
    seed: int = 0
    device: str = "auto"


def run_iterations(config: IterationConfig) -> Path:
    run_dir = Path(config.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    league_path = run_dir / "league.json"
    league = _load_league(league_path)

    best_checkpoint = config.checkpoint or league.get("best_checkpoint")
    selfplay_dirs: list[str] = list(league.get("selfplay_dirs", []))

    for local_iter in range(config.iterations):
        iteration = int(league.get("iterations_completed", 0)) + 1
        iter_seed = config.seed + iteration
        selfplay_dir = run_dir / "selfplay" / f"iter_{iteration:04d}"
        candidate_dir = run_dir / "checkpoints" / f"iter_{iteration:04d}"

        evaluator = (
            load_evaluator(best_checkpoint, device=config.device)
            if best_checkpoint
            else UniformEvaluator()
        )
        generate_self_play(
            evaluator,
            selfplay_dir,
            SelfPlayConfig(
                games=config.games,
                simulations=config.simulations,
                max_plies=config.max_plies,
                temperature_moves=config.temperature_moves,
                seed=iter_seed,
            ),
        )
        selfplay_dirs.append(str(selfplay_dir))

        candidate = train(
            TrainConfig(
                data=selfplay_dirs,
                out=str(candidate_dir),
                checkpoint=best_checkpoint,
                epochs=config.epochs,
                batch_size=config.batch_size,
                lr=config.lr,
                channels=config.channels,
                blocks=config.blocks,
                seed=iter_seed,
                device=config.device,
            )
        )

        eval_config = {
            "checkpoint": str(candidate),
            "games": config.eval_games,
            "simulations": config.eval_simulations,
            "opponent_checkpoint": best_checkpoint,
            "device": config.device,
            "seed": iter_seed,
            "max_plies": config.max_plies,
        }
        metrics = evaluate_checkpoint_from_dict(eval_config)
        promoted = best_checkpoint is None or metrics["score_rate"] >= config.promotion_score
        if promoted:
            best_checkpoint = str(candidate)

        league = {
            "config": asdict(config),
            "best_checkpoint": best_checkpoint,
            "selfplay_dirs": selfplay_dirs,
            "iterations_completed": iteration,
            "history": league.get("history", [])
            + [
                {
                    "iteration": iteration,
                    "candidate": str(candidate),
                    "promoted": promoted,
                    "metrics": metrics,
                    "eval": eval_config,
                }
            ],
        }
        league_path.write_text(json.dumps(league, indent=2))

    return league_path


def evaluate_checkpoint_from_dict(config: dict) -> dict[str, float]:
    from alpha_chess.evaluate import EvalConfig

    return evaluate_checkpoint(EvalConfig(**config))


def _load_league(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())
