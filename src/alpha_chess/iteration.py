"""AlphaZero-style self-play / train / promote iteration driver."""

from __future__ import annotations

import json
import math
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
    self_play_workers: int = 1
    simulations: int = 64
    max_plies: int = 512
    temperature_moves: int = 20
    epochs: int = 2
    batch_size: int = 128
    channels: int = 128
    blocks: int = 6
    lr: float = 1e-3
    value_weight: float = 1.0
    legal_policy_loss: bool = False
    material_value_weight: float = 0.0
    material_value_search_plies: int = 0
    root_mate_search_plies: int = 3
    root_material_search_plies: int = 0
    root_material_max_loss_cp: int = 250
    replay_data: list[str] | None = None
    self_play_weight: float = 1.0
    replay_weights: list[float] | None = None
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
    selfplay_dirs = _filter_nonempty_data_dirs(league.get("selfplay_dirs", []))

    for local_iter in range(config.iterations):
        iteration = int(league.get("iterations_completed", 0)) + 1
        iter_seed = config.seed + iteration
        selfplay_dir = run_dir / "selfplay" / f"iter_{iteration:04d}"
        candidate_dir = run_dir / "checkpoints" / f"iter_{iteration:04d}"

        evaluator = (
            load_evaluator(
                best_checkpoint,
                device=config.device,
                material_value_weight=config.material_value_weight,
                material_value_search_plies=config.material_value_search_plies,
            )
            if best_checkpoint
            else UniformEvaluator()
        )
        written_selfplay = generate_self_play(
            evaluator,
            selfplay_dir,
            SelfPlayConfig(
                games=config.games,
                simulations=config.simulations,
                max_plies=config.max_plies,
                temperature_moves=config.temperature_moves,
                root_mate_search_plies=config.root_mate_search_plies,
                root_material_search_plies=config.root_material_search_plies,
                root_material_max_loss_cp=config.root_material_max_loss_cp,
                seed=iter_seed,
                workers=config.self_play_workers,
            ),
        )
        if written_selfplay:
            selfplay_dirs.append(str(selfplay_dir))
        train_data, data_weights = _build_training_inputs(selfplay_dirs, config)

        candidate = train(
            TrainConfig(
                data=train_data,
                out=str(candidate_dir),
                checkpoint=best_checkpoint,
                epochs=config.epochs,
                batch_size=config.batch_size,
                lr=config.lr,
                value_weight=config.value_weight,
                data_weights=data_weights,
                legal_policy_loss=config.legal_policy_loss,
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
            "material_value_weight": config.material_value_weight,
            "material_value_search_plies": config.material_value_search_plies,
            "root_mate_search_plies": config.root_mate_search_plies,
            "root_material_search_plies": config.root_material_search_plies,
            "root_material_max_loss_cp": config.root_material_max_loss_cp,
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


def _filter_nonempty_data_dirs(paths: list[str]) -> list[str]:
    return [str(path) for path in paths if _has_npz_data(Path(path))]


def _has_npz_data(path: Path) -> bool:
    if path.is_file():
        return path.suffix == ".npz"
    return path.is_dir() and any(path.glob("*.npz"))


def _build_training_inputs(
    selfplay_dirs: list[str], config: IterationConfig
) -> tuple[list[str], list[float] | None]:
    """Combine accumulated self-play with fixed replay sources for training."""

    replay_data = list(config.replay_data or [])
    if not replay_data:
        return list(selfplay_dirs), None

    if not math.isfinite(config.self_play_weight) or config.self_play_weight < 0:
        raise ValueError("self_play_weight must be finite and non-negative")

    if config.replay_weights is None:
        replay_weights = [1.0] * len(replay_data)
    else:
        replay_weights = list(config.replay_weights)
        if len(replay_weights) != len(replay_data):
            raise ValueError(
                "replay_weights must have one entry for each replay_data source"
            )
        if any(not math.isfinite(weight) or weight < 0 for weight in replay_weights):
            raise ValueError("replay_weights must be finite non-negative values")

    train_data = list(selfplay_dirs) + replay_data
    selfplay_weights = (
        [config.self_play_weight / len(selfplay_dirs)] * len(selfplay_dirs)
        if selfplay_dirs
        else []
    )
    return train_data, selfplay_weights + replay_weights
