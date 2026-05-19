"""Command line interface."""

from __future__ import annotations

import argparse
from dataclasses import asdict

from rich import print

from alpha_chess.evaluate import EvalConfig, evaluate_checkpoint
from alpha_chess.iteration import IterationConfig, run_iterations
from alpha_chess.model import blend_checkpoints
from alpha_chess.pgn_import import PGNImportConfig, import_pgn
from alpha_chess.puzzle_import import PuzzleImportConfig, import_puzzles
from alpha_chess.self_play import SelfPlayConfig, generate_self_play_from_checkpoint
from alpha_chess.stockfish_teacher import StockfishTeacherConfig, generate_stockfish_teacher
from alpha_chess.train import TrainConfig, ValidateConfig, train, validate
from alpha_chess.uci import UCIConfig, run_uci


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="alpha-chess")
    subparsers = parser.add_subparsers(dest="command", required=True)

    self_play = subparsers.add_parser("self-play", help="generate self-play NPZ data")
    self_play.add_argument("--checkpoint")
    self_play.add_argument("--device", default="auto")
    self_play.add_argument("--material-value-weight", type=float, default=0.0)
    self_play.add_argument("--material-value-search-plies", type=int, default=0)
    self_play.add_argument("--out", default="data/selfplay/run")
    self_play.add_argument("--games", type=int, default=1)
    self_play.add_argument("--workers", type=int, default=1)
    self_play.add_argument("--simulations", type=int, default=64)
    self_play.add_argument("--c-puct", type=float, default=1.5)
    self_play.add_argument("--policy-prior-temperature", type=float, default=1.0)
    self_play.add_argument("--max-plies", type=int, default=512)
    self_play.add_argument("--temperature-moves", type=int, default=20)
    self_play.add_argument("--root-mate-search-plies", type=int, default=3)
    self_play.add_argument("--root-material-search-plies", type=int, default=0)
    self_play.add_argument("--root-material-max-loss-cp", type=int, default=250)
    self_play.add_argument("--seed", type=int, default=0)

    train_parser = subparsers.add_parser("train", help="train a policy/value checkpoint")
    train_parser.add_argument("--data", required=True, nargs="+")
    train_parser.add_argument("--out", default="checkpoints/run")
    train_parser.add_argument("--checkpoint")
    train_parser.add_argument("--epochs", type=int, default=1)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--value-weight", type=float, default=1.0)
    train_parser.add_argument("--bad-action-weight", type=float, default=0.0)
    train_parser.add_argument("--bad-action-margin", type=float, default=1.0)
    train_parser.add_argument("--data-weights", type=float, nargs="+")
    train_parser.add_argument("--legal-policy-loss", action="store_true")
    train_parser.add_argument("--channels", type=int, default=128)
    train_parser.add_argument("--blocks", type=int, default=6)
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--device", default="auto")

    validate_parser = subparsers.add_parser(
        "validate", help="score a checkpoint on replay datasets without training"
    )
    validate_parser.add_argument("--checkpoint", required=True)
    validate_parser.add_argument("--data", required=True, nargs="+")
    validate_parser.add_argument("--batch-size", type=int, default=256)
    validate_parser.add_argument("--value-weight", type=float, default=1.0)
    validate_parser.add_argument("--bad-action-weight", type=float, default=0.0)
    validate_parser.add_argument("--bad-action-margin", type=float, default=1.0)
    validate_parser.add_argument("--legal-policy-loss", action="store_true")
    validate_parser.add_argument("--device", default="auto")

    blend_parser = subparsers.add_parser(
        "blend-checkpoints", help="linearly interpolate two checkpoints"
    )
    blend_parser.add_argument("--checkpoint-a", required=True)
    blend_parser.add_argument("--checkpoint-b", required=True)
    blend_parser.add_argument("--weight-b", type=float, required=True)
    blend_parser.add_argument("--out", required=True)

    eval_parser = subparsers.add_parser("eval", help="evaluate a checkpoint")
    eval_parser.add_argument("--checkpoint", required=True)
    eval_parser.add_argument("--games", type=int, default=2)
    eval_parser.add_argument("--simulations", type=int, default=64)
    eval_parser.add_argument("--c-puct", type=float, default=1.5)
    eval_parser.add_argument("--policy-prior-temperature", type=float, default=1.0)
    eval_parser.add_argument("--opponent", default="uniform")
    eval_parser.add_argument("--opponent-checkpoint")
    eval_parser.add_argument("--engine-path", default="stockfish")
    eval_parser.add_argument("--engine-time", type=float, default=0.05)
    eval_parser.add_argument("--engine-depth", type=int)
    eval_parser.add_argument("--device", default="auto")
    eval_parser.add_argument("--material-value-weight", type=float, default=0.0)
    eval_parser.add_argument("--material-value-search-plies", type=int, default=0)
    eval_parser.add_argument("--root-mate-search-plies", type=int, default=3)
    eval_parser.add_argument("--root-material-search-plies", type=int, default=0)
    eval_parser.add_argument("--root-material-max-loss-cp", type=int, default=250)
    eval_parser.add_argument("--seed", type=int, default=0)
    eval_parser.add_argument("--max-plies", type=int, default=512)
    eval_parser.add_argument("--pgn-out")

    import_parser = subparsers.add_parser("import-pgn", help="convert expert PGN games to NPZ data")
    import_parser.add_argument("--pgn", required=True)
    import_parser.add_argument("--out", default="data/expert")
    import_parser.add_argument("--max-games", type=int)
    import_parser.add_argument("--max-imported-games", type=int)
    import_parser.add_argument("--min-elo", type=int)
    import_parser.add_argument("--rated-only", action="store_true")
    import_parser.add_argument("--min-initial-seconds", type=int)
    import_parser.add_argument("--min-plies", type=int, default=1)
    import_parser.add_argument("--chunk-size", type=int, default=4096)
    import_parser.add_argument("--dense-policy", action="store_true")

    teacher_parser = subparsers.add_parser(
        "stockfish-teacher", help="generate Stockfish best-move teacher data"
    )
    teacher_parser.add_argument("--pgn", required=True, nargs="+")
    teacher_parser.add_argument("--out", default="data/teacher/stockfish")
    teacher_parser.add_argument("--engine-path", default="stockfish")
    teacher_parser.add_argument("--engine-time", type=float, default=0.02)
    teacher_parser.add_argument("--engine-depth", type=int)
    teacher_parser.add_argument("--max-games", type=int)
    teacher_parser.add_argument("--max-positions", type=int, default=1024)
    teacher_parser.add_argument("--min-elo", type=int)
    teacher_parser.add_argument("--min-initial-seconds", type=int)
    teacher_parser.add_argument("--min-value-delta", type=float)
    teacher_parser.add_argument("--player-name")
    teacher_parser.add_argument("--multipv", type=int, default=1)
    teacher_parser.add_argument("--policy-temperature-cp", type=float, default=200.0)
    teacher_parser.add_argument("--position-stride", type=int, default=4)
    teacher_parser.add_argument("--min-ply", type=int, default=0)
    teacher_parser.add_argument("--max-ply", type=int)
    teacher_parser.add_argument("--pv-plies", type=int, default=0)
    teacher_parser.add_argument("--game-line-plies", type=int, default=0)
    teacher_parser.add_argument("--chunk-size", type=int, default=1024)

    puzzle_parser = subparsers.add_parser(
        "import-puzzles", help="convert Lichess puzzle CSV data to NPZ data"
    )
    puzzle_parser.add_argument("--puzzles", required=True)
    puzzle_parser.add_argument("--out", default="data/puzzles")
    puzzle_parser.add_argument("--max-positions", type=int)
    puzzle_parser.add_argument("--min-rating", type=int)
    puzzle_parser.add_argument("--max-rating", type=int)
    puzzle_parser.add_argument("--theme")
    puzzle_parser.add_argument("--chunk-size", type=int, default=4096)
    puzzle_parser.add_argument("--value", type=float, default=1.0)
    puzzle_parser.add_argument("--include-solution-line", action="store_true")

    iterate_parser = subparsers.add_parser("iterate", help="run self-play/train/eval iterations")
    iterate_parser.add_argument("--run-dir", default="experiments/local")
    iterate_parser.add_argument("--iterations", type=int, default=1)
    iterate_parser.add_argument("--checkpoint")
    iterate_parser.add_argument("--games", type=int, default=16)
    iterate_parser.add_argument("--self-play-workers", type=int, default=1)
    iterate_parser.add_argument("--simulations", type=int, default=64)
    iterate_parser.add_argument("--c-puct", type=float, default=1.5)
    iterate_parser.add_argument("--policy-prior-temperature", type=float, default=1.0)
    iterate_parser.add_argument("--max-plies", type=int, default=512)
    iterate_parser.add_argument("--temperature-moves", type=int, default=20)
    iterate_parser.add_argument("--epochs", type=int, default=2)
    iterate_parser.add_argument("--batch-size", type=int, default=128)
    iterate_parser.add_argument("--channels", type=int, default=128)
    iterate_parser.add_argument("--blocks", type=int, default=6)
    iterate_parser.add_argument("--lr", type=float, default=1e-3)
    iterate_parser.add_argument("--value-weight", type=float, default=1.0)
    iterate_parser.add_argument("--bad-action-weight", type=float, default=0.0)
    iterate_parser.add_argument("--bad-action-margin", type=float, default=1.0)
    iterate_parser.add_argument("--legal-policy-loss", action="store_true")
    iterate_parser.add_argument("--material-value-weight", type=float, default=0.0)
    iterate_parser.add_argument("--material-value-search-plies", type=int, default=0)
    iterate_parser.add_argument("--root-mate-search-plies", type=int, default=3)
    iterate_parser.add_argument("--root-material-search-plies", type=int, default=0)
    iterate_parser.add_argument("--root-material-max-loss-cp", type=int, default=250)
    iterate_parser.add_argument("--replay-data", nargs="+")
    iterate_parser.add_argument("--self-play-weight", type=float, default=1.0)
    iterate_parser.add_argument("--replay-weights", type=float, nargs="+")
    iterate_parser.add_argument("--promotion-score", type=float, default=0.50)
    iterate_parser.add_argument("--eval-games", type=int, default=8)
    iterate_parser.add_argument("--eval-simulations", type=int, default=64)
    iterate_parser.add_argument("--stockfish-gate-games", type=int, default=0)
    iterate_parser.add_argument("--stockfish-gate-simulations", type=int)
    iterate_parser.add_argument("--stockfish-gate-min-score", type=float, default=0.50)
    iterate_parser.add_argument("--stockfish-gate-engine-path", default="stockfish")
    iterate_parser.add_argument("--stockfish-gate-engine-time", type=float, default=0.05)
    iterate_parser.add_argument("--stockfish-gate-engine-depth", type=int)
    iterate_parser.add_argument("--seed", type=int, default=0)
    iterate_parser.add_argument("--device", default="auto")

    uci_parser = subparsers.add_parser("uci", help="serve a checkpoint through UCI")
    uci_parser.add_argument("--checkpoint", required=True)
    uci_parser.add_argument("--simulations", type=int, default=64)
    uci_parser.add_argument("--c-puct", type=float, default=1.5)
    uci_parser.add_argument("--policy-prior-temperature", type=float, default=1.0)
    uci_parser.add_argument("--device", default="auto")
    uci_parser.add_argument("--material-value-weight", type=float, default=0.0)
    uci_parser.add_argument("--material-value-search-plies", type=int, default=0)
    uci_parser.add_argument("--root-mate-search-plies", type=int, default=3)
    uci_parser.add_argument("--root-material-search-plies", type=int, default=0)
    uci_parser.add_argument("--root-material-max-loss-cp", type=int, default=250)

    args = parser.parse_args(argv)

    if args.command == "self-play":
        config = SelfPlayConfig(
            games=args.games,
            simulations=args.simulations,
            c_puct=args.c_puct,
            policy_prior_temperature=args.policy_prior_temperature,
            max_plies=args.max_plies,
            temperature_moves=args.temperature_moves,
            root_mate_search_plies=args.root_mate_search_plies,
            root_material_search_plies=args.root_material_search_plies,
            root_material_max_loss_cp=args.root_material_max_loss_cp,
            seed=args.seed,
            workers=args.workers,
        )
        paths = generate_self_play_from_checkpoint(
            args.checkpoint,
            args.out,
            config,
            device=args.device,
            material_value_weight=args.material_value_weight,
            material_value_search_plies=args.material_value_search_plies,
        )
        print({"written": [str(path) for path in paths], "config": asdict(config)})
    elif args.command == "train":
        kwargs = vars(args).copy()
        kwargs.pop("command", None)
        config = TrainConfig(**kwargs)
        checkpoint = train(config)
        print({"checkpoint": str(checkpoint), "config": asdict(config)})
    elif args.command == "validate":
        kwargs = vars(args).copy()
        kwargs.pop("command", None)
        config = ValidateConfig(**kwargs)
        print({"metrics": validate(config), "config": asdict(config)})
    elif args.command == "blend-checkpoints":
        output = blend_checkpoints(
            args.checkpoint_a,
            args.checkpoint_b,
            args.out,
            args.weight_b,
        )
        print({"checkpoint": str(output)})
    elif args.command == "eval":
        kwargs = vars(args).copy()
        kwargs.pop("command", None)
        config = EvalConfig(**kwargs)
        print(evaluate_checkpoint(config))
    elif args.command == "import-pgn":
        kwargs = vars(args).copy()
        kwargs.pop("command", None)
        config = PGNImportConfig(**kwargs)
        paths = import_pgn(config)
        print({"written": [str(path) for path in paths], "config": asdict(config)})
    elif args.command == "stockfish-teacher":
        kwargs = vars(args).copy()
        kwargs.pop("command", None)
        config = StockfishTeacherConfig(**kwargs)
        paths = generate_stockfish_teacher(config)
        print({"written": [str(path) for path in paths], "config": asdict(config)})
    elif args.command == "import-puzzles":
        kwargs = vars(args).copy()
        kwargs.pop("command", None)
        config = PuzzleImportConfig(**kwargs)
        paths = import_puzzles(config)
        print({"written": [str(path) for path in paths], "config": asdict(config)})
    elif args.command == "iterate":
        kwargs = vars(args).copy()
        kwargs.pop("command", None)
        config = IterationConfig(**kwargs)
        league = run_iterations(config)
        print({"league": str(league), "config": asdict(config)})
    elif args.command == "uci":
        kwargs = vars(args).copy()
        kwargs.pop("command", None)
        run_uci(UCIConfig(**kwargs))


if __name__ == "__main__":
    main()
