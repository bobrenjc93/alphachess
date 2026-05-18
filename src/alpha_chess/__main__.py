"""Command line interface."""

from __future__ import annotations

import argparse
from dataclasses import asdict

from rich import print

from alpha_chess.evaluate import EvalConfig, evaluate_checkpoint
from alpha_chess.evaluator import UniformEvaluator, load_evaluator
from alpha_chess.iteration import IterationConfig, run_iterations
from alpha_chess.pgn_import import PGNImportConfig, import_pgn
from alpha_chess.self_play import SelfPlayConfig, generate_self_play
from alpha_chess.train import TrainConfig, train
from alpha_chess.uci import UCIConfig, run_uci


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="alpha-chess")
    subparsers = parser.add_subparsers(dest="command", required=True)

    self_play = subparsers.add_parser("self-play", help="generate self-play NPZ data")
    self_play.add_argument("--checkpoint")
    self_play.add_argument("--device", default="auto")
    self_play.add_argument("--out", default="data/selfplay/run")
    self_play.add_argument("--games", type=int, default=1)
    self_play.add_argument("--simulations", type=int, default=64)
    self_play.add_argument("--max-plies", type=int, default=512)
    self_play.add_argument("--temperature-moves", type=int, default=20)
    self_play.add_argument("--seed", type=int, default=0)

    train_parser = subparsers.add_parser("train", help="train a policy/value checkpoint")
    train_parser.add_argument("--data", required=True)
    train_parser.add_argument("--out", default="checkpoints/run")
    train_parser.add_argument("--checkpoint")
    train_parser.add_argument("--epochs", type=int, default=1)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--value-weight", type=float, default=1.0)
    train_parser.add_argument("--channels", type=int, default=128)
    train_parser.add_argument("--blocks", type=int, default=6)
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--device", default="auto")

    eval_parser = subparsers.add_parser("eval", help="evaluate a checkpoint")
    eval_parser.add_argument("--checkpoint", required=True)
    eval_parser.add_argument("--games", type=int, default=2)
    eval_parser.add_argument("--simulations", type=int, default=64)
    eval_parser.add_argument("--opponent", default="uniform")
    eval_parser.add_argument("--opponent-checkpoint")
    eval_parser.add_argument("--engine-path", default="stockfish")
    eval_parser.add_argument("--engine-time", type=float, default=0.05)
    eval_parser.add_argument("--engine-depth", type=int)
    eval_parser.add_argument("--device", default="auto")
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

    iterate_parser = subparsers.add_parser("iterate", help="run self-play/train/eval iterations")
    iterate_parser.add_argument("--run-dir", default="experiments/local")
    iterate_parser.add_argument("--iterations", type=int, default=1)
    iterate_parser.add_argument("--checkpoint")
    iterate_parser.add_argument("--games", type=int, default=16)
    iterate_parser.add_argument("--simulations", type=int, default=64)
    iterate_parser.add_argument("--max-plies", type=int, default=512)
    iterate_parser.add_argument("--temperature-moves", type=int, default=20)
    iterate_parser.add_argument("--epochs", type=int, default=2)
    iterate_parser.add_argument("--batch-size", type=int, default=128)
    iterate_parser.add_argument("--channels", type=int, default=128)
    iterate_parser.add_argument("--blocks", type=int, default=6)
    iterate_parser.add_argument("--lr", type=float, default=1e-3)
    iterate_parser.add_argument("--promotion-score", type=float, default=0.50)
    iterate_parser.add_argument("--eval-games", type=int, default=8)
    iterate_parser.add_argument("--eval-simulations", type=int, default=64)
    iterate_parser.add_argument("--seed", type=int, default=0)
    iterate_parser.add_argument("--device", default="auto")

    uci_parser = subparsers.add_parser("uci", help="serve a checkpoint through UCI")
    uci_parser.add_argument("--checkpoint", required=True)
    uci_parser.add_argument("--simulations", type=int, default=64)
    uci_parser.add_argument("--device", default="auto")

    args = parser.parse_args(argv)

    if args.command == "self-play":
        evaluator = (
            load_evaluator(args.checkpoint, device=args.device)
            if args.checkpoint
            else UniformEvaluator()
        )
        config = SelfPlayConfig(
            games=args.games,
            simulations=args.simulations,
            max_plies=args.max_plies,
            temperature_moves=args.temperature_moves,
            seed=args.seed,
        )
        paths = generate_self_play(evaluator, args.out, config)
        print({"written": [str(path) for path in paths], "config": asdict(config)})
    elif args.command == "train":
        kwargs = vars(args).copy()
        kwargs.pop("command", None)
        config = TrainConfig(**kwargs)
        checkpoint = train(config)
        print({"checkpoint": str(checkpoint), "config": asdict(config)})
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
