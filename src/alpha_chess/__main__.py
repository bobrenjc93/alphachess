"""Command line interface."""

from __future__ import annotations

import argparse
from dataclasses import asdict

from rich import print

from alpha_chess.evaluate import EvalConfig, evaluate_checkpoint
from alpha_chess.evaluator import UniformEvaluator, load_evaluator
from alpha_chess.pgn_import import PGNImportConfig, import_pgn
from alpha_chess.self_play import SelfPlayConfig, generate_self_play
from alpha_chess.train import TrainConfig, train


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
    eval_parser.add_argument("--device", default="auto")
    eval_parser.add_argument("--seed", type=int, default=0)
    eval_parser.add_argument("--max-plies", type=int, default=512)

    import_parser = subparsers.add_parser("import-pgn", help="convert expert PGN games to NPZ data")
    import_parser.add_argument("--pgn", required=True)
    import_parser.add_argument("--out", default="data/expert")
    import_parser.add_argument("--max-games", type=int)
    import_parser.add_argument("--min-plies", type=int, default=1)
    import_parser.add_argument("--chunk-size", type=int, default=4096)

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


if __name__ == "__main__":
    main()
