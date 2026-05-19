"""Minimal UCI protocol wrapper for AlphaChess checkpoints."""

from __future__ import annotations

import sys
from dataclasses import dataclass

import chess

from alpha_chess.chess_env import action_to_move
from alpha_chess.evaluator import load_evaluator
from alpha_chess.mcts import AlphaZeroMCTS, MCTSConfig


@dataclass
class UCIConfig:
    checkpoint: str
    simulations: int = 64
    device: str = "auto"
    material_value_weight: float = 0.0
    material_value_search_plies: int = 0
    root_mate_search_plies: int = 3
    root_material_search_plies: int = 0
    root_material_max_loss_cp: int = 250


def run_uci(config: UCIConfig) -> None:
    evaluator = load_evaluator(
        config.checkpoint,
        device=config.device,
        material_value_weight=config.material_value_weight,
        material_value_search_plies=config.material_value_search_plies,
    )
    board = chess.Board()

    def send(message: str) -> None:
        print(message, flush=True)

    send("id name AlphaChess")
    send("id author bobrenjc93")
    send("option name Simulations type spin default 64 min 1 max 100000")
    send("option name MaterialValueSearchPlies type spin default 0 min 0 max 8")
    send("option name RootMateSearchPlies type spin default 3 min 0 max 8")
    send("option name RootMaterialSearchPlies type spin default 0 min 0 max 8")
    send("option name RootMaterialMaxLossCp type spin default 250 min 0 max 5000")

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        if line == "uci":
            send("id name AlphaChess")
            send("id author bobrenjc93")
            send("option name Simulations type spin default 64 min 1 max 100000")
            send("option name MaterialValueSearchPlies type spin default 0 min 0 max 8")
            send("option name RootMateSearchPlies type spin default 3 min 0 max 8")
            send("option name RootMaterialSearchPlies type spin default 0 min 0 max 8")
            send("option name RootMaterialMaxLossCp type spin default 250 min 0 max 5000")
            send("uciok")
        elif line == "isready":
            send("readyok")
        elif line == "ucinewgame":
            board = chess.Board()
        elif line.startswith("setoption"):
            config = _parse_setoption(line, config)
        elif line.startswith("position"):
            board = _parse_position(line)
        elif line.startswith("go"):
            move = _choose_move(board, evaluator, config)
            send(f"bestmove {move.uci() if move is not None else '0000'}")
        elif line == "quit":
            break


def _choose_move(board: chess.Board, evaluator, config: UCIConfig) -> chess.Move | None:
    if board.is_game_over(claim_draw=True):
        return None
    search = AlphaZeroMCTS(
        evaluator,
        MCTSConfig(
            simulations=config.simulations,
            root_mate_search_plies=config.root_mate_search_plies,
            root_material_search_plies=config.root_material_search_plies,
            root_material_max_loss_cp=config.root_material_max_loss_cp,
        ),
    )
    result = search.run(board)
    action = result.select_action(temperature=0.0, rng=search.rng)
    return action_to_move(action, board) if action is not None else None


def _parse_position(line: str) -> chess.Board:
    tokens = line.split()
    if len(tokens) < 2:
        return chess.Board()

    move_index: int | None = None
    if "moves" in tokens:
        move_index = tokens.index("moves")

    if tokens[1] == "startpos":
        board = chess.Board()
    elif tokens[1] == "fen":
        fen_end = move_index if move_index is not None else len(tokens)
        board = chess.Board(" ".join(tokens[2:fen_end]))
    else:
        board = chess.Board()

    if move_index is not None:
        for move_text in tokens[move_index + 1 :]:
            board.push_uci(move_text)
    return board


def _parse_setoption(line: str, config: UCIConfig) -> UCIConfig:
    tokens = line.split()
    lowered = [token.lower() for token in tokens]
    if "name" not in lowered or "value" not in lowered:
        return config
    name_index = lowered.index("name")
    value_index = lowered.index("value")
    name = " ".join(tokens[name_index + 1 : value_index]).lower()
    value = " ".join(tokens[value_index + 1 :])
    if name == "simulations":
        try:
            return UCIConfig(
                checkpoint=config.checkpoint,
                simulations=max(1, int(value)),
                device=config.device,
                material_value_weight=config.material_value_weight,
                material_value_search_plies=config.material_value_search_plies,
                root_mate_search_plies=config.root_mate_search_plies,
                root_material_search_plies=config.root_material_search_plies,
                root_material_max_loss_cp=config.root_material_max_loss_cp,
            )
        except ValueError:
            return config
    if name == "materialvaluesearchplies":
        try:
            return UCIConfig(
                checkpoint=config.checkpoint,
                simulations=config.simulations,
                device=config.device,
                material_value_weight=config.material_value_weight,
                material_value_search_plies=max(0, int(value)),
                root_mate_search_plies=config.root_mate_search_plies,
                root_material_search_plies=config.root_material_search_plies,
                root_material_max_loss_cp=config.root_material_max_loss_cp,
            )
        except ValueError:
            return config
    if name == "rootmatesearchplies":
        try:
            return UCIConfig(
                checkpoint=config.checkpoint,
                simulations=config.simulations,
                device=config.device,
                material_value_weight=config.material_value_weight,
                material_value_search_plies=config.material_value_search_plies,
                root_mate_search_plies=max(0, int(value)),
                root_material_search_plies=config.root_material_search_plies,
                root_material_max_loss_cp=config.root_material_max_loss_cp,
            )
        except ValueError:
            return config
    if name == "rootmaterialsearchplies":
        try:
            return UCIConfig(
                checkpoint=config.checkpoint,
                simulations=config.simulations,
                device=config.device,
                material_value_weight=config.material_value_weight,
                material_value_search_plies=config.material_value_search_plies,
                root_mate_search_plies=config.root_mate_search_plies,
                root_material_search_plies=max(0, int(value)),
                root_material_max_loss_cp=config.root_material_max_loss_cp,
            )
        except ValueError:
            return config
    if name == "rootmaterialmaxlosscp":
        try:
            return UCIConfig(
                checkpoint=config.checkpoint,
                simulations=config.simulations,
                device=config.device,
                material_value_weight=config.material_value_weight,
                material_value_search_plies=config.material_value_search_plies,
                root_mate_search_plies=config.root_mate_search_plies,
                root_material_search_plies=config.root_material_search_plies,
                root_material_max_loss_cp=max(0, int(value)),
            )
        except ValueError:
            return config
    return config
