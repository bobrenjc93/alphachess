"""Mine model-preferred moves that Stockfish scores as value drops."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chess
import chess.engine
import numpy as np
import torch
from torch.utils.data import DataLoader

from alpha_chess.chess_env import action_to_move
from alpha_chess.dataset import SelfPlayDataset, collate_samples
from alpha_chess.hard_negatives import _target_actions
from alpha_chess.model import load_checkpoint
from alpha_chess.stockfish_teacher import _value_drop_after_move
from alpha_chess.train import _legal_action_mask_from_fens, resolve_device


@dataclass
class ModelBlunderConfig:
    checkpoint: str
    data: str | list[str]
    out: str
    engine_path: str = "stockfish"
    engine_time: float = 0.02
    engine_depth: int | None = None
    max_positions: int = 8192
    min_value_delta: float = 0.08
    bad_actions_per_position: int = 1
    batch_size: int = 256
    chunk_size: int = 1024
    prefer_action_labels: bool = True
    device: str = "auto"


def mine_model_blunders(config: ModelBlunderConfig) -> list[Path]:
    device = resolve_device(config.device)
    dataset = SelfPlayDataset(
        config.data,
        in_memory=True,
        prefer_action_labels=config.prefer_action_labels,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_samples,
    )
    model = load_checkpoint(config.checkpoint, map_location=device)
    model.to(device)
    model.eval()

    out_dir = Path(config.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = _ModelBlunderWriter(out_dir, max(1, int(config.chunk_size)))
    limit = chess.engine.Limit(time=config.engine_time, depth=config.engine_depth)

    positions_seen = 0
    model_wrong_positions = 0
    blunder_positions = 0
    bad_action_count = 0

    with chess.engine.SimpleEngine.popen_uci(config.engine_path) as engine:
        with torch.no_grad():
            for batch in loader:
                if positions_seen >= config.max_positions:
                    break
                if "fen" not in batch:
                    raise ValueError("model-blunder mining requires replay data with FENs")
                fens = batch["fen"]
                if not isinstance(fens, list):
                    raise TypeError("Batch FENs must be a list of strings")

                boards = batch["board"].to(device)
                targets = _target_actions(batch).to(device)
                policy_logits, _value = model(boards)
                legal_mask = _legal_action_mask_from_fens(fens, device)
                target_is_legal = legal_mask.gather(1, targets.unsqueeze(1)).squeeze(1)
                if not bool(target_is_legal.all().item()):
                    raise ValueError("model-blunder mining received an illegal target action")

                masked_logits = policy_logits.masked_fill(~legal_mask, -1e9)
                ranked_actions = _ranked_legal_actions(
                    masked_logits,
                    legal_mask,
                    max(1, int(config.bad_actions_per_position)),
                )

                for row, fen in enumerate(fens):
                    if positions_seen >= config.max_positions:
                        break
                    positions_seen += 1

                    target_action = int(targets[row].item())
                    if int(ranked_actions[row, 0].item()) == target_action:
                        continue
                    model_wrong_positions += 1

                    board = chess.Board(fen)
                    value = float(batch["value"][row].item())
                    bad_actions, value_deltas = _stockfish_bad_actions(
                        engine=engine,
                        board=board,
                        ranked_actions=[
                            int(action)
                            for action in ranked_actions[row].detach().cpu().tolist()
                        ],
                        target_action=target_action,
                        best_value=value,
                        limit=limit,
                        min_value_delta=float(config.min_value_delta),
                        max_bad_actions=max(1, int(config.bad_actions_per_position)),
                    )
                    if not bad_actions:
                        continue

                    blunder_positions += 1
                    bad_action_count += len(bad_actions)
                    writer.add(
                        board=batch["board"][row].cpu().numpy().astype(np.float32),
                        action=target_action,
                        value=value,
                        fen=fen,
                        bad_actions=bad_actions,
                        value_deltas=value_deltas,
                    )

    paths = writer.finish()
    if not paths:
        raise ValueError("No model blunders mined")

    summary = [
        f"checkpoint={config.checkpoint}",
        f"data={config.data}",
        f"engine_path={config.engine_path}",
        f"engine_time={config.engine_time}",
        f"engine_depth={config.engine_depth}",
        f"positions_seen={positions_seen}",
        f"model_wrong_positions={model_wrong_positions}",
        f"blunder_positions={blunder_positions}",
        f"bad_actions={bad_action_count}",
        f"min_value_delta={config.min_value_delta}",
        f"bad_actions_per_position={config.bad_actions_per_position}",
        f"chunks={len(paths)}",
    ]
    (out_dir / "model_blunder_summary.txt").write_text("\n".join(summary) + "\n")
    return paths


def _ranked_legal_actions(
    masked_logits: torch.Tensor,
    legal_mask: torch.Tensor,
    bad_actions_per_position: int,
) -> torch.Tensor:
    legal_counts = legal_mask.sum(dim=1)
    max_legal = int(legal_counts.max().item()) if legal_counts.numel() else 1
    width = min(masked_logits.shape[1], max_legal, bad_actions_per_position + 1)
    return torch.topk(masked_logits, k=max(1, width), dim=-1).indices


def _stockfish_bad_actions(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    ranked_actions: list[int],
    target_action: int,
    best_value: float,
    limit: chess.engine.Limit,
    min_value_delta: float,
    max_bad_actions: int,
) -> tuple[list[int], list[float]]:
    bad_actions: list[int] = []
    value_deltas: list[float] = []
    seen: set[int] = set()
    for action in ranked_actions:
        if action == target_action or action in seen:
            continue
        seen.add(action)
        move = action_to_move(action, board)
        if move is None or move not in board.legal_moves:
            continue
        after_board = board.copy(stack=False)
        after_board.push(move)
        after_info = engine.analyse(after_board, limit)
        value_delta = _value_drop_after_move(
            best_value=best_value,
            after_score=after_info.get("score"),
            after_turn=after_board.turn,
        )
        if value_delta < min_value_delta:
            continue
        bad_actions.append(action)
        value_deltas.append(float(value_delta))
        if len(bad_actions) >= max_bad_actions:
            break
    return bad_actions, value_deltas


class _ModelBlunderWriter:
    def __init__(self, out_dir: Path, chunk_size: int) -> None:
        self.out_dir = out_dir
        self.chunk_size = chunk_size
        self.paths: list[Path] = []
        self.boards: list[np.ndarray] = []
        self.actions: list[int] = []
        self.values: list[float] = []
        self.fens: list[str] = []
        self.bad_actions: list[list[int]] = []
        self.value_deltas: list[list[float]] = []

    def add(
        self,
        board: np.ndarray,
        action: int,
        value: float,
        fen: str,
        bad_actions: list[int],
        value_deltas: list[float],
    ) -> None:
        self.boards.append(board)
        self.actions.append(action)
        self.values.append(value)
        self.fens.append(fen)
        self.bad_actions.append(bad_actions)
        self.value_deltas.append(value_deltas)
        if len(self.actions) >= self.chunk_size:
            self._flush()

    def finish(self) -> list[Path]:
        if self.actions:
            self._flush()
        return self.paths

    def _flush(self) -> None:
        index = len(self.paths)
        max_bad_actions = max(len(actions) for actions in self.bad_actions)
        bad_actions = np.full((len(self.actions), max_bad_actions), -1, dtype=np.int64)
        value_deltas = np.zeros((len(self.actions), max_bad_actions), dtype=np.float32)
        for row, (actions, deltas) in enumerate(zip(self.bad_actions, self.value_deltas)):
            bad_actions[row, : len(actions)] = actions
            value_deltas[row, : len(deltas)] = deltas

        path = self.out_dir / f"model_blunders_{index:06d}.npz"
        np.savez_compressed(
            path,
            boards=np.asarray(self.boards, dtype=np.float32),
            actions=np.asarray(self.actions, dtype=np.int64),
            values=np.asarray(self.values, dtype=np.float32),
            fens=np.asarray(self.fens),
            bad_actions=bad_actions,
            value_deltas=value_deltas,
            source=np.asarray("model_blunders"),
        )
        self.paths.append(path)
        self.boards.clear()
        self.actions.clear()
        self.values.clear()
        self.fens.clear()
        self.bad_actions.clear()
        self.value_deltas.clear()
