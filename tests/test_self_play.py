from alpha_chess.evaluator import UniformEvaluator
from alpha_chess.self_play import (
    SelfPlayConfig,
    generate_self_play,
    generate_self_play_from_checkpoint,
)


def test_generate_self_play_uses_workers(tmp_path) -> None:
    paths = generate_self_play(
        UniformEvaluator(),
        tmp_path,
        SelfPlayConfig(games=3, simulations=1, max_plies=4, seed=7, workers=2),
    )

    assert len(paths) == 3
    assert [path.name.endswith(f"_{index:06d}.npz") for index, path in enumerate(paths)] == [
        True,
        True,
        True,
    ]
    assert all(path.exists() for path in paths)


def test_generate_self_play_from_checkpoint_uses_process_workers(tmp_path) -> None:
    paths = generate_self_play_from_checkpoint(
        None,
        tmp_path,
        SelfPlayConfig(games=3, simulations=1, max_plies=4, seed=7, workers=2),
        device="cpu",
    )

    assert len(paths) == 3
    assert [path.name.endswith(f"_{index:06d}.npz") for index, path in enumerate(paths)] == [
        True,
        True,
        True,
    ]
    assert all(path.exists() for path in paths)
