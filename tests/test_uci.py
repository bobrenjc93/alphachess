import chess

from alpha_chess.uci import UCIConfig, _parse_position, _parse_setoption


def test_parse_position_startpos_moves() -> None:
    board = _parse_position("position startpos moves e2e4 e7e5")
    assert board.piece_at(chess.E4).piece_type == chess.PAWN
    assert board.piece_at(chess.E5).piece_type == chess.PAWN
    assert board.turn == chess.WHITE


def test_parse_position_fen() -> None:
    board = _parse_position("position fen 8/8/8/8/8/8/8/K6k w - - 0 1")
    assert board.king(chess.WHITE) == chess.A1
    assert board.king(chess.BLACK) == chess.H1


def test_parse_simulations_option() -> None:
    config = _parse_setoption(
        "setoption name Simulations value 17",
        UCIConfig(checkpoint="model.pt", simulations=1),
    )
    assert config.simulations == 17


def test_parse_root_material_options() -> None:
    config = _parse_setoption(
        "setoption name MaterialValueSearchPlies value 2",
        UCIConfig(checkpoint="model.pt"),
    )
    config = _parse_setoption(
        "setoption name RootMaterialSearchPlies value 2",
        config,
    )
    config = _parse_setoption(
        "setoption name RootMaterialMaxLossCp value 250",
        config,
    )

    assert config.material_value_search_plies == 2
    assert config.root_material_search_plies == 2
    assert config.root_material_max_loss_cp == 250
