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


def test_parse_c_puct_option() -> None:
    config = _parse_setoption(
        "setoption name CPuct value 0.75",
        UCIConfig(checkpoint="model.pt"),
    )
    assert config.c_puct == 0.75


def test_parse_policy_prior_temperature_option() -> None:
    config = _parse_setoption(
        "setoption name PolicyPriorTemperature value 2.5",
        UCIConfig(checkpoint="model.pt"),
    )
    assert config.policy_prior_temperature == 2.5


def test_parse_root_material_options() -> None:
    config = _parse_setoption(
        "setoption name MaterialValueSearchPlies value 2",
        UCIConfig(checkpoint="model.pt"),
    )
    config = _parse_setoption(
        "setoption name LeafMaterialValueWeight value 0.75",
        config,
    )
    config = _parse_setoption(
        "setoption name LeafMaterialSearchPlies value 3",
        config,
    )
    config = _parse_setoption(
        "setoption name RootMaterialSearchPlies value 2",
        config,
    )
    config = _parse_setoption(
        "setoption name RootMateSearchPlies value 5",
        config,
    )
    config = _parse_setoption(
        "setoption name RootMaterialMaxLossCp value 250",
        config,
    )
    config = _parse_setoption(
        "setoption name RootKingSafetySearchPlies value 1",
        config,
    )
    config = _parse_setoption(
        "setoption name RootKingSafetyMaxLossCp value 300",
        config,
    )
    config = _parse_setoption(
        "setoption name RootTacticalPriorWeight value 0.4",
        config,
    )
    config = _parse_setoption(
        "setoption name RootTacticalPriorTemperatureCp value 150",
        config,
    )

    assert config.material_value_search_plies == 2
    assert config.leaf_material_value_weight == 0.75
    assert config.leaf_material_search_plies == 3
    assert config.root_mate_search_plies == 5
    assert config.root_material_search_plies == 2
    assert config.root_material_max_loss_cp == 250
    assert config.root_king_safety_search_plies == 1
    assert config.root_king_safety_max_loss_cp == 300
    assert config.root_tactical_prior_weight == 0.4
    assert config.root_tactical_prior_temperature_cp == 150
