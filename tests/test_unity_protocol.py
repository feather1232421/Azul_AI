import json

import pytest

from config import FIRST_PLAYER
from logic import AzulGame
from reconstruction_test import TableData
from server import convert_move_to_action, handle_obs_message, validate_online_table_data


def area(color=0, empty=True):
    return {"empty": empty, "color": color}


def player(score, seat_id=None, client_id=None):
    result = {
        "score": score,
        "manualAreas": [
            [area() for _ in range(row + 1)]
            for row in range(5)
        ],
        "coloredAreas": [
            [area() for _ in range(5)]
            for _ in range(5)
        ],
        "loseAreas": [area() for _ in range(7)],
    }
    if seat_id is not None:
        result["seatId"] = seat_id
    if client_id is not None:
        result["clientId"] = client_id
    return result


def table_payload(player_count=3):
    center = [area(color=0, empty=False)]
    center.extend(area() for _ in range(23))
    center.append(area(color=5, empty=False))
    return {
        "totalPlayerCount": player_count,
        "factories": [
            [area() for _ in range(4)]
            for _ in range({2: 5, 3: 7, 4: 9}[player_count])
        ],
        "center": center,
        "me": player(score=10, seat_id=7, client_id=1234),
        "opponents": [
            player(score=20 + index, seat_id=42 + index, client_id=2000 + index)
            for index in range(player_count - 1)
        ],
        "remainTokens": [],
        "loseTokens": [],
    }


class FixedAgent:
    def decide(self, game):
        return "center", 5, 2


def test_new_protocol_preserves_opponent_order_and_reads_full_center():
    table_data = TableData(**table_payload(3))
    validate_online_table_data(table_data)

    game = AzulGame.from_table_data(table_data)

    assert game.num_players == 3
    assert [board.score for board in game.players] == [10, 20, 21]
    assert game.public_board.center == [FIRST_PLAYER, 5]


def test_server_returns_pascal_case_action_with_original_ids():
    reply = handle_obs_message(
        json.dumps(table_payload(3)),
        agent=FixedAgent(),
    )

    assert json.loads(reply) == {
        "ClientId": 1234,
        "SeatId": 7,
        "FactoryId": -1,
        "ColorType": 5,
        "Row": 2,
    }


def test_factory_and_floor_keep_legacy_sentinel_mapping():
    action = convert_move_to_action((4, 3, 5), client_id=1234, seat_id=7)

    payload = action.model_dump() if hasattr(action, "model_dump") else action.dict()
    assert payload == {
        "ClientId": 1234,
        "SeatId": 7,
        "FactoryId": 4,
        "ColorType": 3,
        "Row": -1,
    }


def test_online_protocol_rejects_player_count_mismatch_and_keeps_ids_in_error():
    payload = table_payload(3)
    payload["totalPlayerCount"] = 4

    reply = handle_obs_message(json.dumps(payload), agent=FixedAgent())

    assert json.loads(reply) == {
        "ClientId": 1234,
        "SeatId": 7,
        "FactoryId": -1,
        "ColorType": 0,
        "Row": -1,
    }


def test_shared_table_data_remains_compatible_with_legacy_payloads():
    payload = table_payload(2)
    payload.pop("totalPlayerCount")
    payload["me"].pop("seatId")
    payload["me"].pop("clientId")
    for opponent in payload["opponents"]:
        opponent.pop("seatId")
        opponent.pop("clientId")

    table_data = TableData(**payload)
    game = AzulGame.from_table_data(table_data)

    assert table_data.totalPlayerCount is None
    assert game.num_players == 2


@pytest.mark.parametrize("player_count", [2, 3, 4])
def test_factory_count_matches_explicit_player_count(player_count):
    table_data = TableData(**table_payload(player_count))
    validate_online_table_data(table_data)
    assert AzulGame.from_table_data(table_data).num_players == player_count
