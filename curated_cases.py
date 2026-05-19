from config import REVERSE_LOOKUP
from logic import AzulGame
from reconstruction_test import TableData
import numpy as np


def cell(color):
    return {"empty": False, "color": color}


def empty_cell():
    return {"empty": True, "color": 0}


END_ROUND_FIRST_PLAYER_CASE = {
    "id": "end_round_first_player_floor_choice",
    "category": "search_bug_regression",
    "description": (
        "End-of-round floor choice. Current player should take 1 tile to floor and force the "
        "opponent to take 4, instead of eating the 4-tile floor penalty themselves."
    ),
    "best_move": ("center", 2, 5),
    "z": 1.0,
    "payload": {
        "factories": [[empty_cell() for _ in range(4)] for _ in range(5)],
        "center": [cell(2), cell(4), cell(4), cell(4), cell(4)] + [empty_cell() for _ in range(19)],
        "me": {
            "score": 2,
            "manualAreas": [
                [cell(4)],
                [cell(2), cell(2)],
                [cell(1), cell(1), empty_cell()],
                [cell(1), cell(1), cell(1), empty_cell()],
                [cell(3), cell(3), empty_cell(), empty_cell(), empty_cell()],
            ],
            "coloredAreas": [
                [empty_cell(), empty_cell(), cell(1), empty_cell(), empty_cell()],
                [empty_cell(), empty_cell(), empty_cell(), cell(1), empty_cell()],
                [cell(1), empty_cell(), empty_cell(), empty_cell(), empty_cell()],
                [empty_cell(), empty_cell(), empty_cell(), empty_cell(), empty_cell()],
                [empty_cell(), empty_cell(), empty_cell(), empty_cell(), empty_cell()],
            ],
            "loseAreas": [empty_cell() for _ in range(7)],
        },
        "opponents": [
            {
                "score": 1,
                "manualAreas": [
                    [cell(3)],
                    [cell(1), cell(1)],
                    [cell(1), cell(1), empty_cell()],
                    [cell(2), cell(2), cell(2), cell(2)],
                    [cell(5), cell(5), cell(5), cell(5), cell(5)],
                ],
                "coloredAreas": [
                    [empty_cell(), empty_cell(), empty_cell(), cell(1), empty_cell()],
                    [empty_cell(), empty_cell(), cell(1), empty_cell(), empty_cell()],
                    [empty_cell(), empty_cell(), empty_cell(), empty_cell(), empty_cell()],
                    [empty_cell(), empty_cell(), empty_cell(), empty_cell(), empty_cell()],
                    [empty_cell(), empty_cell(), empty_cell(), empty_cell(), empty_cell()],
                ],
                "loseAreas": [cell(6), cell(5)] + [empty_cell() for _ in range(5)],
            }
        ],
        "remainTokens": [{"color": color, "number": 12} for color in range(1, 6)],
        "loseTokens": [],
    },
}


WHITE_PLACEMENT_CASE = {
    "id": "white_three_tiles_fifth_row_vs_second_row",
    "category": "policy_quality_case",
    "description": (
        "Three white tiles from a factory can either finish row 5 cleanly or overfill row 2. "
        "Row 5 is the stronger local choice for this position."
    ),
    "best_move": (3, 5, 4),
    "z": 0.0,
    "payload": {
        "factories": [
            [empty_cell(), empty_cell(), empty_cell(), empty_cell()],
            [empty_cell(), empty_cell(), empty_cell(), empty_cell()],
            [empty_cell(), empty_cell(), empty_cell(), empty_cell()],
            [cell(5), cell(5), cell(5), cell(1)],
            [empty_cell(), empty_cell(), empty_cell(), empty_cell()],
        ],
        "center": [cell(2), cell(1), cell(1)] + [empty_cell() for _ in range(21)],
        "me": {
            "score": 29,
            "manualAreas": [
                [cell(4)],
                [cell(5), empty_cell()],
                [cell(3), cell(3), cell(3)],
                [empty_cell(), empty_cell(), empty_cell(), empty_cell()],
                [cell(5), cell(5), empty_cell(), empty_cell(), empty_cell()],
            ],
            "coloredAreas": [
                [cell(1), cell(1), cell(1), empty_cell(), empty_cell()],
                [empty_cell(), cell(1), cell(1), cell(1), empty_cell()],
                [empty_cell(), cell(1), cell(1), cell(1), empty_cell()],
                [empty_cell(), empty_cell(), cell(1), cell(1), empty_cell()],
                [empty_cell(), empty_cell(), empty_cell(), empty_cell(), empty_cell()],
            ],
            "loseAreas": [cell(3)] + [empty_cell() for _ in range(6)],
        },
        "opponents": [
            {
                "score": 11,
                "manualAreas": [
                    [empty_cell()],
                    [cell(3), cell(3)],
                    [empty_cell(), empty_cell(), empty_cell()],
                    [cell(2), cell(2), cell(2), cell(2)],
                    [cell(4), cell(4), cell(4), cell(4), cell(4)],
                ],
                "coloredAreas": [
                    [empty_cell(), cell(1), empty_cell(), cell(1), empty_cell()],
                    [cell(1), cell(1), empty_cell(), empty_cell(), cell(1)],
                    [cell(1), empty_cell(), empty_cell(), cell(1), empty_cell()],
                    [cell(1), cell(1), empty_cell(), empty_cell(), empty_cell()],
                    [empty_cell(), empty_cell(), empty_cell(), empty_cell(), empty_cell()],
                ],
                "loseAreas": [cell(6)] + [empty_cell() for _ in range(6)],
            }
        ],
        "remainTokens": [{"color": color, "number": 4} for color in range(1, 6)],
        "loseTokens": [],
    },
}


CURATED_CASES = [
    END_ROUND_FIRST_PLAYER_CASE,
    WHITE_PLACEMENT_CASE,
]


def build_game_from_case(case):
    payload = case["payload"]
    state = TableData(**payload)
    return AzulGame.from_table_data(state)


def build_policy_target(best_move, legal_moves, best_prob=1.0):
    pi = np.zeros(180, dtype=np.float32)
    if best_prob >= 1.0 or len(legal_moves) <= 1:
        pi[REVERSE_LOOKUP[best_move]] = 1.0
        return pi

    spill = (1.0 - best_prob) / max(len(legal_moves) - 1, 1)
    for move in legal_moves:
        pi[REVERSE_LOOKUP[move]] = spill
    pi[REVERSE_LOOKUP[best_move]] = best_prob
    return pi


def build_mask(legal_moves):
    mask = np.zeros(180, dtype=np.float32)
    for move in legal_moves:
        mask[REVERSE_LOOKUP[move]] = 1.0
    return mask


def build_training_sample(case, best_prob=1.0):
    game = build_game_from_case(case)
    obs = game.state_to_vector_np(game.get_observation_current())
    legal_moves = game.get_legal_moves()
    best_move = case["best_move"]
    if best_move not in legal_moves:
        raise ValueError(f"Best move {best_move} is not legal for case {case['id']}")
    pi = build_policy_target(best_move, legal_moves, best_prob=best_prob)
    mask = build_mask(legal_moves)
    z = float(case.get("z", 0.0))
    return obs, pi, z, mask
