from collections import Counter

import pytest

from benchmark_multiplayer import build_balanced_schedule, summarize_results


@pytest.mark.parametrize("players", [3, 4])
def test_schedule_balances_model_seats_and_first_players(players):
    schedule = build_balanced_schedule(players, cycles=1, seed=7)

    assert len(schedule) == 6
    seat_appearances = Counter(label for item in schedule for label in item["roster"])
    model_starts = Counter(item["roster"][item["first_player"]] for item in schedule)
    assert seat_appearances == {"A": len(schedule) * players // 2, "B": len(schedule) * players // 2}
    assert model_starts == {"A": 3, "B": 3}


def test_summary_uses_per_model_seat_averages():
    results = [
        {
            "roster": ["A", "A", "B"],
            "scores": [60, 30, 45],
            "rank_values": [1.0, -1.0, 0.0],
            "winners": [0],
        },
        {
            "roster": ["A", "B", "B"],
            "scores": [30, 50, 40],
            "rank_values": [-1.0, 1.0, 0.0],
            "winners": [1],
        },
    ]

    summary = summarize_results(results)

    assert summary["models"]["A"]["game_wins"] == 1
    assert summary["models"]["B"]["game_wins"] == 1
    assert summary["models"]["A"]["avg_score"] == 40.0
    assert summary["models"]["B"]["avg_score"] == 45.0
    assert summary["model_a_avg_score_margin"] == pytest.approx(-7.5)
