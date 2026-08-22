import argparse
import gc
import json
import random
import time
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from battle import build_mcts_agent
from logic import AzulGame


def build_balanced_rosters(num_players):
    if num_players == 3:
        a_counts = (1, 2)
    elif num_players == 4:
        a_counts = (2,)
    else:
        raise ValueError(f"Multiplayer benchmark supports 3 or 4 players, got {num_players}")

    rosters = []
    seats = range(num_players)
    for a_count in a_counts:
        for a_seats in combinations(seats, a_count):
            a_seats = set(a_seats)
            rosters.append(tuple("A" if seat in a_seats else "B" for seat in seats))
    return rosters


def build_balanced_schedule(num_players, cycles=1, seed=42):
    if cycles < 1:
        raise ValueError("cycles must be at least 1")

    rng = random.Random(seed)
    rosters = build_balanced_rosters(num_players)
    scheduled_rosters = []
    for _ in range(cycles):
        cycle_rosters = list(rosters)
        rng.shuffle(cycle_rosters)
        scheduled_rosters.extend(cycle_rosters)

    model_starts = Counter()
    seat_starts = Counter()
    schedule = []
    for game_idx, roster in enumerate(scheduled_rosters):
        if model_starts["A"] == model_starts["B"]:
            desired_model = "A" if game_idx % 2 == 0 else "B"
        else:
            desired_model = min(("A", "B"), key=lambda label: model_starts[label])

        candidate_seats = [seat for seat, label in enumerate(roster) if label == desired_model]
        first_player = min(candidate_seats, key=lambda seat: (seat_starts[seat], seat))
        model_starts[desired_model] += 1
        seat_starts[first_player] += 1
        schedule.append({"roster": roster, "first_player": first_player})

    return schedule


def play_multiplayer_game(num_players, roster, first_player, agents_by_model):
    game = AzulGame(num_players=num_players)
    game.first_player = first_player
    game.current_player_idx = first_player
    agents_by_seat = {
        seat: agents_by_model[model_label]
        for seat, model_label in enumerate(roster)
    }

    while not game.is_game_over():
        game.advance_until_next_decision(agents_by_seat)

    scores = [player.look_score() for player in game.players]
    return {
        "roster": list(roster),
        "first_player": first_player,
        "scores": scores,
        "rank_values": game.get_rank_based_value_vector(),
        "winners": game.get_winners(),
    }


def summarize_results(results):
    model_stats = {
        "A": {"seats": 0, "score_sum": 0.0, "rank_value_sum": 0.0, "first_place_share": 0.0},
        "B": {"seats": 0, "score_sum": 0.0, "rank_value_sum": 0.0, "first_place_share": 0.0},
    }
    game_wins = Counter()
    score_margins = []

    for result in results:
        roster = result["roster"]
        winners = result["winners"]
        winner_models = {roster[seat] for seat in winners}
        if len(winner_models) == 1:
            game_wins[next(iter(winner_models))] += 1
        else:
            game_wins["draw"] += 1

        winner_share = 1.0 / len(winners) if winners else 0.0
        game_scores = {"A": [], "B": []}
        for seat, model_label in enumerate(roster):
            stats = model_stats[model_label]
            stats["seats"] += 1
            stats["score_sum"] += result["scores"][seat]
            stats["rank_value_sum"] += result["rank_values"][seat]
            game_scores[model_label].append(result["scores"][seat])
            if seat in winners:
                stats["first_place_share"] += winner_share

        mean_a = sum(game_scores["A"]) / len(game_scores["A"])
        mean_b = sum(game_scores["B"]) / len(game_scores["B"])
        score_margins.append(mean_a - mean_b)

    models = {}
    for label, stats in model_stats.items():
        seats = max(stats["seats"], 1)
        models[label] = {
            "game_wins": game_wins[label],
            "seat_appearances": stats["seats"],
            "avg_score": stats["score_sum"] / seats,
            "avg_rank_value": stats["rank_value_sum"] / seats,
            "first_place_share": stats["first_place_share"],
        }

    return {
        "games": len(results),
        "draws": game_wins["draw"],
        "models": models,
        "model_a_avg_score_margin": sum(score_margins) / len(score_margins) if score_margins else 0.0,
    }


def run_benchmark(
    model_a_path,
    model_b_path,
    num_players,
    cycles,
    sims,
    worlds,
    puct_c,
    prior_temperature,
    seed,
    device,
    verbose=False,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model_a_path = Path(model_a_path)
    model_b_path = Path(model_b_path)
    print(f"Device: {device}")
    agent_a = build_mcts_agent(
        model_a_path, 0, device, sims, worlds, puct_c, prior_temperature
    )
    agent_b = build_mcts_agent(
        model_b_path, 0, device, sims, worlds, puct_c, prior_temperature
    )

    schedule = build_balanced_schedule(num_players, cycles=cycles, seed=seed)
    results = []
    start_time = time.time()
    iterator = enumerate(schedule, start=1)
    if not verbose:
        iterator = tqdm(iterator, total=len(schedule), desc=f"{num_players}P arena")

    for game_idx, item in iterator:
        result = play_multiplayer_game(
            num_players=num_players,
            roster=item["roster"],
            first_player=item["first_player"],
            agents_by_model={"A": agent_a, "B": agent_b},
        )
        results.append(result)
        if verbose:
            print(
                f"[{game_idx:03d}/{len(schedule):03d}] "
                f"roster={''.join(item['roster'])} first=P{item['first_player']} "
                f"scores={result['scores']} winners={result['winners']}"
            )
        gc.collect()

    elapsed = time.time() - start_time
    summary = summarize_results(results)
    summary.update({
        "players": num_players,
        "cycles": cycles,
        "model_a": model_a_path.name,
        "model_b": model_b_path.name,
        "sims": sims,
        "worlds": worlds,
        "seed": seed,
        "elapsed_seconds": elapsed,
        "avg_seconds_per_game": elapsed / max(len(results), 1),
    })
    return summary, results


def print_summary(summary):
    a = summary["models"]["A"]
    b = summary["models"]["B"]
    print("\nMultiplayer Arena Summary")
    print(f"  Table: {summary['players']}P, games={summary['games']}, sims={summary['sims']}, worlds={summary['worlds']}")
    print(f"  A: {summary['model_a']}")
    print(f"  B: {summary['model_b']}")
    print(f"  Game wins: A={a['game_wins']} B={b['game_wins']} draws={summary['draws']}")
    print(f"  Avg score: A={a['avg_score']:.2f} B={b['avg_score']:.2f} margin={summary['model_a_avg_score_margin']:+.2f}")
    print(f"  Avg rank value: A={a['avg_rank_value']:+.3f} B={b['avg_rank_value']:+.3f}")
    print(f"  First-place share: A={a['first_place_share']:.2f} B={b['first_place_share']:.2f}")
    print(f"  Avg time/game: {summary['avg_seconds_per_game']:.2f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Seat-balanced 3P/4P arena between two 300-action models."
    )
    parser.add_argument(
        "--model-a",
        default="models/transformer_action300_mixed_round1_last.pt",
    )
    parser.add_argument(
        "--model-b",
        default="models/transformer_action300_legacy2p_distill_last.pt",
    )
    parser.add_argument("--players", type=int, choices=[3, 4], default=3)
    parser.add_argument("--cycles", type=int, default=1, help="One cycle is six seat-balanced games.")
    parser.add_argument("--sims", type=int, default=80)
    parser.add_argument("--worlds", type=int, default=2)
    parser.add_argument("--puct-c", type=float, default=1.0)
    parser.add_argument("--prior-temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--json-output", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary, results = run_benchmark(
        model_a_path=args.model_a,
        model_b_path=args.model_b,
        num_players=args.players,
        cycles=args.cycles,
        sims=args.sims,
        worlds=args.worlds,
        puct_c=args.puct_c,
        prior_temperature=args.prior_temperature,
        seed=args.seed,
        device=device,
        verbose=args.verbose,
    )
    print_summary(summary)

    if args.json_output:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump({"summary": summary, "games": results}, f, ensure_ascii=True, indent=2)
        print(f"  JSON: {output_path}")


if __name__ == "__main__":
    main()
