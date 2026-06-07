import argparse
import json
import pickle
from pathlib import Path

from curated_cases import build_training_sample, cell, empty_cell
from logic import AzulGame
from reconstruction_test import TableData


DEFAULT_TOKEN_COUNTS = {str(color): 0 for color in range(1, 6)}


def parse_move(text):
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 3:
        raise ValueError("Move must have three comma-separated fields: source,color,destination")

    source_text, color_text, destination_text = parts
    source = "center" if source_text.lower() in {"center", "c", "-1"} else int(source_text)
    color = int(color_text)
    if destination_text.lower() in {"floor", "f", "-1"}:
        destination = 5
    else:
        destination = int(destination_text)
    return source, color, destination


def move_to_compact(move):
    source, color, destination = move
    return [
        source,
        int(color),
        "floor" if destination == 5 else int(destination),
    ]


def compact_move_to_tuple(move):
    source, color, destination = move
    if isinstance(source, str) and source.lower() in {"center", "c", "-1"}:
        source = "center"
    else:
        source = int(source)
    if isinstance(destination, str) and destination.lower() in {"floor", "f", "-1"}:
        destination = 5
    else:
        destination = int(destination)
    return source, int(color), destination


def token_cell(value):
    value = int(value)
    return empty_cell() if value == 0 else cell(value)


def wall_cell(value):
    return empty_cell() if int(value) == 0 else cell(1)


def pad(values, length, fill=0):
    values = list(values)
    return values[:length] + [fill] * max(0, length - len(values))


def compact_player_to_payload(player):
    return {
        "score": int(player.get("score", 0)),
        "manualAreas": [
            [token_cell(value) for value in pad(row, row_idx + 1)]
            for row_idx, row in enumerate(player["manualAreas"])
        ],
        "coloredAreas": [
            [wall_cell(value) for value in pad(row, 5)]
            for row in pad(player["coloredAreas"], 5, [0, 0, 0, 0, 0])
        ],
        "loseAreas": [token_cell(value) for value in pad(player.get("loseAreas", []), 7)],
    }


def token_counts_to_payload(counts):
    if isinstance(counts, list):
        return counts
    merged = dict(DEFAULT_TOKEN_COUNTS)
    merged.update({str(k): int(v) for k, v in counts.items()})
    return [{"color": int(color), "number": int(merged[str(color)])} for color in range(1, 6)]


def compact_payload_to_table_payload(payload):
    factories = [
        [token_cell(value) for value in pad(factory, 4)]
        for factory in payload["factories"]
    ]
    return {
        "factories": factories,
        "center": [token_cell(value) for value in pad(payload.get("center", []), 24)],
        "me": compact_player_to_payload(payload["me"]),
        "opponents": [compact_player_to_payload(player) for player in payload.get("opponents", [])],
        "remainTokens": token_counts_to_payload(payload.get("remainTokens", DEFAULT_TOKEN_COUNTS)),
        "loseTokens": token_counts_to_payload(payload.get("loseTokens", DEFAULT_TOKEN_COUNTS)),
    }


def table_payload_to_compact(payload):
    def compact_area(area):
        return [0 if item.get("empty") else int(item.get("color", 0)) for item in area]

    def compact_wall(area):
        return [0 if item.get("empty") else 1 for item in area]

    def compact_player(player):
        return {
            "score": int(player.get("score", 0)),
            "manualAreas": [compact_area(row) for row in player["manualAreas"]],
            "coloredAreas": [compact_wall(row) for row in player["coloredAreas"]],
            "loseAreas": [value for value in compact_area(player.get("loseAreas", [])) if value != 0],
        }

    def compact_counts(rows):
        return {str(item["color"]): int(item["number"]) for item in rows}

    return {
        "factories": [compact_area(row) for row in payload["factories"]],
        "center": [value for value in compact_area(payload.get("center", [])) if value != 0],
        "me": compact_player(payload["me"]),
        "opponents": [compact_player(player) for player in payload.get("opponents", [])],
        "remainTokens": compact_counts(payload.get("remainTokens", [])),
        "loseTokens": compact_counts(payload.get("loseTokens", [])),
    }


def load_case(path):
    with Path(path).open("r", encoding="utf-8") as f:
        case = json.load(f)
    case = dict(case)
    case["best_move"] = compact_move_to_tuple(case["best_move"])
    case["payload"] = compact_payload_to_table_payload(case["payload"])
    return case


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def make_template(case_id, best_move, players=2):
    empty_manual = [[0], [0, 0], [0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0, 0]]
    empty_wall = [[0, 0, 0, 0, 0] for _ in range(5)]
    empty_player = {
        "score": 0,
        "manualAreas": empty_manual,
        "coloredAreas": empty_wall,
        "loseAreas": [],
    }
    return {
        "id": case_id,
        "category": "expert_video",
        "description": "",
        "current_player": 0,
        "best_move": move_to_compact(best_move),
        "z_vec": [0.0, 0.0, 0.0, 0.0],
        "payload": {
            "factories": [[0, 0, 0, 0] for _ in range(5 if players <= 2 else 7 if players == 3 else 9)],
            "center": [6],
            "me": empty_player,
            "opponents": [empty_player for _ in range(players - 1)],
            "remainTokens": DEFAULT_TOKEN_COUNTS,
            "loseTokens": DEFAULT_TOKEN_COUNTS,
        },
    }


def case_from_log(log_path, line_number, case_id, best_move=None):
    lines = Path(log_path).read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[line_number - 1])
    request_payload = json.loads(row["request_raw"])
    if best_move is None:
        response = json.loads(row["response_raw"])
        best_move = (
            "center" if response["sourceId"] == -1 else int(response["sourceId"]),
            int(response["color"]),
            5 if response["destinationId"] == -1 else int(response["destinationId"]),
        )
    return {
        "id": case_id,
        "category": "marked_position",
        "description": f"Marked from {Path(log_path).name}:{line_number}",
        "current_player": 0,
        "best_move": move_to_compact(best_move),
        "z_vec": [0.0, 0.0, 0.0, 0.0],
        "payload": table_payload_to_compact(request_payload),
    }


def print_case(case):
    game = AzulGame.from_table_data(TableData(**case["payload"]))
    print(f"id: {case['id']}")
    print(f"category: {case.get('category', '')}")
    print(f"description: {case.get('description', '')}")
    print(f"best_move: {case['best_move']}")
    print(f"scores: {[player.score for player in game.players]}")
    print(f"factories: {game.public_board.factories}")
    print(f"center: {game.public_board.center}")
    for idx, player in enumerate(game.players):
        label = "me" if idx == 0 else f"opponent_{idx}"
        print(f"{label}: score={player.score} pattern={player.pattern_lines} floor={player.floor}")
        print(f"{label} wall:")
        for row in player.wall:
            print(f"  {row}")
    print("legal_moves:")
    for move in game.get_legal_moves():
        marker = "*" if move == case["best_move"] else " "
        print(f"{marker} {move}")


def validate_case(path):
    case = load_case(path)
    game = AzulGame.from_table_data(TableData(**case["payload"]))
    legal_moves = game.get_legal_moves()
    if case["best_move"] not in legal_moves:
        raise ValueError(f"{path}: best_move {case['best_move']} is not legal. Legal moves: {legal_moves}")
    build_training_sample(case)
    print(f"OK {path}: best_move={case['best_move']} legal_moves={len(legal_moves)}")


def export_cases(paths, output_path, by_episode=False, best_prob=1.0):
    samples = []
    manifest_rows = []
    for path in paths:
        case = load_case(path)
        sample = build_training_sample(case, best_prob=best_prob)
        samples.append([sample] if by_episode else sample)
        manifest_rows.append({
            "id": case["id"],
            "category": case.get("category", ""),
            "best_move": list(case["best_move"]),
            "description": case.get("description", ""),
            "source": str(path),
        })

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(samples, f)

    manifest_path = output_path.with_suffix(".jsonl")
    with manifest_path.open("w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved {len(samples)} samples to {output_path}")
    print(f"Saved manifest to {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description="Record, inspect, validate, and export compact Azul curated positions.")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    template_parser = subparsers.add_parser("template")
    template_parser.add_argument("--output", required=True)
    template_parser.add_argument("--id", required=True)
    template_parser.add_argument("--best-move", required=True, help="Example: center,5,floor or 1,3,4")
    template_parser.add_argument("--players", type=int, default=2)

    from_log_parser = subparsers.add_parser("from-log")
    from_log_parser.add_argument("--log", required=True)
    from_log_parser.add_argument("--line", type=int, required=True)
    from_log_parser.add_argument("--output", required=True)
    from_log_parser.add_argument("--id", required=True)
    from_log_parser.add_argument("--best-move", default=None)

    show_parser = subparsers.add_parser("show")
    show_parser.add_argument("case")

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("cases", nargs="+")

    export_parser = subparsers.add_parser("export")
    export_parser.add_argument("cases", nargs="+")
    export_parser.add_argument("--output", required=True)
    export_parser.add_argument("--by-episode", action="store_true")
    export_parser.add_argument("--best-prob", type=float, default=1.0)

    args = parser.parse_args()

    if args.cmd == "template":
        write_json(args.output, make_template(args.id, parse_move(args.best_move), players=args.players))
    elif args.cmd == "from-log":
        best_move = parse_move(args.best_move) if args.best_move else None
        write_json(args.output, case_from_log(args.log, args.line, args.id, best_move=best_move))
    elif args.cmd == "show":
        print_case(load_case(args.case))
    elif args.cmd == "validate":
        for path in args.cases:
            validate_case(path)
    elif args.cmd == "export":
        export_cases(args.cases, args.output, by_episode=args.by_episode, best_prob=args.best_prob)


if __name__ == "__main__":
    main()
