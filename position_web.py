import json
import pickle
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from curated_cases import build_training_sample
from logic import AzulGame
from position_tool import (
    case_from_log,
    compact_move_to_tuple,
    compact_payload_to_table_payload,
    export_cases,
    load_case,
    make_template,
    move_to_compact,
    parse_move,
    write_json,
)
from reconstruction_test import TableData


ROOT = Path(__file__).resolve().parent
CASE_DIR = ROOT / "artifacts" / "positions"
EPISODE_DIR = ROOT / "artifacts" / "episodes"
LOG_DIR = ROOT / "unity_logs"
STATIC_DIR = ROOT / "position_web_static"


app = FastAPI(title="Azul Position Lab")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class TemplateRequest(BaseModel):
    id: str
    best_move: str
    players: int = 2


class EmptyCaseRequest(BaseModel):
    id: str = "new_position_001"
    best_move: str = "center,1,floor"
    players: int = 2


class FromLogRequest(BaseModel):
    log: str
    line: int
    id: str
    best_move: str | None = None


class SaveCaseRequest(BaseModel):
    case: dict


class MoveRequest(BaseModel):
    case: dict
    move: list


class NewEpisodeRequest(BaseModel):
    id: str
    description: str = ""


class SaveEpisodeRequest(BaseModel):
    episode: dict


class AddStepRequest(BaseModel):
    case: dict
    note: str = ""


class ExportRequest(BaseModel):
    cases: list[str]
    output: str = "artifacts/curated_positions/web_cases.pkl"
    by_episode: bool = True
    best_prob: float = 1.0


class ExportEpisodesRequest(BaseModel):
    episodes: list[str]
    output: str = "artifacts/curated_positions/web_episodes.pkl"
    best_prob: float = 1.0


def safe_name(name: str, suffix=".json") -> str:
    path = Path(name)
    if path.name != name:
        raise HTTPException(status_code=400, detail="Use a file name, not a path.")
    if suffix and path.suffix != suffix:
        name = f"{path.stem}{suffix}"
    return name


def case_path(name: str) -> Path:
    return CASE_DIR / safe_name(name)


def episode_path(name: str) -> Path:
    return EPISODE_DIR / safe_name(name)


def log_path(name: str) -> Path:
    name = safe_name(name, suffix="")
    path = LOG_DIR / name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Log not found.")
    return path


def read_compact_case(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_episode(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_episode(episode: dict) -> dict:
    episode = dict(episode)
    if "id" not in episode:
        raise HTTPException(status_code=400, detail="Episode is missing id.")
    episode.setdefault("description", "")
    episode.setdefault("steps", [])
    for idx, step in enumerate(episode["steps"], 1):
        if "case" not in step:
            raise HTTPException(status_code=400, detail=f"Step {idx} is missing case.")
        normalize_case(step["case"])
        step.setdefault("step", idx)
        step.setdefault("note", "")
    return episode


def normalize_case(case: dict) -> dict:
    case = dict(case)
    if "payload" not in case:
        raise HTTPException(status_code=400, detail="Case is missing payload.")
    if "best_move" not in case:
        raise HTTPException(status_code=400, detail="Case is missing best_move.")
    case.setdefault("current_player", 0)
    player_count = 1 + len(case["payload"].get("opponents", []))
    if int(case["current_player"]) < 0 or int(case["current_player"]) >= player_count:
        raise HTTPException(status_code=400, detail="current_player is out of range.")
    compact_move_to_tuple(case["best_move"])
    compact_payload_to_table_payload(case["payload"])
    return case


def table_case_from_compact(case: dict) -> dict:
    normalized = normalize_case(case)
    return {
        **normalized,
        "best_move": compact_move_to_tuple(normalized["best_move"]),
        "payload": compact_payload_to_table_payload(payload_for_current_player(normalized)),
    }


def build_game_from_compact(case: dict) -> AzulGame:
    payload = compact_payload_to_table_payload(payload_for_current_player(case))
    return AzulGame.from_table_data(TableData(**payload))


def payload_for_current_player(case: dict) -> dict:
    payload = case["payload"]
    current_player = int(case.get("current_player", 0))
    players = [payload["me"], *payload.get("opponents", [])]
    if current_player < 0 or current_player >= len(players):
        raise HTTPException(status_code=400, detail="current_player is out of range.")
    if current_player == 0:
        return payload
    ordered = players[current_player:] + players[:current_player]
    return {
        **payload,
        "me": ordered[0],
        "opponents": ordered[1:],
    }


def game_summary(game: AzulGame) -> dict:
    return {
        "scores": [player.score for player in game.players],
        "factories": game.public_board.factories,
        "center": game.public_board.center,
        "players": [
            {
                "score": player.score,
                "pattern_lines": player.pattern_lines,
                "floor": player.floor,
                "wall": player.wall,
            }
            for player in game.players
        ],
        "legal_moves": [move_to_compact(move) for move in game.get_legal_moves()],
    }


def validate_compact_case(case: dict) -> dict:
    table_case = table_case_from_compact(case)
    game = AzulGame.from_table_data(TableData(**table_case["payload"]))
    legal_moves = game.get_legal_moves()
    if table_case["best_move"] not in legal_moves:
        return {
            "ok": False,
            "error": "best_move is not legal",
            "best_move": move_to_compact(table_case["best_move"]),
            "legal_moves": [move_to_compact(move) for move in legal_moves],
            "summary": game_summary(game),
        }
    build_training_sample(table_case)
    return {
        "ok": True,
        "best_move": move_to_compact(table_case["best_move"]),
        "legal_moves": [move_to_compact(move) for move in legal_moves],
        "summary": game_summary(game),
    }


def validate_episode(episode: dict) -> dict:
    episode = normalize_episode(episode)
    results = []
    ok = True
    for step in episode["steps"]:
        result = validate_compact_case(step["case"])
        ok = ok and result["ok"]
        results.append({
            "step": step.get("step"),
            "note": step.get("note", ""),
            "ok": result["ok"],
            "error": result.get("error"),
            "best_move": result.get("best_move"),
        })
    return {
        "ok": ok,
        "steps": results,
        "count": len(results),
    }


def export_episodes(paths, output_path, best_prob=1.0):
    dataset = []
    manifest_rows = []
    for path in paths:
        episode = normalize_episode(read_episode(path))
        samples = []
        for idx, step in enumerate(episode["steps"], 1):
            table_case = table_case_from_compact(step["case"])
            samples.append(build_training_sample(table_case, best_prob=best_prob))
            manifest_rows.append({
                "episode_id": episode["id"],
                "step": step.get("step", idx),
                "note": step.get("note", ""),
                "case_id": step["case"].get("id", ""),
                "best_move": list(table_case["best_move"]),
                "source": path.name,
            })
        if samples:
            dataset.append(samples)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(dataset, f)

    manifest_path = output_path.with_suffix(".jsonl")
    with manifest_path.open("w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "output": output_path,
        "manifest": manifest_path,
        "episodes": len(dataset),
        "samples": len(manifest_rows),
    }


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/builder")
def builder():
    return FileResponse(STATIC_DIR / "builder.html")


def player_to_compact(player) -> dict:
    return {
        "score": int(player.score),
        "manualAreas": [row[:] for row in player.pattern_lines],
        "coloredAreas": [[1 if cell else 0 for cell in row] for row in player.wall],
        "loseAreas": player.floor[:],
    }


def game_to_compact_payload(game: AzulGame) -> dict:
    remain = {str(color): game.public_board.bag.count(color) for color in range(1, 6)}
    discard = {str(color): game.public_board.discard_pile.count(color) for color in range(1, 6)}
    return {
        "factories": [factory[:] for factory in game.public_board.factories],
        "center": game.public_board.center[:],
        "me": player_to_compact(game.players[0]),
        "opponents": [player_to_compact(player) for player in game.players[1:]],
        "remainTokens": remain,
        "loseTokens": discard,
    }


@app.get("/api/cases")
def list_cases():
    CASE_DIR.mkdir(parents=True, exist_ok=True)
    return [
        {
            "name": path.name,
            "mtime": path.stat().st_mtime,
            "size": path.stat().st_size,
        }
        for path in sorted(CASE_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    ]


@app.get("/api/cases/{name}")
def get_case(name: str):
    path = case_path(name)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Case not found.")
    case = read_compact_case(path)
    result = validate_compact_case(case)
    return {"case": case, "validation": result}


@app.put("/api/cases/{name}")
def save_case(name: str, request: SaveCaseRequest):
    path = case_path(name)
    case = normalize_case(request.case)
    CASE_DIR.mkdir(parents=True, exist_ok=True)
    write_json(path, case)
    return {"saved": path.relative_to(ROOT).as_posix(), "validation": validate_compact_case(case)}


@app.delete("/api/cases/{name}")
def delete_case(name: str):
    path = case_path(name)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Case not found.")
    path.unlink()
    return {"deleted": path.name}


@app.get("/api/episodes")
def list_episodes():
    EPISODE_DIR.mkdir(parents=True, exist_ok=True)
    return [
        {
            "name": path.name,
            "mtime": path.stat().st_mtime,
            "size": path.stat().st_size,
            "steps": len(read_episode(path).get("steps", [])),
        }
        for path in sorted(EPISODE_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    ]


@app.get("/api/episodes/{name}")
def get_episode(name: str):
    path = episode_path(name)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Episode not found.")
    episode = read_episode(path)
    return {"episode": episode, "validation": validate_episode(episode)}


@app.post("/api/episodes")
def create_episode(request: NewEpisodeRequest):
    episode = {
        "id": request.id,
        "description": request.description,
        "steps": [],
    }
    name = safe_name(f"{request.id}.json")
    path = EPISODE_DIR / name
    EPISODE_DIR.mkdir(parents=True, exist_ok=True)
    write_json(path, episode)
    return {"name": name, "episode": episode, "validation": validate_episode(episode)}


@app.put("/api/episodes/{name}")
def save_episode(name: str, request: SaveEpisodeRequest):
    path = episode_path(name)
    episode = normalize_episode(request.episode)
    EPISODE_DIR.mkdir(parents=True, exist_ok=True)
    write_json(path, episode)
    return {"saved": path.relative_to(ROOT).as_posix(), "validation": validate_episode(episode)}


@app.delete("/api/episodes/{name}")
def delete_episode(name: str):
    path = episode_path(name)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Episode not found.")
    path.unlink()
    return {"deleted": path.name}


@app.post("/api/episodes/{name}/steps")
def add_episode_step(name: str, request: AddStepRequest):
    path = episode_path(name)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Episode not found.")
    episode = normalize_episode(read_episode(path))
    case = normalize_case(request.case)
    next_step = len(episode["steps"]) + 1
    episode["steps"].append({
        "step": next_step,
        "note": request.note,
        "case": case,
    })
    write_json(path, episode)
    return {"episode": episode, "validation": validate_episode(episode)}


@app.delete("/api/episodes/{name}/steps/{step_index}")
def delete_episode_step(name: str, step_index: int):
    path = episode_path(name)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Episode not found.")
    episode = normalize_episode(read_episode(path))
    idx = step_index - 1
    if idx < 0 or idx >= len(episode["steps"]):
        raise HTTPException(status_code=404, detail="Step not found.")
    episode["steps"].pop(idx)
    for new_idx, step in enumerate(episode["steps"], 1):
        step["step"] = new_idx
    write_json(path, episode)
    return {"episode": episode, "validation": validate_episode(episode)}


@app.post("/api/cases/{name}/validate")
def validate_case_file(name: str):
    path = case_path(name)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Case not found.")
    return validate_compact_case(read_compact_case(path))


@app.post("/api/validate")
def validate_case_body(request: SaveCaseRequest):
    return validate_compact_case(request.case)


@app.post("/api/empty-case")
def empty_case(request: EmptyCaseRequest = EmptyCaseRequest()):
    case = make_template(request.id, parse_move(request.best_move), players=request.players)
    return {"case": case, "summary": game_summary(build_game_from_compact(case))}


@app.post("/api/case-summary")
def case_summary(request: SaveCaseRequest):
    case = normalize_case(request.case)
    return game_summary(build_game_from_compact(case))


@app.post("/api/apply-move")
def apply_move(request: MoveRequest):
    case = normalize_case(request.case)
    move = compact_move_to_tuple(request.move)
    game = build_game_from_compact(case)
    legal_moves = game.get_legal_moves()
    if move not in legal_moves:
        raise HTTPException(status_code=400, detail="Move is not legal.")
    game.play_turn(*move)
    next_case = dict(case)
    next_case["payload"] = game_to_compact_payload(game)
    return {
        "case": next_case,
        "summary": game_summary(game),
    }


@app.post("/api/template")
def create_template(request: TemplateRequest):
    case = make_template(request.id, parse_move(request.best_move), players=request.players)
    name = safe_name(f"{request.id}.json")
    path = CASE_DIR / name
    CASE_DIR.mkdir(parents=True, exist_ok=True)
    write_json(path, case)
    return {"name": name, "case": case, "validation": validate_compact_case(case)}


@app.post("/api/from-log")
def create_from_log(request: FromLogRequest):
    source = log_path(request.log)
    best_move = parse_move(request.best_move) if request.best_move else None
    case = case_from_log(source, request.line, request.id, best_move=best_move)
    name = safe_name(f"{request.id}.json")
    path = CASE_DIR / name
    CASE_DIR.mkdir(parents=True, exist_ok=True)
    write_json(path, case)
    return {"name": name, "case": case, "validation": validate_compact_case(case)}


@app.get("/api/logs")
def list_logs():
    if not LOG_DIR.exists():
        return []
    return [
        {
            "name": path.name,
            "mtime": path.stat().st_mtime,
            "size": path.stat().st_size,
        }
        for path in sorted(LOG_DIR.glob("unity_raw_*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    ]


@app.get("/api/logs/{name}/lines")
def list_log_lines(name: str):
    path = log_path(name)
    rows = []
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        row = json.loads(line)
        response = json.loads(row["response_raw"])
        data = json.loads(row["request_raw"])
        game = AzulGame.from_table_data(TableData(**data))
        rows.append(
            {
                "line": idx,
                "response": response,
                "scores": [player.score for player in game.players],
                "center": game.public_board.center,
                "factories": game.public_board.factories,
            }
        )
    return rows


@app.post("/api/export")
def export_selected(request: ExportRequest):
    paths = [case_path(name) for name in request.cases]
    missing = [path.name for path in paths if not path.exists()]
    if missing:
        raise HTTPException(status_code=404, detail=f"Missing cases: {missing}")
    export_cases(paths, ROOT / request.output, by_episode=request.by_episode, best_prob=request.best_prob)
    manifest = (ROOT / request.output).with_suffix(".jsonl")
    return {
        "output": request.output,
        "manifest": manifest.relative_to(ROOT).as_posix(),
        "count": len(paths),
    }


@app.post("/api/export-episodes")
def export_selected_episodes(request: ExportEpisodesRequest):
    paths = [episode_path(name) for name in request.episodes]
    missing = [path.name for path in paths if not path.exists()]
    if missing:
        raise HTTPException(status_code=404, detail=f"Missing episodes: {missing}")
    result = export_episodes(paths, ROOT / request.output, best_prob=request.best_prob)
    return {
        "output": result["output"].relative_to(ROOT).as_posix(),
        "manifest": result["manifest"].relative_to(ROOT).as_posix(),
        "episodes": result["episodes"],
        "samples": result["samples"],
    }


@app.get("/api/exported/{path:path}")
def read_export_manifest(path: str):
    target = ROOT / path
    if target.suffix != ".jsonl" or not target.exists():
        raise HTTPException(status_code=404, detail="Manifest not found.")
    return [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
