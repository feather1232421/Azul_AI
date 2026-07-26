from logic import AzulGame
from reconstruction_test import TableData
from pydantic import BaseModel
import socket
import struct
import json
import gc
import traceback
import os
import sys
from datetime import datetime
from pathlib import Path
from explore_mtcs import MCTSAgent
from ai import GreedyAgent
import torch
from config import ACTION_DIM, PLAYER_FACTORY_MAP, TRANSFORMER_OBS_DIM
from model_utils import load_model


class PlayerActionData(BaseModel):
    ClientId: int
    SeatId: int
    FactoryId: int
    ColorType: int
    Row: int


def resource_dir() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def model_path(filename: str) -> str:
    return os.path.join(resource_dir(), filename)

# =========================
# 1. 网络基础
# =========================


def recv_all(sock, n):
    data = b""
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data


def recv_message(sock):
    header = recv_all(sock, 4)
    if header is None:
        return None

    msg_len = struct.unpack("<I", header)[0]  # 👈 小端

    body = recv_all(sock, msg_len)
    if body is None:
        return None

    return body.decode("utf-8")


def send_message(sock, msg: str):
    data = msg.encode("utf-8")
    header = struct.pack("<I", len(data))  # 👈 小端
    sock.sendall(header + data)


def append_raw_log(raw_log_path, raw_msg: str, reply: str, error: str = None):
    if raw_log_path is None:
        return

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "request_raw": raw_msg,
        "response_raw": reply,
    }
    if error is not None:
        row["error"] = error

    raw_log_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# =========================
# 2. move → Unity Action
# =========================

def convert_move_to_action(move, client_id, seat_id):
    """
    move: (source, color, destination)
    """

    source, color, destination = move

    # source 处理
    if source == "center":
        source_id = -1
    else:
        source_id = int(source)

    # destination 处理
    if destination == 5:
        destination_id = -1
    else:
        destination_id = int(destination)

    return PlayerActionData(
        ClientId=int(client_id),
        SeatId=int(seat_id),
        FactoryId=source_id,
        ColorType=int(color),
        Row=destination_id,
    )


def validate_online_table_data(table_data):
    if table_data.totalPlayerCount is None:
        raise ValueError("totalPlayerCount is required")
    if table_data.me.seatId is None:
        raise ValueError("me.seatId is required")
    if table_data.me.clientId is None:
        raise ValueError("me.clientId is required")

    player_count = table_data.totalPlayerCount
    if player_count not in PLAYER_FACTORY_MAP:
        raise ValueError(f"totalPlayerCount must be 2, 3, or 4, got {player_count}")
    if len(table_data.opponents) != player_count - 1:
        raise ValueError(
            "totalPlayerCount does not match me + opponents: "
            f"{player_count} != {1 + len(table_data.opponents)}"
        )

    expected_factories = PLAYER_FACTORY_MAP[player_count]
    if len(table_data.factories) != expected_factories:
        raise ValueError(
            f"{player_count}P requires {expected_factories} factories, "
            f"got {len(table_data.factories)}"
        )
    for factory_idx, factory in enumerate(table_data.factories):
        if len(factory) != 4:
            raise ValueError(
                f"factory {factory_idx} must contain 4 areas, got {len(factory)}"
            )

    players = [table_data.me, *table_data.opponents]
    seat_ids = [player.seatId for player in players]
    if any(seat_id is None for seat_id in seat_ids):
        raise ValueError("every player must include seatId")
    if len(set(seat_ids)) != len(seat_ids):
        raise ValueError(f"seatId values must be unique, got {seat_ids}")


def action_to_json(action):
    if hasattr(action, "model_dump"):
        payload = action.model_dump()
    else:
        payload = action.dict()
    return json.dumps(payload, ensure_ascii=False)


# =========================
# 3. 选择动作
# =========================

def choose_move(game, client_id, seat_id, agent=None):
    if agent is None:
        agent = GreedyAgent()
    legal_moves = game.get_legal_moves()

    if not legal_moves:
        raise ValueError("no legal moves")

    move = agent.decide(game)
    # move = legal_moves[0]  # 👈 先用最简单策略

    print("chosen move:", move)

    action = convert_move_to_action(move, client_id=client_id, seat_id=seat_id)

    print("converted action:", action)

    return action


# =========================
# 4. 处理 Unity 消息
# =========================

def handle_obs_message(raw_msg: str, agent=None, raw_log_path=None) -> str:

    request_client_id = -1
    request_seat_id = -1
    try:
        data = json.loads(raw_msg)
        me_data = data.get("me") if isinstance(data, dict) else None
        if isinstance(me_data, dict):
            request_client_id = me_data.get("clientId", -1)
            request_seat_id = me_data.get("seatId", -1)

        table_data = TableData(**data)
        validate_online_table_data(table_data)
        game = AzulGame.from_table_data(table_data)
        game.display_all_info()
        action = choose_move(
            game,
            client_id=table_data.me.clientId,
            seat_id=table_data.me.seatId,
            agent=agent,
        )
        reply = action_to_json(action)
        append_raw_log(raw_log_path, raw_msg, reply)
        return reply

    except Exception as e:
        traceback.print_exc()

        # 出错也要返回一个合法结构，防止 Unity 崩
        reply = action_to_json(PlayerActionData(
            ClientId=request_client_id,
            SeatId=request_seat_id,
            FactoryId=-1,
            ColorType=0,
            Row=-1,
        ))
        append_raw_log(raw_log_path, raw_msg, reply, error=repr(e))
        return reply


# =========================
# 5. 主循环
# =========================

def run_server(host="127.0.0.1", port=9999, agent=None, raw_log_dir=None):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((host, port))
    server.listen(1)
    raw_log_path = None
    if raw_log_dir:
        raw_log_dir = Path(raw_log_dir)
        raw_log_path = raw_log_dir / f"unity_raw_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        print(f"[Python] Raw logs -> {raw_log_path}")

    print(f"[Python] Server listening on {host}:{port}")

    try:
        conn, addr = server.accept()
        print(f"[Python] Client connected from {addr}")

        with conn:
            while True:
                raw_msg = recv_message(conn)
                if raw_msg is None:
                    print("[Python] Client disconnected")
                    break

                print("\n[Python] Received obs (前200字符):")
                print(raw_msg[:200])

                reply = handle_obs_message(raw_msg, agent, raw_log_path=raw_log_path)

                print("[Python] Sending action:")
                print(reply)

                send_message(conn, reply)
                raw_msg = None
                reply = None
                gc.collect()

    finally:
        server.close()
        print("[Python] Server closed")


# =========================
# 6. 启动
# =========================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/transformer_action300_legacy2p_distill_last.pt")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--n-simulations", type=int, default=1000)
    parser.add_argument("--n-determinizations", type=int, default=4)
    parser.add_argument("--puct-c", type=float, default=1.0)
    parser.add_argument("--prior-temperature", type=float, default=1.0)
    parser.add_argument("--raw-log-dir", type=str, default=None)
    parser.add_argument("--no-policy", action="store_true")
    parser.add_argument("--no-value", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, _, resolved_model_type, _ = load_model(
        model_path(args.model),
        device=device,
        obs_dim=TRANSFORMER_OBS_DIM,
        action_dim=ACTION_DIM,
        allow_partial_load=True,
    )
    print(f"[Python] Loaded {args.model} as {resolved_model_type}")
    model_0 = MCTSAgent(
        n_simulations=args.n_simulations,
        n_determinizations=args.n_determinizations,
        my_player_idx=0,
        net=net,
        device=device,
        use_policy=not args.no_policy,
        use_value=not args.no_value,
        puct_c=args.puct_c,
        prior_temperature=args.prior_temperature,
    )
    run_server(host=args.host, port=args.port, agent=model_0, raw_log_dir=args.raw_log_dir)
