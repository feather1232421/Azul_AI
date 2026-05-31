import time
from logic import AzulGame


def benchmark_legal_moves(game, n=1000):
    t0 = time.perf_counter()
    total = 0
    for _ in range(n):
        moves = game.get_legal_moves()
        total += len(moves)
    t1 = time.perf_counter()
    print("legal_moves avg ms:", (t1 - t0) / n * 1000)
    print("avg num moves:", total / n)


def benchmark_clone(game, n=500):
    t0 = time.perf_counter()
    for _ in range(n):
        g = game.clone()
    t1 = time.perf_counter()
    print("clone avg ms:", (t1 - t0) / n * 1000)


def benchmark_search_unit(game, n=200):
    t0 = time.perf_counter()
    for _ in range(n):
        g = game.clone()
        moves = g.get_legal_moves()
        if moves:
            g.play_turn(*moves[0])
            _ = g.get_legal_moves()
    t1 = time.perf_counter()
    print("search unit avg ms:", (t1 - t0) / n * 1000)


if __name__ == "__main__":
    game = AzulGame()
    benchmark_clone(game)
    benchmark_search_unit(game)
    benchmark_legal_moves(game)
