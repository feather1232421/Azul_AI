from config import REVERSE_LOOKUP
from curated_cases import END_ROUND_FIRST_PLAYER_CASE, build_game_from_case
from explore_mtcs import MCTSAgent
from model_utils import load_model
import torch


def evaluate_forced_line(first_move):
    game = build_game_from_case(END_ROUND_FIRST_PLAYER_CASE)
    game.play_turn(*first_move)
    legal = game.get_legal_moves()
    assert len(legal) == 1, f"Expected forced reply, got {legal}"
    game.play_turn(*legal[0])
    return [player.look_score() for player in game.players]


def main():
    good_move = ("center", 2, 5)
    bad_move = ("center", 4, 5)

    good_scores = evaluate_forced_line(good_move)
    bad_scores = evaluate_forced_line(bad_move)

    print("Forced-line scores:")
    print(f" - {good_move}: {good_scores}")
    print(f" - {bad_move}: {bad_scores}")

    assert good_scores[0] > good_scores[1], "Expected good move to leave player 0 ahead."
    assert bad_scores[0] < bad_scores[1], "Expected bad move to leave player 0 behind."

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, _, resolved_model_type = load_model(
        "models/transformer_champion.pt",
        device=device,
        obs_dim=567,
        action_dim=180,
    )
    print(f"Loaded champion as {resolved_model_type} on {device}")

    game = build_game_from_case(END_ROUND_FIRST_PLAYER_CASE)
    agent = MCTSAgent(
        n_simulations=100,
        n_determinizations=1,
        my_player_idx=0,
        net=net,
        device=device,
        action_dim=180,
        puct_c=1.0,
        prior_temperature=1.0,
        root_dirichlet_alpha=0.0,
        root_exploration_fraction=0.0,
    )
    move, pi, _ = agent.decide_with_info(game)
    good_prob = float(pi[REVERSE_LOOKUP[good_move]])
    bad_prob = float(pi[REVERSE_LOOKUP[bad_move]])

    print("Search choice:")
    print(f" - chosen move: {move}")
    print(f" - good move prob: {good_prob:.4f}")
    print(f" - bad move prob: {bad_prob:.4f}")

    assert move == good_move, f"Expected {good_move}, got {move}"
    assert good_prob > bad_prob, "Expected the good move to receive higher visit probability."
    print("Regression test passed.")


if __name__ == "__main__":
    main()
