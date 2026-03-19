import random


class RandomAgent:
    def decide(self, game):
        legal_moves = game.get_legal_moves()
        # 随机抓一个动作，这就是 AI 的“思考”
        return random.choice(legal_moves)