import torch
import torch.nn as nn
from sb3_contrib import MaskablePPO
from environment import RewardDebugCallback
from abandon_teach import BCPolicy


def build_ppo_from_bc(env, bc_path="bc_greedy_policy_best.pt"):
    # 1. 先加载 BC checkpoint
    checkpoint = torch.load(bc_path, map_location="cpu")

    bc_model = BCPolicy(
        obs_dim=checkpoint["obs_dim"],
        action_dim=checkpoint["action_dim"],
    )
    bc_model.load_state_dict(checkpoint["model_state_dict"])
    bc_model.eval()

    # 2. 新建一个 PPO，注意 policy 结构要和 BC 对齐
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        activation_fn=nn.ReLU,
    )

    model = MaskablePPO(
        "MlpPolicy",
        env=env,
        learning_rate=5e-5,
        n_steps=4096,
        batch_size=256,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.97,
        ent_coef=0.01,
        clip_range=0.2,
        verbose=1,
        tensorboard_log="./azul_logs/",
        policy_kwargs=policy_kwargs,
    )

    # 3. 把 BC 权重拷进 PPO 的 actor 分支
    with torch.no_grad():
        pi_net = model.policy.mlp_extractor.policy_net
        action_net = model.policy.action_net

        # BC: net[0], net[2], net[4]
        # PPO actor: pi_net[0], pi_net[2], action_net
        assert isinstance(pi_net, nn.Sequential)
        assert len(pi_net) >= 3

        # 第一层
        pi_net[0].weight.copy_(bc_model.net[0].weight)
        pi_net[0].bias.copy_(bc_model.net[0].bias)

        # 第二层
        pi_net[2].weight.copy_(bc_model.net[2].weight)
        pi_net[2].bias.copy_(bc_model.net[2].bias)

        # 输出层
        action_net.weight.copy_(bc_model.net[4].weight)
        action_net.bias.copy_(bc_model.net[4].bias)

    return model


def train_bc_initialized_ppo():
    from environment import AzulEnv
    from ai import GreedyAgent  # 改成你的真实导入

    opponents_pool = [GreedyAgent()]
    env = AzulEnv(opponents=opponents_pool)

    model = build_ppo_from_bc(env, bc_path="bc_greedy_policy_best.pt")

    # 先保存一个“刚灌完 BC 的 PPO”
    model.save("azul_ppo_bc_init")

    # 再继续 PPO 微调
    model.learn(
        total_timesteps=200_000,
        progress_bar=True,
        callback=RewardDebugCallback(),
    )

    model.save("azul_ppo_bc_finetuned")


if __name__ == "__main__":
    train_bc_initialized_ppo()
