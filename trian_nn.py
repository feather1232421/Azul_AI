import torch
import torch.nn as nn
from sb3_contrib import MaskablePPO
from train_bc import DistillModel  # 导入你之前的 BC 模型类
from environment import AzulEnv  # 你的环境
from sb3_contrib.common.maskable.utils import get_action_masks
from ai import GreedyAgent

def inject_weights(ppo_model_path, bc_weights_path, env):
    # 1. 加载 BC 权重字典
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bc_state_dict = torch.load(bc_weights_path, map_location=device)

    # 2. 创建一个全新的 PPO 模型 (或者加载现有的)
    # 注意：这里的 lr 要设得很低，防止初始化即爆炸
    # 定义 PPO 的架构，使其与 BC 模型完全对齐
    policy_kwargs = dict(
        net_arch=dict(
            pi=[256, 256],  # 策略网络 (Actor)
            vf=[256, 256]  # 价值网络 (Critic)
        )
    )

    # 创建模型时传入 policy_kwargs
    model = MaskablePPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=2e-5
    )

    # 3. 开始“手术”：手动拷贝权重
    # SB3 的 MlpPolicy 结构通常是：
    # policy.mlp_extractor.policy_net (隐藏层)
    # policy.action_net (输出层)

    with torch.no_grad():
        # 拷贝第一层 (Linear 256)
        model.policy.mlp_extractor.policy_net[0].weight.copy_(bc_state_dict["net.0.weight"])
        model.policy.mlp_extractor.policy_net[0].bias.copy_(bc_state_dict["net.0.bias"])

        # 拷贝第二层 (Linear 256)
        model.policy.mlp_extractor.policy_net[2].weight.copy_(bc_state_dict["net.2.weight"])
        model.policy.mlp_extractor.policy_net[2].bias.copy_(bc_state_dict["net.2.bias"])

        # 拷贝输出层 (Action Head)
        # 注意：BC 的最后一层是 net.4，对应 PPO 的 action_net
        model.policy.action_net.weight.copy_(bc_state_dict["net.4.weight"])
        model.policy.action_net.bias.copy_(bc_state_dict["net.4.bias"])

    with torch.no_grad():
        # 顺便把 Value Net 的特征提取层也初始化了，这比随机初始化要好得多
        model.policy.mlp_extractor.value_net[0].weight.copy_(bc_state_dict["net.0.weight"])
        model.policy.mlp_extractor.value_net[0].bias.copy_(bc_state_dict["net.0.bias"])

        model.policy.mlp_extractor.value_net[2].weight.copy_(bc_state_dict["net.2.weight"])
        model.policy.mlp_extractor.value_net[2].bias.copy_(bc_state_dict["net.2.bias"])

    print("✅ 权重灌入完成！PPO 现在已经继承了搜索代理的直觉。")
    return model


if __name__ == "__main__":
    # 1. 初始化环境和模型
    opponents = [GreedyAgent()]
    env = AzulEnv(opponents=opponents)
    model = inject_weights("v9_init.zip", "bc_distill_best.pt", env)

    # 2. 设定微调参数
    # 关键：我们不需要太大的熵（entropy），因为 BC 已经给了我们很好的确定性
    model.ent_coef = 0.001
    model.learning_rate = 1e-5 # 极低的学习率，像雕刻一样微调

    # 3. 开启训练
    print("开始 V9 阶段微调训练...")

    model.learn(
        total_timesteps=500000,
        progress_bar=True,
        tb_log_name="PPO_V9_FineTune"
    )

    # 4. 保存最终成果
    model.save("Azul_V9_Final")