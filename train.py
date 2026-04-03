import os
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import get_schedule_fn
from environment import RewardDebugCallback
from sb3_contrib import MaskablePPO
from environment import AzulEnv  # 导入你分家后的环境
from ai import GreedyAgent, RandomAgent,PPOAgent  # 导入对手


def train():
    # 1. 组建陪练团
    opponents_pool = [RandomAgent]
    env = AzulEnv(opponents=opponents_pool)

    # 2. “热启动”：加载 V6 覆盖参数
    # 强制覆盖学习率和采样步数，让它在 V6 基础上精修
    from sb3_contrib import MaskablePPO

    model = MaskablePPO.load(
        "azul_ppo_v8.zip",
        env=env,
        learning_rate=5e-5,
        n_steps=4096,
        n_epochs=10,
        batch_size=256,
        ent_coef=0.01,
        gamma=0.995,
        gae_lambda=0.97,
        clip_range=0.2,
        verbose=1,
        tensorboard_log="./azul_logs/",
    )
    # # A. 加载模型
    # model = MaskablePPO.load("azul_ppo_v7.zip", env=env)
    #
    # # B. 覆盖简单参数
    # model.ent_coef = 0.005  # 建议调小，进入收敛
    # model.n_epochs = 10
    # model.batch_size = 256
    #
    # # C. 🌟 覆盖学习率 (必须进优化器)
    # new_lr = 2e-5
    # for param_group in model.policy.optimizer.param_groups:
    #     param_group['lr'] = new_lr

    # D. 🌟 覆盖 n_steps (必须重置 Buffer)
    model.n_steps = 4096
    model.rollout_buffer.buffer_size = 4096
    model.rollout_buffer.reset()  # 强制清空旧数据

    print("🚀 V8 引擎参数调优完成，准备起飞！")
    # 3. 开始进化
    model.learn(total_timesteps=200_000, progress_bar=True, callback=RewardDebugCallback())

    # 4. 保存成果
    model.save("azul_ppo_v8")


def train_old():
    # 0. 🌟 关键的一步：先创建环境实例
    env = AzulEnv()
    # 1. 定义存档路径
    SAVE_PATH = "./models/"
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    # 2. 检查是否有旧模型想接着练
    MODEL_FILE = "azul_ppo_v6.zip"
    if os.path.exists(MODEL_FILE):
        print(f"📖 发现旧模型 {MODEL_FILE}，正在读取并续练...")
        model = MaskablePPO.load(MODEL_FILE, env=env)
    else:
        print("✨ 未发现模型，正在初始化全新大脑...")
        policy_kwargs = dict(
            # net_arch 定义了网络结构
            # pi: 策略网络（管动作），vf: 价值网络（管估分）
            net_arch=dict(pi=[256, 256], vf=[256, 256])
        )
        model = MaskablePPO(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,  # 🌟 注入更强的大脑
            verbose=1,
            tensorboard_log="./azul_logs/"
        )
    # 3. 设置定时自动存档（每 50000 步存一次，防止停电或崩溃）
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=SAVE_PATH,
        name_prefix="azul_step"
    )
    model.ent_coef = 0.05
    # 🌟 核心：手动重置学习率调度器

    model.lr_schedule = get_schedule_fn(5e-5)  #
    # 4. 开始大练兵
    print("🚀 启动长线训练...")
    model.learn(
        total_timesteps=200000,
        callback=checkpoint_callback,
        reset_num_timesteps=False  # 🌟 关键：保持 TensorBoard 曲线连续
    )
    # 5. 最后保存一个终版
    model.save("azul_ppo_v8")
    print("✅ 训练圆满结束！")


def train_new():
    # ppo = PPOAgent("azul_ppo_v8")
    opponents_pool = [GreedyAgent()]
    # opponents_pool = [ppo]
    env = AzulEnv(opponents=opponents_pool)

    policy_kwargs = dict(
        # net_arch 定义了网络结构
        # pi: 策略网络（管动作），vf: 价值网络（管估分）
        net_arch=dict(pi=[512, 256], vf=[512, 256])
    )

    model = MaskablePPO.load(
        "Azul_new_vector_v1",
        env=env,
        learning_rate=7e-5,
        n_steps=4096,
        n_epochs=10,
        batch_size=256,
        ent_coef=0.01,
        gamma=0.98,
        gae_lambda=0.97,
        clip_range=0.2,
        verbose=1,
        tensorboard_log="./azul_logs/",
    )
    # model = MaskablePPO(
    #     "MlpPolicy",
    #     env=env,
    #     policy_kwargs=policy_kwargs,
    #     learning_rate=1e-4,
    #     n_steps=4096,
    #     batch_size=256,
    #     n_epochs=10,
    #     gamma=0.98,
    #     gae_lambda=0.97,
    #     ent_coef=0.01,
    #     clip_range=0.2,
    #     verbose=1,
    #     tensorboard_log="./azul_logs/",
    # )
    model.learn(total_timesteps=200_000,
                progress_bar=True,
                callback=RewardDebugCallback(),
                reset_num_timesteps=False,
                tb_log_name="PPO_new_vector")
    # 4. 保存成果
    model.save("Azul_new_vector_v1")


if __name__ == "__main__":
    train_new()

