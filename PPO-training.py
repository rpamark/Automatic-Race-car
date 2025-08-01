import gymnasium as gym
import torch
import numpy as np
import os

# 导入必要的包装器
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
# 导入用于解决警告的模块
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
# 导入用于检查环境类型的函数
from stable_baselines3.common.env_util import is_wrapped

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

def make_eval_env():
    """创建并返回一个用于评估的环境，包装以解决警告"""
    # 1. 创建基础环境
    env = gym.make("CarRacing-v3", render_mode="rgb_array") # 评估时通常不渲染
    # 2. 用 Monitor 包装以记录奖励和长度
    env = Monitor(env)
    # 3. 包装成 DummyVecEnv (这是 EvalCallback 默认做的)
    env = DummyVecEnv([lambda: env])
    return env

def train_agent_simple(n_timesteps=5000):
    """使用 PPO 训练智能体"""

    # 1. 创建训练环境 (PPO 会自动处理 VecTransposeImage)
    #    为了与评估环境类型更匹配，也可以显式包装
    train_env = gym.make("CarRacing-v3", render_mode="rgb_array")
    # 如果你想显式地确保训练环境也被 VecTransposeImage 包装（虽然 PPO 会做），
    # 可以使用 make_vec_env，但这会创建向量化环境。
    # 为了简单起见，我们保持原样，让 PPO 处理。

    # 2. 定义 PPO 模型
    model = PPO(
        "CnnPolicy",
        train_env, # 使用未被 VecTransposeImage 显式包装的 env
        verbose=1,
        device=device,
    )

    # 3. 创建一个经过适当包装的评估环境以解决警告
    eval_env = make_eval_env()

    # 4. 定义回调函数：保存最佳模型
    #    verbose=1 或 2 会在评估时打印平均奖励等信息
    eval_callback = EvalCallback(
        eval_env, # 使用我们创建的已包装评估环境
        best_model_save_path="./", # 保存到当前目录
        eval_freq=max(int(n_timesteps / 100), 5000), # 评估频率
        n_eval_episodes=3,
        deterministic=True,
        render=False,
        verbose=1 # 设置为1或2以输出评估信息，包括平均奖励
    )

    print("开始训练 PPO 智能体 for CarRacing...")
    print(f"训练设备: {device}")
    print(f"总训练步数: {n_timesteps}")

    try:
        model.learn(
            total_timesteps=n_timesteps,
            callback=eval_callback,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n训练被用户中断。")

    # 保存最终模型
    model.save("ppo_carracing_final_model")
    print("\n最终模型已保存到 ppo_carracing_final_model.zip")

    # 关闭环境
    train_env.close()
    eval_env.close()

    return model, train_env # 或者返回 None 作为 env，因为我们已经关闭了它

# 测试函数保持不变
def test_agent_simple(model_path, n_episodes=3):
    """测试训练好的智能体"""
    # 加载模型时需要指定 env=None 或提供一个兼容的环境
    # PPO.load 通常能处理环境差异，但显式指定 device 更好
    model = PPO.load(model_path, device=device, env=None)
    # 创建用于测试的环境，开启渲染
    test_env = gym.make("CarRacing-v3", render_mode="human")

    scores = []
    print(f"\n开始测试智能体 ({n_episodes} 个回合)...")
    for i_episode in range(n_episodes):
        # 注意：gymnasium 的 reset 返回 obs, info
        obs, _ = test_env.reset()
        score = 0
        # 注意：gymnasium 使用 terminated 和 truncated
        terminated, truncated = False, False
        step_count = 0
        while not (terminated or truncated) and step_count < 1000:
            # 使用模型预测动作 (确定性策略)
            action, _states = model.predict(obs, deterministic=True)
            # 注意：gymnasium 的 step 返回 obs, reward, terminated, truncated, info
            obs, reward, terminated, truncated, info = test_env.step(action)
            # done = terminated or truncated # 通常不需要，直接用 terminated/truncated
            score += reward
            step_count += 1
            # 渲染由环境的 render_mode="human" 处理
            test_env.render()

        scores.append(score)
        print(f'测试回合 {i_episode + 1}: 得分 = {score:.2f}, 步数 = {step_count}')

    avg_score = np.mean(scores)
    print(f'\n测试平均得分: {avg_score:.2f}')
    test_env.close()
    return scores

def main():
    """主函数"""
    print("=" * 40)
    print("PPO 训练 for CarRacing-v3")
    print("=" * 40)
    print(f"硬件配置: {device}")
    print("- 使用 stable-baselines3 PPO")
    print("- 解决环境类型不匹配警告")
    print("- 评估时输出平均奖励")
    print("-" * 40)

    # n_timesteps = 500000 # 原始较长的训练时间
    n_timesteps = 50000   # 为了快速测试

    # 训练阶段
    trained_model, _ = train_agent_simple(n_timesteps=n_timesteps)

    print("\n" + "=" * 40)
    print("训练完成！")
    print("=" * 40)

    # 测试阶段: 测试最佳模型
    best_model_path = "./best_model.zip"
    if os.path.exists(best_model_path):
        print(f"测试最佳模型: {best_model_path}")
        test_scores = test_agent_simple(best_model_path, n_episodes=3)
        print(f"\n最佳模型测试平均得分: {np.mean(test_scores):.2f}")
    else:
        print("未找到最佳模型检查点。")

    '''# 也可以测试最终模型
    final_model_path = "ppo_carracing_final_model.zip"
    if os.path.exists(final_model_path):
        print(f"\n测试最终模型: {final_model_path}")
        test_scores_final = test_agent_simple(final_model_path, n_episodes=3)
        print(f"\n最终模型测试平均得分: {np.mean(test_scores_final):.2f}")'''

if __name__ == "__main__":
    main()