import gymnasium as gym
import torch
import numpy as np
import os

# --- stable-baselines3 imports ---
from stable_baselines3 import PPO

def test_final_model(model_path, n_episodes=5, render_mode="human"):
    """
    测试训练好的最终 PPO 模型。

    Args:
        model_path (str): 训练好的模型文件路径 (例如 'ppo_carracing_final_model.zip')。
        n_episodes (int): 测试运行的回合数。
        render_mode (str): 环境渲染模式 ('human' for 显示, 'rgb_array' for 不显示)。
    """
    # --- 1. 检查模型文件是否存在 ---
    if not os.path.exists(model_path):
        print(f"错误: 找不到模型文件 '{model_path}'")
        return

    # --- 2. 配置设备 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备进行测试: {device}")

    # --- 3. 加载训练好的模型 ---
    try:
        # env=None 表示在加载时不绑定特定环境，PPO.load 通常能处理兼容性
        model = PPO.load(model_path, device=device, env=None)
        print(f"成功加载模型: {model_path}")
    except Exception as e:
        print(f"加载模型时出错: {e}")
        return

    # --- 4. 创建测试环境 ---
    # 使用 'human' 模式以便可视化测试过程
    try:
        test_env = gym.make("CarRacing-v3", render_mode=render_mode)
        print(f"成功创建测试环境: CarRacing-v3 (render_mode='{render_mode}')")
    except Exception as e:
        print(f"创建测试环境时出错: {e}")
        return

    # --- 5. 运行测试回合 ---
    scores = []
    print(f"\n开始测试智能体 ({n_episodes} 个回合)...")
    print("-" * 40)

    for i_episode in range(n_episodes):
        # 重置环境，获取初始观测
        obs, _ = test_env.reset()
        score = 0
        step_count = 0
        done = False

        # 单个回合循环
        while not done and step_count < 1000: # 限制最大步数
            # 使用模型预测动作 (通常测试时使用确定性策略)
            action, _states = model.predict(obs, deterministic=True)
            
            # 执行动作，获取结果
            obs, reward, terminated, truncated, info = test_env.step(action)
            done = terminated or truncated

            # 累积奖励和步数
            score += reward
            step_count += 1

            # 如果是 human 模式，这会显示画面
            if render_mode == "human":
                test_env.render()

        # 回合结束，记录得分
        scores.append(score)
        print(f'回合 {i_episode + 1:2d}: 得分 = {score:8.2f}, 步数 = {step_count:4d}')

    # --- 6. 输出测试结果统计 ---
    if scores:
        avg_score = np.mean(scores)
        std_score = np.std(scores)
        min_score = np.min(scores)
        max_score = np.max(scores)

        print("-" * 40)
        print("测试结果摘要:")
        print(f"  平均得分: {avg_score:8.2f} ± {std_score:6.2f}")
        print(f"  最高得分: {max_score:8.2f}")
        print(f"  最低得分: {min_score:8.2f}")
        print("-" * 40)
    else:
        print("测试未完成，无得分记录。")

    # --- 7. 清理 ---
    test_env.close()
    print("测试环境已关闭。")


# --- 主执行入口 ---
if __name__ == "__main__":
    # 指定要测试的最终模型文件路径
    final_model_file = "ppo_best_model.zip"

    # 运行测试
    test_final_model(model_path=final_model_file, n_episodes=5, render_mode="human")
