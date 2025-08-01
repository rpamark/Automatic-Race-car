import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import cv2
import os

# --- 1. 定义与训练时相同的网络结构和参数 ---
# 确保这些与你训练时的 DQNAgent 和 DQN 类完全一致

# 动作空间 (必须与训练时完全相同)
# 注意：这是之前为 CarRacing-v3 定义的离散动作集
DISCRETE_ACTIONS = [
    np.array([0, 0, 0], dtype=np.float32),      # 0: 无操作
    np.array([-1, 0, 0], dtype=np.float32),   # 1: 轻微左转
    np.array([1.0, 0, 0], dtype=np.float32),    # 2: 轻微右转
    np.array([0, 0.15, 0], dtype=np.float32),    # 3: 中等加速
    np.array([0, 0.5, 0], dtype=np.float32),    # 4: 全力加速
    np.array([0, 0, 0.5], dtype=np.float32),    # 5: 中等刹车
    np.array([0, 0, 1.0], dtype=np.float32),    # 6: 全力刹车
]
ACTION_DIM = len(DISCRETE_ACTIONS)
FRAME_STACK_SIZE = 4 # 假设训练时使用了4帧堆叠

# 检查GPU可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device for inference: {device}")

# --- 2. 定义与训练时相同的CNN Q-Network ---
class DQN(nn.Module):
    def __init__(self, action_size, seed=42):
        super(DQN, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        # 卷积层 (输入: 4 channels, 84x84) - 使用4帧堆叠
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4) # 输出: (32, 20, 20)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2) # 输出: (64, 9, 9)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1) # 输出: (64, 7, 7)
        
        # 计算卷积层输出的扁平化尺寸: 64 * 7 * 7 = 3136
        conv_out_size = 64 * 7 * 7
        
        # 全连接层 
        self.fc1 = nn.Linear(conv_out_size, 512) 
        self.fc2 = nn.Linear(512, action_size)

    def forward(self, x):
        # x shape: (batch, 4, 84, 84)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # Flatten
        x = x.view(x.size(0), -1) # (batch, 3136)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --- 3. 定义一个简化版的Agent用于推理 ---
class InferenceAgent:
    def __init__(self, action_size, model_path):
        self.action_size = action_size
        self.frame_stack_size = FRAME_STACK_SIZE
        self.stacked_frames = deque(maxlen=self.frame_stack_size)
        
        # 初始化网络
        self.qnetwork = DQN(action_size).to(device)
        
        # 加载模型权重
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        checkpoint = torch.load(model_path, map_location=device)
        self.qnetwork.load_state_dict(checkpoint['model_state_dict'])
        # 如果保存时包含了 epsilon 等信息，也可以加载
        # self.epsilon = checkpoint.get('epsilon', 0.0) # 推理时通常不使用epsilon
        
        self.qnetwork.eval() # 设置为评估模式
        print(f"Inference agent loaded model from {model_path}")

    def _stack_frames(self, frame):
        """堆叠帧以提供运动信息"""
        # 初始化堆叠帧
        if len(self.stacked_frames) == 0:
            for _ in range(self.frame_stack_size):
                self.stacked_frames.append(frame)
        else:
            self.stacked_frames.append(frame)
        # 返回堆叠后的帧 (4, 84, 84)
        return np.stack(self.stacked_frames, axis=0)

    def reset_episode(self):
        """在每个episode开始时重置帧堆叠"""
        self.stacked_frames.clear()

    def act(self, state):
        """根据当前状态选择动作 (确定性策略，不探索)"""
        # state shape: (4, 84, 84) - numpy array
        # 转换为 (1, 4, 84, 84) - torch tensor
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
        
        with torch.no_grad(): # 禁用梯度计算以提高效率
            action_values = self.qnetwork(state_tensor)
        
        # 选择具有最高Q值的动作
        action_idx = np.argmax(action_values.cpu().data.numpy())
        return action_idx

# --- 4. 图像预处理函数 (必须与训练时完全相同) ---
def preprocess_frame(frame):
    """优化的图像预处理，平衡信息量和计算效率"""
    if frame is None:
        return np.zeros((84, 84), dtype=np.float32) # 返回单帧零矩阵

    # 转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # 调整大小到 84x84 
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    # 归一化到 [0, 1]
    normalized = resized.astype(np.float32) / 255.0
    # 返回单帧 (84, 84)
    return normalized # shape: (84, 84)

# --- 5. 主函数：加载模型并运行智能体 ---
def run_trained_agent(model_path='best_checkpoint.pth', episodes=5, render_mode="human"):
    """
    加载训练好的模型并运行智能体
    
    参数:
    - model_path: 模型文件路径
    - episodes: 运行的回合数
    - render_mode: 渲染模式 ("human" 显示, "rgb_array" 不显示)
    """
    try:
        # 1. 创建推理Agent
        print("Initializing inference agent...")
        agent = InferenceAgent(ACTION_DIM, model_path)
        
        # 2. 创建环境
        print("Creating CarRacing environment...")
        # 使用 human 模式可以看到可视化效果
        env = gym.make("CarRacing-v3", render_mode=render_mode) 
        
        scores = []
        print(f"\n开始使用训练好的智能体驾驶小车 ({episodes} 个回合)...")
        print("-" * 50)
        
        for i_episode in range(episodes):
            # 重置环境和agent的帧堆叠
            raw_state, _ = env.reset()
            agent.reset_episode() 
            
            # 预处理初始帧并堆叠
            processed_frame = preprocess_frame(raw_state)
            state = agent._stack_frames(processed_frame) 
            
            score = 0
            done = False
            step_count = 0
            
            print(f"开始测试回合 {i_episode + 1}...")
            
            while not done and step_count < 1000: # 可以设置最大步数
                # Agent 根据当前状态选择动作
                action_idx = agent.act(state) 
                # 将离散动作索引转换为环境需要的连续动作
                action = DISCRETE_ACTIONS[action_idx]
                
                # 执行动作
                raw_next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # 预处理下一帧并更新堆叠
                processed_next_frame = preprocess_frame(raw_next_state)
                next_state = agent._stack_frames(processed_next_frame) 
                
                # 更新状态和分数
                state = next_state
                score += reward
                step_count += 1
                
                # 渲染环境 (如果模式是 "human")
                if render_mode == "human":
                    env.render()
                    
            scores.append(score)
            print(f'测试回合 {i_episode + 1}: 得分 = {score:.2f}, 步数 = {step_count}')
            
        # 6. 打印最终统计信息
        if scores:
            avg_score = np.mean(scores)
            std_score = np.std(scores)
            print(f'\n=== 最终测试结果 ===')
            print(f'平均得分: {avg_score:.2f} ± {std_score:.2f}')
            print(f'最高得分: {np.max(scores):.2f}')
            print(f'最低得分: {np.min(scores):.2f}')
        
        env.close()
        print("\n智能体驾驶演示完成。")
        return scores

    except Exception as e:
        print(f"运行智能体时发生错误: {e}")
        raise # 重新抛出异常以便调试

if __name__ == "__main__":
    # 指定模型文件路径（确保它在当前工作目录下）
    MODEL_FILE = 'best_checkpoint.pth'
    
    if not os.path.exists(MODEL_FILE):
        print(f"错误: 找不到模型文件 '{MODEL_FILE}'。请确保它在当前目录下。")
        print("当前目录包含的文件:")
        for f in os.listdir('.'):
            print(f"  - {f}")
    else:
        print(f"找到模型文件: {MODEL_FILE}")
        # 运行智能体，使用 human 模式进行可视化
        # 如果只想看结果不看过程，可以将 render_mode 改为 "rgb_array"
        run_trained_agent(model_path=MODEL_FILE, episodes=3, render_mode="human")
