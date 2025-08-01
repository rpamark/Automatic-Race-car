import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast
import os
import cv2 # 用于图像预处理

# --- 硬件配置适配 ---
# 检查GPU可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# --- 1. 动作空间离散化 (保持不变或微调) ---
# 定义离散动作集 (简化动作集以减少复杂度)
DISCRETE_ACTIONS = [
    [0, 0, 0],      # 0: 无操作
    [-1, 0, 0],   # 1: 左转
    [1, 0, 0],    # 2: 右转
    [0, 0.15, 0],    # 3: 中等加速
    [0, 0.3, 0],    # 4: 全力加速
    [0, 0, 0.5],    # 5: 中等刹车
    [0, 0, 1.0],    # 6: 全力刹车
    # [-0.5, 0.3, 0], # 7: 左转 + 轻微加速 (可选，增加复杂度)
    # [0.5, 0.3, 0],  # 8: 右转 + 轻微加速 (可选)
]
ACTION_DIM = len(DISCRETE_ACTIONS)

# --- 2. 图像预处理 (优化) ---
def preprocess_frame(frame):
    """优化的图像预处理，平衡信息量和计算效率"""
    if frame is None:
        return np.zeros((4, 84, 84), dtype=np.float32) # 返回4帧零矩阵

    # 转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # 调整大小到 84x84 (比之前稍小，减少计算量)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    # 归一化到 [0, 1]
    normalized = resized.astype(np.float32) / 255.0
    # 注意：这里返回单帧 (84, 84)，堆叠将在Agent中处理
    return normalized # shape: (84, 84)

# --- 3. CNN Q-Network (优化结构) ---
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
        
        # 全连接层 (减少神经元数量以节省显存)
        self.fc1 = nn.Linear(conv_out_size, 512) 
        self.fc2 = nn.Linear(512, action_size)
        
        # 可选：添加Dropout层防止过拟合 (在资源有限时谨慎使用)
        # self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        # x shape: (batch, 4, 84, 84)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # Flatten
        x = x.view(x.size(0), -1) # (batch, 3136)
        x = F.relu(self.fc1(x))
        # x = self.dropout(x) # 可选
        x = self.fc2(x)
        return x

# --- 4. 经验回放 (调整缓冲区大小) ---
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        # 根据显存调整 buffer_size，8GB显存建议 50000-100000
        self.memory = deque(maxlen=buffer_size) 
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        # state/next_state shape: (4, 84, 84) - numpy arrays
        states = torch.from_numpy(np.stack([e[0] for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.stack([e[3] for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)

# --- 5. DQN Agent (核心调整) ---
class DQNAgent:
    def __init__(self, action_size, seed=42):
        self.action_size = action_size
        self.seed = random.seed(seed)
        
        # --- 显存和性能优化参数 ---
        self.frame_stack_size = 4 # 堆叠4帧以提供运动信息
        self.buffer_size = 200000  # 减少经验回放缓冲区大小以节省显存
        self.batch_size = 64      # 使用较小的batch size (32是常用且对8GB显存友好的值)
        self.gamma = 0.99
        self.tau = 1e-3
        self.lr = 1e-4            # 稍微降低学习率可能有助于稳定
        self.update_every = 4
        self.epsilon = 1.0
        self.epsilon_min = 0.08
        self.epsilon_decay = 0.999 # 稍微减慢epsilon衰减，增加探索时间
        
        # Q-Network
        self.qnetwork_local = DQN(action_size).to(device)
        self.qnetwork_target = DQN(action_size).to(device)
        # 使用较小的学习率
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr, eps=1e-4) 
        self.scaler = GradScaler()

        # 经验回放缓冲区
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)
        self.t_step = 0
        
        # 用于帧堆叠的缓存
        self.stacked_frames = deque(maxlen=self.frame_stack_size)

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

    def step(self, state, action, reward, next_state, done):
        # 保存经验
        self.memory.add(state, action, reward, next_state, done)
        # 学习
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)

    def act(self, state, training=True):
        # state shape: (4, 84, 84) - numpy array
        # 转换为 (1, 4, 84, 84) - torch tensor
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state_tensor)
        self.qnetwork_local.train()
        
        # ε-贪婪策略
        if training:
            if random.random() > self.epsilon:
                return np.argmax(action_values.cpu().data.numpy())
            else:
                return random.choice(np.arange(self.action_size))
        else:
            return np.argmax(action_values.cpu().data.numpy())

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
        
        # 奖励裁剪以稳定训练
        rewards = torch.clamp(rewards, -100.0, 100.0)
        
        with autocast():#使用贝尔曼公式获取目标Q值进行梯度下降
            # 获取下一个状态的最大预测Q值
            Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
            # 计算目标Q值
            Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
            # 获取当前Q值
            Q_expected = self.qnetwork_local(states).gather(1, actions)
            # 计算损失
            loss = F.mse_loss(Q_expected, Q_targets)
            
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        # 梯度裁剪，防止梯度爆炸，对稳定训练很重要
        torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), max_norm=10)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # 软更新目标网络
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

        # 更新epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


# --- 6. 训练函数 (调整训练参数) ---
def train_agent(n_episodes=1500, max_t=1000, solve_score=900.0, initial_agent=None):
    # 使用 rgb_array 以便我们能获取帧进行预处理
    # 可以考虑关闭渲染以加快训练速度
    env = gym.make("CarRacing-v3", render_mode="rgb_array") 
    action_size = ACTION_DIM
    if initial_agent is not None:
        agent = initial_agent
        print(f"使用加载的 agent 开始训练，初始 epsilon: {agent.epsilon:.4f}")
        # 可以选择重置 epsilon 或保持加载时的值
        # 例如，如果你想让它继续衰减，可以保持；如果你想重新探索，可以重置
        # 这里我们保持加载的 epsilon 值
        # agent.epsilon = 1.0 # 如果需要重置 epsilon，取消注释此行
    else:
        agent = DQNAgent(action_size=action_size)
        print("创建新的 agent 开始训练...")
    
    scores = []
    scores_window = deque(maxlen=10)
    best_total_reward = -np.inf  # 初始化最佳平均分数
    print("开始训练DQN智能体 for CarRacing...")
    print(f"动作空间大小: {action_size}")
    print(f"训练设备: {device}")
    print(f"帧堆叠数: {agent.frame_stack_size}")
    print(f"经验回放缓冲区大小: {agent.buffer_size}")
    print(f"Batch Size: {agent.batch_size}")
    print("-" * 50)
    
    for i_episode in range(1, n_episodes+1):
        raw_state, _ = env.reset()
        agent.reset_episode() # 重置帧堆叠
        processed_frame = preprocess_frame(raw_state)
        state = agent._stack_frames(processed_frame) # 初始化堆叠帧
        
        score = 0
        total_reward = 0 # 用于跟踪原始奖励
        
        for t in range(max_t):
            action_idx = agent.act(state)
            # 将离散动作索引转换为环境需要的连续动作
            action = DISCRETE_ACTIONS[action_idx]
            
            raw_next_state, reward, terminated, truncated, _ = env.step(np.array(action, dtype=np.float32))
            done = terminated or truncated
            
            # 累积原始奖励用于显示
            total_reward += reward
            
            # --- 奖励塑形 (重要优化) ---
            # 原始奖励可能不利于学习，这里进行一些调整
            # 1. 鼓励前进：如果奖励为正，稍微放大
            if reward > 0:
                reward = reward * 2.0
            if reward < 0:
                reward = -0.5
            # 2. 惩罚无效操作：如果长时间没有进展，可以给小的负奖励
            # (这个逻辑比较复杂，这里简化处理)
            
            processed_next_frame = preprocess_frame(raw_next_state)
            next_state = agent._stack_frames(processed_next_frame)
            
            agent.step(state, action_idx, reward, next_state, done)
            state = next_state
            score += reward # 这里用调整后的奖励计算score
            
            if done:
                break
                
        scores_window.append(score)
        scores.append(score)
        current_average_score = np.mean(scores_window)
        if total_reward > best_total_reward:
            best_total_reward = total_reward
            save_model(agent, 'best_checkpoint.pth')
            print(f'\nNew best model saved with Total Reward: {best_total_reward:.2f}')
        # 打印进度，包括原始奖励
        print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}\tLast Score: {score:.2f}\tLast Total Reward: {total_reward:.2f}\tEpsilon: {agent.epsilon:.3f}', end="")
        
        # 定期保存检查点
        if i_episode % 200 == 0 and i_episode > 0:
            #save_model(agent, f'carracing_dqn_checkpoint_{i_episode}.pth')
            #print(f'\nCheckpoint saved at episode {i_episode}')
            pass
            
        if i_episode % 50 == 0:
            print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}')
            
        # 检查是否达到解决标准 
        if np.mean(scores_window) >= solve_score:
            print(f'\nEnvironment solved in {i_episode:d} episodes!\tAverage Score: {np.mean(scores_window):.2f}')
            save_model(agent, 'carracing_dqn_model_solved.pth')
            print(f"模型已保存到 'carracing_dqn_model_solved.pth'")
            break
            
    # 如果训练完成但未达到解决标准，也保存模型
    if np.mean(scores_window) < solve_score and i_episode == n_episodes:
        save_model(agent, 'carracing_dqn_model_final.pth')
        print(f"\n训练完成，模型已保存到 'carracing_dqn_model_final.pth'")
        
    env.close()
    return scores, agent


# --- 7. 模型保存/加载 (基本保持不变) ---
def save_model(agent, filepath):
    """保存训练好的模型"""
    torch.save({
        'model_state_dict': agent.qnetwork_local.state_dict(),
        'target_model_state_dict': agent.qnetwork_target.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'epsilon': agent.epsilon,
        'action_size': agent.action_size,
        # 保存帧堆叠大小等超参数
        'frame_stack_size': agent.frame_stack_size,
    }, filepath)
    print(f"模型参数已保存到 {filepath}")

def load_model(filepath, action_size):
    """加载训练好的模型"""
    # 注意：这里需要根据保存的参数创建agent
    checkpoint = torch.load(filepath, map_location=device)
    frame_stack_size = checkpoint.get('frame_stack_size', 4)
    
    # 创建agent并设置帧堆叠大小
    agent = DQNAgent(action_size=action_size)
    # 如果frame_stack_size不同，可能需要特殊处理或警告
    
    agent.qnetwork_local.load_state_dict(checkpoint['model_state_dict'])
    agent.qnetwork_target.load_state_dict(checkpoint['target_model_state_dict'])
    agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    agent.epsilon = checkpoint['epsilon']
    agent.qnetwork_local.eval()
    print(f"模型已从 {filepath} 加载")
    return agent

# --- 8. 测试函数 (保持不变) ---
def test_agent(agent, n_episodes=3, render_mode="human"):
    """测试训练好的智能体"""
    env = gym.make("CarRacing-v3", render_mode=render_mode)
    scores = []
    print(f"\n开始测试智能体 ({n_episodes} 个回合)...")
    print("-" * 30)
    
    for i_episode in range(n_episodes):
        raw_state, _ = env.reset()
        agent.reset_episode() # 重置帧堆叠
        processed_frame = preprocess_frame(raw_state)
        state = agent._stack_frames(processed_frame)
        
        score = 0
        done = False
        step_count = 0
        
        while not done and step_count < 1000: # 限制测试步数
            action_idx = agent.act(state, training=False) # 测试模式
            action = DISCRETE_ACTIONS[action_idx]
            
            raw_next_state, reward, terminated, truncated, _ = env.step(np.array(action, dtype=np.float32))
            done = terminated or truncated
            
            processed_next_frame = preprocess_frame(raw_next_state)
            next_state = agent._stack_frames(processed_next_frame)
            
            state = next_state
            score += reward
            step_count += 1
            
            if render_mode == "human":
                env.render()
                
        scores.append(score)
        print(f'测试回合 {i_episode + 1}: 得分 = {score:.2f}, 步数 = {step_count}')
        
    avg_score = np.mean(scores)
    std_score = np.std(scores)
    print(f'\n测试结果:')
    print(f'平均得分: {avg_score:.2f} ± {std_score:.2f}')
    print(f'最高得分: {np.max(scores):.2f}')
    print(f'最低得分: {np.min(scores):.2f}')
    env.close()
    return scores

# --- 9. 绘制结果 (基本保持不变) ---
def plot_training_results(scores):
    """绘制训练结果"""
    if not scores:
        print("没有分数数据可绘制。")
        return

    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(scores, alpha=0.7)
    plt.title('score_curve')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    window_size = min(100, len(scores))
    if len(scores) >= window_size:
        moving_avg = [np.mean(scores[i-window_size:i]) for i in range(window_size, len(scores)+1)]
        plt.plot(moving_avg, 'r-', linewidth=2)
        plt.title(f'ave_score_movmean (window_size={window_size})')
        plt.xlabel('Episode')
        plt.ylabel('Average Score')
        plt.grid(True)
    else:
        plt.text(0.5, 0.5, '数据不足\n无法计算滑动平均', 
                horizontalalignment='center', verticalalignment='center',
                transform=plt.gca().transAxes)
        plt.title('ave_score_movmean')
        
    plt.subplot(1, 3, 3)
    plt.hist(scores, bins=30, alpha=0.7, color='green')
    plt.title('score_distribution')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('carracing_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("训练结果图表已保存为 'carracing_training_results.png'")


# --- 10. 主函数 (调整训练回合数) ---
def main():
    """主函数：完整的训练、保存、测试流程"""
    print("=" * 60)
    print("第1阶段：训练DQN智能体 for CarRacing-v3")
    print("=" * 60)
    print(f"硬件配置: RTX 4090 D 24GB")
    print("优化策略:")
    print("- 使用4帧堆叠提供运动信息")
    print("- 减小网络和batch size以适应显存")
    print("- 调整经验回放缓冲区大小")
    print("- 实施奖励塑形和梯度裁剪")
    print("- 定期保存检查点")
    print("-" * 60)
    
     # --- 新增：尝试从检查点加载模型 ---
    CHECKPOINT_PATH = 'best_checkpoint.pth'
    env_for_init = gym.make("CarRacing-v3", render_mode="rgb_array")
    action_size_for_init = ACTION_DIM
    env_for_init.close()
    
    if os.path.exists(CHECKPOINT_PATH):
        print(f"正在从检查点 '{CHECKPOINT_PATH}' 加载模型...")
        try:
            # 使用 load_model 函数加载
            loaded_agent = load_model(CHECKPOINT_PATH, action_size_for_init)
            print("模型加载成功!")
            initial_epsilon = loaded_agent.epsilon
            # 我们将使用加载的 agent 开始训练
            # 注意：train_agent 内部会创建自己的 agent 实例，所以我们需要传递加载的 agent 的状态
            # 更简单的方法是在 train_agent 内部支持接收一个预训练的 agent
            # 但为了最小化修改，我们在 train_agent 内部处理检查点加载
            # 这里我们只需通知 train_agent 从检查点开始
            # 通过修改 train_agent 函数签名和内部逻辑来实现
        except Exception as e:
            print(f"加载检查点 '{CHECKPOINT_PATH}' 时出错: {e}")
            print("将从头开始训练...")
            loaded_agent = None
            initial_epsilon = None
    else:
        print(f"检查点文件 '{CHECKPOINT_PATH}' 未找到，将从头开始训练...")
        loaded_agent = None
        initial_epsilon = None
    # 1. 训练阶段 
    # 对于8GB显存，1500回合是一个合理的起点。根据你的观察可以调整。
    scores, trained_agent = train_agent(n_episodes=1500, max_t=1000, solve_score=900.0) 
    
    print("\n" + "=" * 60)
    print("第2阶段：绘制训练结果")
    print("=" * 60)
    plot_training_results(scores)
    
    print("\n" + "=" * 60)
    print("第3阶段：测试训练好的智能体")
    print("=" * 60)
    
    # 2. 测试阶段
    # 尝试加载已保存的最佳模型进行测试
    model_paths = ['carracing_dqn_model_solved.pth', 'carracing_dqn_model_final.pth']
    agent_to_test = None
    
    for path in model_paths:
        if os.path.exists(path):
            print(f"加载模型 {path} 进行测试...")
            agent_to_test = load_model(path, ACTION_DIM)
            break
            
    if agent_to_test is None:
        print("未找到已保存的模型，使用训练结束时的agent进行测试...")
        agent_to_test = trained_agent
        
    # 使用 human 模式进行测试渲染
    test_scores = test_agent(agent_to_test, n_episodes=3, render_mode="human")
    
    print("\n" + "=" * 60)
    print("训练和测试完成！")
    print("=" * 60)
    if scores:
        print(f"最终训练平均得分 (最后100回合): {np.mean(scores[-100:]):.2f}")
    print(f"测试平均得分: {np.mean(test_scores):.2f}")
    print(f"保存的模型文件: {[p for p in model_paths if os.path.exists(p)]}")
    print(f"训练结果图表: carracing_training_results.png")


if __name__ == "__main__":
    main()