import gymnasium as gym
import pygame
import numpy as np
import sys

class KeyboardControlAgent:
    def __init__(self):
        # 初始化pygame
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("Car Racing - Keyboard Control")
        self.clock = pygame.time.Clock()
        
        # 动作映射
        self.action = np.array([0.0, 0.0, 0.0])  # [转向, 加速, 刹车]
        
    def get_action(self):
        """获取键盘输入并转换为动作"""
        # 重置动作
        self.action = np.array([0.0, 0.0, 0.0])
        
        # 处理事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return None
        
        # 获取按键状态
        keys = pygame.key.get_pressed()
        
        # 转向控制 (A/左箭头 左转, D/右箭头 右转)
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            self.action[0] = -1.0  # 左转
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            self.action[0] = 1.0   # 右转
        
        # 加速控制 (W/上箭头 加速)
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            self.action[1] = 1.0   # 加速
        # 刹车控制 (S/下箭头 刹车)
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
            self.action[2] = 0.8   # 刹车
        
        return self.action
    
    def render_info(self, step, reward, total_reward):
        """渲染游戏信息"""
        font = pygame.font.Font(None, 36)
        
        # 显示控制说明
        control_text = [
            "Controls:",
            "W/↑ - Accelerate",
            "S/↓ - Brake",
            "A/← - Turn Left",
            "D/→ - Turn Right",
            "ESC - Quit"
        ]
        
        # 显示状态信息
        info_text = [
            f"Step: {step}",
            f"Reward: {reward:.2f}",
            f"Total Reward: {total_reward:.2f}",
            f"Steering: {self.action[0]:.2f}",
            f"Throttle: {self.action[1]:.2f}",
            f"Brake: {self.action[2]:.2f}"
        ]
        
        # 渲染文本
        y_offset = 10
        for text in control_text:
            text_surface = font.render(text, True, (255, 255, 255))
            self.screen.blit(text_surface, (10, y_offset))
            y_offset += 30
            
        y_offset += 20
        for text in info_text:
            text_surface = font.render(text, True, (255, 255, 0))
            self.screen.blit(text_surface, (10, y_offset))
            y_offset += 30
    
    def close(self):
        """关闭pygame"""
        pygame.quit()

def run_keyboard_control(episodes=5, max_steps=3000):
    """
    运行键盘控制的小车
    
    参数:
    - episodes: 回合数
    - max_steps: 每回合最大步数
    """
    
    # 创建键盘控制器
    controller = KeyboardControlAgent()
    
    # 创建环境（使用rgb_array模式，因为我们自己渲染）
    env = gym.make("CarRacing-v3", render_mode="rgb_array")
    
    try:
        for episode in range(episodes):
            print(f"Starting Episode {episode + 1}")
            
            # 重置环境
            observation, info = env.reset()
            total_reward = 0
            step_count = 0
            
            # 渲染初始画面
            frame = env.render()
            frame_surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
            controller.screen.blit(frame_surface, (0, 0))
            pygame.display.flip()
            
            for step in range(max_steps):
                # 获取键盘动作
                action = controller.get_action()
                
                # 检查是否退出
                if action is None:
                    print("Exiting...")
                    return
                
                # 执行动作
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                step_count += 1
                
                # 渲染环境
                frame = env.render()
                if frame is not None:
                    # 转换frame格式以适应pygame
                    frame_surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
                    controller.screen.blit(frame_surface, (0, 0))
                
                # 渲染控制信息
                controller.render_info(step_count, reward, total_reward)
                
                # 更新显示
                pygame.display.flip()
                controller.clock.tick(30)  # 30 FPS
                
                # 检查回合结束
                if terminated or truncated:
                    print(f"Episode {episode + 1} finished - Steps: {step_count}, Total Reward: {total_reward:.2f}")
                    break
            
            print(f"Episode {episode + 1} completed - Total Reward: {total_reward:.2f}")
            
            # 回合间暂停
            if episode < episodes - 1:
                print("Press any key to continue to next episode...")
                waiting = True
                while waiting:
                    for event in pygame.event.get():
                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_ESCAPE:
                                waiting = False
                                return
                            else:
                                waiting = False
                        elif event.type == pygame.QUIT:
                            waiting = False
                            return
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        # 清理资源
        env.close()
        controller.close()
        print("Environment closed")

# 简化版本（如果上面的版本有问题）
def run_simple_keyboard_control():
    """简化版键盘控制"""
    
    # 初始化pygame
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Car Racing - Simple Control")
    clock = pygame.time.Clock()
    
    # 创建环境
    env = gym.make("CarRacing-v3", render_mode="rgb_array")
    observation, info = env.reset()
    
    print("Controls:")
    print("W/↑ - Accelerate")
    print("S/↓ - Brake") 
    print("A/← - Turn Left")
    print("D/→ - Turn Right")
    print("ESC - Quit")
    print("开始控制小车！")
    
    running = True
    total_reward = 0
    
    try:
        while running:
            # 处理事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
            
            # 默认动作
            action = np.array([0.0, 0.0, 0.0])
            
            # 获取按键状态
            keys = pygame.key.get_pressed()
            
            # 转向
            if keys[pygame.K_a] or keys[pygame.K_LEFT]:
                action[0] = -0.5  # 左转
            elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
                action[0] = 0.5   # 右转
            
            # 加速/刹车
            if keys[pygame.K_w] or keys[pygame.K_UP]:
                action[1] = 0.8   # 加速
            elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
                action[2] = 0.8   # 刹车
            
            # 执行动作
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # 渲染
            frame = env.render()
            if frame is not None:
                frame_surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
                screen.blit(frame_surface, (0, 0))
            
            # 显示信息
            font = pygame.font.Font(None, 36)
            info_text = [
                f"Reward: {reward:.2f}",
                f"Total: {total_reward:.2f}",
                "ESC to quit"
            ]
            
            for i, text in enumerate(info_text):
                text_surface = font.render(text, True, (255, 255, 0))
                screen.blit(text_surface, (10, 10 + i * 30))
            
            pygame.display.flip()
            clock.tick(30)
            
            # 检查结束条件
            if terminated:
                print("Episode finished! Resetting...")
                observation, info = env.reset()
                total_reward = 0
    
    finally:
        env.close()
        pygame.quit()
        print(f"Final total reward: {total_reward:.2f}")

if __name__ == "__main__":
    print("选择控制模式:")
    print("1. 完整版键盘控制（带详细信息显示）")
    print("2. 简化版键盘控制")
    
    try:
        choice = input("请输入选择 (1 或 2): ").strip()
        
        if choice == "1":
            print("启动完整版键盘控制...")
            run_keyboard_control(episodes=3, max_steps=5000)
        elif choice == "2":
            print("启动简化版键盘控制...")
            run_simple_keyboard_control()
        else:
            print("无效选择，使用简化版...")
            run_simple_keyboard_control()
            
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"发生错误: {e}")