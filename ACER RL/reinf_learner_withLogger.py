import gymnasium as gym
import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import time

# Custom Logger.
from tensorboardX import SummaryWriter
log_dir = "tensorboard_logs/A2C"  
global_summary_writer = SummaryWriter(log_dir)
training_start = time.time()


# Create environment and wrap it with DummyVecEnv
env = gym.make("LunarLander-v2")
env = Monitor(env)
env = DummyVecEnv([lambda: env])

# Create model with A2C policy and enable TensorBoard callback
model = A2C('MlpPolicy', env, verbose=1,tensorboard_log="tensorboard_logs/A2C")

class TensorboardCallback(BaseCallback):
    reward = 0
    step = 0
    episode_rewards = []
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        self.step = 1+self.step
        keysList = list(self.locals.keys())
        
        if ("done" in keysList):
            if self.locals["done"] == False:
                self.reward = self.reward + self.locals["rewards"]
            else:
                self.episode_rewards.append(self.reward)
                global_summary_writer.add_scalar("A2C/Reward/time",self.reward,(time.time()-training_start))
                global_summary_writer.add_scalar("A2C/Reward-Mean-ofall/time",np.mean(self.episode_rewards),(time.time()-training_start))
                self.reward = 0
        return True
    
    

# Train the model with TensorBoard logging
model.learn(total_timesteps=5500000, progress_bar=True,callback=TensorboardCallback())

# Evaluate the trained model (optional)
env = gym.make("LunarLander-v2", render_mode="human")
results = evaluate_policy(model, env, n_eval_episodes=3, render=True)
env.close()
print(results)
