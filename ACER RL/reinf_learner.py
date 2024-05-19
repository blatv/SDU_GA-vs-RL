import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make("LunarLander-v2")
env = DummyVecEnv([lambda: env])
model = A2C('MlpPolicy', env, verbose = 1)

model.learn(total_timesteps=20000,progress_bar=True)
env = gym.make("LunarLander-v2",render_mode="human")
results = evaluate_policy(model, env, n_eval_episodes=3, render=True)
env.close()
print(results)

