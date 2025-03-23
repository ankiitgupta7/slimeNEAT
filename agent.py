import numpy as np
np.bool8 = bool    # <<< ADD THIS

import gym
import slimevolleygym

def evaluate_baseline(n_episodes: int = 100):
    env = gym.make("SlimeVolley-v0")
    total_score = 0.0

    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            ep_reward += reward
        total_score += ep_reward

    avg_random_vs_baseline = total_score / n_episodes
    print(f"Baseline avg score vs random: {-avg_random_vs_baseline:.3f}")

if __name__ == "__main__":
    evaluate_baseline(100)
