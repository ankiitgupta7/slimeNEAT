import pickle
import neat
import os
import gym
import slimevolleygym
import numpy as np
np.bool8 = bool  # Fix for the "no attribute bool8" bug in Gym checker

import matplotlib.pyplot as plt
import json

# Use the same action mapping as before
ACTION_MAPPING = {
    0: [0, 0, 0],    # do nothing
    1: [-1, 0, 0],   # move left
    2: [1, 0, 0],    # move right
    3: [0, 1, 0],    # jump
    4: [-1, 1, 0],   # jump left
    5: [1, 1, 0]     # jump right
}

def make_env():
    """Creates the SlimeVolley environment."""
    env = gym.make("SlimeVolley-v0")
    return env

def normalize_obs(obs):
    """
    Normalizes the 12-dimensional observation.
    Positions (indices 0,1,4,5,8,9) assumed in [0,1];
    velocities (indices 2,3,6,7,10,11) scaled by 5 to approximately get [-1,1].
    """
    obs_norm = obs.copy()
    position_indices = [0, 1, 4, 5, 8, 9]
    velocity_indices = [2, 3, 6, 7, 10, 11]
    obs_norm[position_indices] /= 1.0
    obs_norm[velocity_indices] /= 5.0
    return obs_norm

def test_genome(genome, config, n_episodes):
    """
    Runs the best genome on the SlimeVolley environment for n_episodes,
    returns the average reward and a list of per-episode rewards.
    """
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    total_reward = 0.0
    episode_rewards = []
    for ep in range(n_episodes):
        env = make_env()
        obs = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            obs_norm = normalize_obs(obs)
            action_values = net.activate(obs_norm)
            discrete_action = int(np.argmax(action_values))
            action = ACTION_MAPPING[discrete_action]
            obs, reward, done, info = env.step(action)
            ep_reward += reward
        episode_rewards.append(ep_reward)
        total_reward += ep_reward
        env.close()
    avg_reward = total_reward / n_episodes
    return avg_reward, episode_rewards

def main():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-slime.ini")
    
    # Load NEAT config
    config = neat.Config(neat.DefaultGenome,
                         neat.DefaultReproduction,
                         neat.DefaultSpeciesSet,
                         neat.DefaultStagnation,
                         config_path)
    
    # Load the saved best genome
    with open("best_genome.pkl", "rb") as f:
        best_genome = pickle.load(f)
    
    # Define the sets of episodes to test
    episode_sets = [5, 10, 20, 50, 100, 200, 500, 1000]
    results = {}
    
    for n in episode_sets:
        avg, rewards = test_genome(best_genome, config, n)
        results[n] = {"average_reward": avg, "episode_rewards": rewards}
        print(f"Test over {n} episodes: Average reward = {avg}")
    
    # Save detailed results to a JSON file
    with open("test_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("Saved test results to test_results.json")
    
    # Plot average reward vs. number of episodes
    x = sorted(results.keys())
    y = [results[k]["average_reward"] for k in x]
    plt.figure(figsize=(8,6))
    plt.plot(x, y, marker='o', linestyle='-')
    plt.xlabel("Number of Episodes")
    plt.ylabel("Average Reward")
    plt.title("Best Genome Performance vs. Number of Test Episodes")
    plot_path = os.path.join(local_dir, "best_genome_performance.png")
    plt.savefig(plot_path)
    plt.show()
    print(f"Saved performance plot to {plot_path}")

if __name__ == "__main__":
    main()
