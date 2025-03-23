########################################################################
# File: agent.py
########################################################################

import numpy as np
np.bool8 = bool  # temporary fix for the "no attribute bool8" bug in Gym checker

import gym
import slimevolleygym
import neat
import os
from viz import BestGenomeSaver, create_evolution_gif, plot_fitness


# Mapping from discrete action index (0–5) to a 3-element action vector.
ACTION_MAPPING = {
    0: [0, 0, 0],    # do nothing
    1: [-1, 0, 0],   # move left
    2: [1, 0, 0],    # move right
    3: [0, 1, 0],    # jump
    4: [-1, 1, 0],   # jump left
    5: [1, 1, 0]     # jump right
}

def make_env():
    """
    Creates the SlimeVolley environment with the built-in baseline policy
    controlling the 'right' side, and we control the 'left' side (NEAT).
    """
    env = gym.make("SlimeVolley-v0")
    return env

def normalize_obs(obs):
    obs_norm = obs.copy()
    position_indices = [0, 1, 4, 5, 8, 9]
    velocity_indices = [2, 3, 6, 7, 10, 11]

    obs_norm[position_indices] /= 1.0   # already 0 to 1 range
    obs_norm[velocity_indices] /= 5.0   # velocities scaled to [-1,1] approximately

    return obs_norm


def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    n_episodes = 5
    total_reward = 0.0
    for _ in range(n_episodes):
        env = make_env()
        obs = env.reset()
        done = False
        ep_reward = 0.0
        steps_survived = 0
        while not done:
            obs_norm = normalize_obs(obs)
            action_values = net.activate(obs_norm)
            discrete_action = int(np.argmax(action_values))
            action = ACTION_MAPPING[discrete_action]
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            steps_survived += 1
        total_reward += ep_reward + (steps_survived / 3000.0)
        env.close()
    genome.fitness = total_reward / n_episodes
    return genome.fitness


from neat.parallel import ParallelEvaluator
import multiprocessing

def run_neat(config_file):
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file
    )

    pop = neat.Population(config)

    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    best_genome_saver = BestGenomeSaver(config)
    pop.add_reporter(best_genome_saver)

    # Use the correct single-genome evaluation function with ParallelEvaluator
    num_workers = multiprocessing.cpu_count()
    pe = ParallelEvaluator(num_workers, eval_genome)

    winner = pop.run(pe.evaluate, 300)  # <-- Corrected call (CRUCIAL FIX)

    print("\nBest genome:\n", winner)

    create_evolution_gif()
    plot_fitness(best_genome_saver.best_fitness_over_time)

    test_winner(winner, config, n_episodes=10)


def test_winner(genome, config, n_episodes=10):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    total_reward = 0.0

    for i in range(n_episodes):
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

        print(f"Episode {i} reward: {ep_reward}")
        total_reward += ep_reward
        env.close()

    avg_reward = total_reward / n_episodes
    print(f"Average reward over {n_episodes} test episodes: {avg_reward}")

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-slime.ini")
    run_neat(config_path)
