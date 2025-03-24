########################################################################
# File: slime-neat.py
########################################################################

import numpy as np
np.bool8 = bool  # Fix for the "no attribute bool8" bug in Gym checker

import gym
import slimevolleygym
import neat
import os
from viz import BestGenomeSaver, create_evolution_gif, plot_fitness

# For parallel evaluation (optional):
from neat.parallel import ParallelEvaluator
import multiprocessing

# ACTION_MAPPING from discrete (6) to MultiBinary(3)
ACTION_MAPPING = {
    0: [0, 0, 0],    # do nothing
    1: [-1, 0, 0],   # move left
    2: [1, 0, 0],    # move right
    3: [0, 1, 0],    # jump
    4: [-1, 1, 0],   # jump + left
    5: [1, 1, 0]     # jump + right
}

def make_env():
    """
    Creates the SlimeVolley environment with the built-in baseline policy
    controlling the right side, and we control the left side (NEAT).
    """
    env = gym.make("SlimeVolley-v0")
    return env

def normalize_obs(obs):
    """
    Normalizes SlimeVolleyGym's 12D observation.
    Positions typically in [0,1], velocities ~[-5,5].
    """
    obs_norm = obs.copy()
    position_indices = [0, 1, 4, 5, 8, 9]
    velocity_indices = [2, 3, 6, 7, 10, 11]

    obs_norm[position_indices] /= 1.0  # Already 0..1 in normal usage
    obs_norm[velocity_indices] /= 5.0  # Scale velocities to ~[-1,1]
    return obs_norm

def eval_genome(genome, config):
    """
    Evaluates a single genome with partial step-based reward,
    but at a smaller value to avoid overshadowing +/- 1 scoring.
    """
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    n_episodes = 10  # Evaluate over more episodes -> more stable fitness
    total_fitness = 0.0

    for _ in range(n_episodes):
        env = make_env()
        obs = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            obs_norm = normalize_obs(obs)
            action_values = net.activate(obs_norm)
            discrete_action = int(np.argmax(action_values))
            action = ACTION_MAPPING[discrete_action]

            # Step in environment
            obs, reward, done, info = env.step(action)

            # Standard SlimeVolley reward is +1 if opponent loses a life, -1 if we lose a life
            # We'll add a very small step-based reward, e.g., +0.001 per time-step
            ep_reward += reward
            ep_reward += 0.001  # Very small partial credit for each step

        env.close()
        total_fitness += ep_reward

    # Average the fitness across episodes
    genome.fitness = total_fitness / n_episodes
    return genome.fitness

def test_winner(genome, config, n_episodes=10):
    """
    Tests the best genome over a specified number of episodes using the default env scoring.
    """
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

    # Evaluate single-genome in parallel
    num_workers = multiprocessing.cpu_count()
    pe = ParallelEvaluator(num_workers, eval_genome)

    # Try e.g. 300-500 generations
    winner = pop.run(pe.evaluate, 3000)

    print("\nBest genome:\n", winner)
    create_evolution_gif()
    plot_fitness(best_genome_saver.best_fitness_over_time)

    # Thorough test on 20 episodes
    test_winner(winner, config, n_episodes=20)

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-slime.ini")
    run_neat(config_path)
