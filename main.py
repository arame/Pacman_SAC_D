import gym
import numpy as np
from sac_torch import Agent
from utils import plot_learning_curve
import numpy as np
from config import Hyper, Constants
from atari_image import make_env
import os
CUDA_LAUNCH_BLOCKING=1

def main():
    Hyper.init()
    env = make_env(Constants.env_id)    # See wrapper code for environment in atari_image.py
    Hyper.n_actions = env.action_space.n
    shape = (env.observation_space.shape)
    agent = Agent(input_dims=shape, env=env, n_actions=env.action_space.n)
    filename = f"{Constants.env_id}_games{Hyper.n_games}_alpha{Hyper.alpha}.png"
    figure_file = f'plots/{filename}'

    best_ave_score = env.reward_range[0]
    best_score = 0
    score_history = []
    load_checkpoint = False
    if load_checkpoint:
        agent.load_models()
        env.render(mode='human')
    total_steps = 0
    for i in range(Hyper.n_games):
        observation = env.reset()
        done = False
        steps = 0
        score = 0
        while not done:
            # Sample action from the policy
            action = agent.choose_action(observation) 

            # Sample transition from the environment  
            new_observation, reward, done, info = env.step(action)
            steps += 1
            total_steps += 1

            # Store transition in the replay buffer
            agent.remember(observation, action, reward, new_observation, done)
            if not load_checkpoint:
                agent.learn()
            score += reward
            observation = new_observation
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        if score > best_score:
            best_score = score

        if avg_score > best_ave_score:
            best_ave_score = avg_score
            if not load_checkpoint:
                agent.save_models()

        episode = i + 1
        print(f"episode {episode}: score {score}, best_score {best_score}, best ave score {best_ave_score}, trailing 100 games avg {avg_score}, steps {steps}, total steps {total_steps}")

    print(f"total number of steps taken: {total_steps}")
    if not load_checkpoint:
        x = [i+1 for i in range(Hyper.n_games)]
        plot_learning_curve(x, score_history, figure_file)

if __name__ == '__main__':
    main()
