import gym
import numpy as np
from sac_torch import Agent
from utils import plot_learning_curve
import numpy as np
from config import Hyper, Constants

def main():
    env = gym.make(Constants.env_id)
    agent = Agent(input_dims=env.observation_space.shape, env=env, n_actions=env.action_space.n)
    filename = f"{Constants.env_id}_{Hyper.n_games}games_scale{agent.scale}_clamp_on_sigma.png"
    figure_file = f'plots/{filename}'

    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = False
    if load_checkpoint:
        agent.load_models()
        env.render(mode='human')
    steps = 0
    for i in range(Hyper.n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)
            new_observation, reward, done, info = env.step(action)
            steps += 1
            agent.remember(observation, action, reward, new_observation, done)
            if not load_checkpoint:
                agent.learn()
            score += reward
            observation = new_observation
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()

        print(f"episode {i}: score {score}, trailing 100 games avg {avg_score}, steps {steps}, {Constants.env_id} scale {agent.scale}")

    if not load_checkpoint:
        x = [i+1 for i in range(Hyper.n_games)]
        plot_learning_curve(x, score_history, figure_file)

if __name__ == '__main__':
    main()
