import os
import torch as T
import torch.nn.functional as F
import numpy as np
from buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork, ValueNetwork
from config import Hyper, Constants
CUDA_LAUNCH_BLOCKING=1

class Agent():
    def __init__(self, input_dims, env, n_actions):
        self.memory = ReplayBuffer(input_dims)
        self.n_actions = n_actions

        self.actor_nn = ActorNetwork(input_dims, n_actions=n_actions, name=Constants.env_id+'_actor', max_action=env.action_space.n)
        self.critic_1_nn = CriticNetwork(input_dims, n_actions=n_actions, name=Constants.env_id+'_critic_1')
        self.critic_2_nn = CriticNetwork(input_dims, n_actions=n_actions, name=Constants.env_id+'_critic_2')
        self.value_nn = ValueNetwork(input_dims, name=Constants.env_id+'_value')
        self.target_value_nn = ValueNetwork(input_dims, name=Constants.env_id+'_target_value')
        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        state = T.Tensor([observation]).to(Constants.device)
        _, _, max_probability_action = self.actor_nn.sample_action(state)
        return max_probability_action

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        if self.memory.mem_cntr < Hyper.batch_size:
            return

        state, action, reward, new_state, done = self.memory.sample_buffer()

        reward = T.tensor(reward, dtype=T.float).to(Constants.device)
        done = T.tensor(done).to(Constants.device)
        new_state = T.tensor(new_state, dtype=T.float).to(Constants.device)
        state = T.tensor(state, dtype=T.float).to(Constants.device)
        action = T.tensor(action, dtype=T.float).to(Constants.device)

        value_from_nn = self.value_nn(state).view(-1)
        new_value_from_nn = self.target_value_nn(new_state).view(-1)
        new_value_from_nn[done] = 0.0
       
        actions, (action_probabilities, log_action_probabilities), max_probability_action = self.actor_nn.sample_action(state)
        actions = actions.to(Constants.device)
        q1_new_policy = self.critic_1_nn.forward(state)
        q2_new_policy = self.critic_2_nn.forward(state)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)
        
        self.value_nn.optimizer.zero_grad()
        value_target = critic_value - log_action_probabilities[:, max_probability_action]   # FIX
        value_loss = 0.5 * (F.mse_loss(value_from_nn, value_target))
        value_loss.backward(retain_graph=True)
        self.value_nn.optimizer.step()

        actions, (action_probabilities, log_action_probabilities), maximum_prob_action = self.actor_nn.sample_action(state)
        q1_new_policy = self.critic_1_nn.forward(state)
        q2_new_policy = self.critic_2_nn.forward(state)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        actor_loss = log_action_probabilities[:, maximum_prob_action] - critic_value    # FIX
        actor_loss = T.mean(actor_loss)
        self.actor_nn.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_nn.optimizer.step()

        self.critic_1_nn.optimizer.zero_grad()
        self.critic_2_nn.optimizer.zero_grad()
        q_hat = Hyper.reward_scale*reward + Hyper.gamma*new_value_from_nn
        q1_old_policy = self.critic_1_nn.forward(state).view(-1)
        q2_old_policy = self.critic_2_nn.forward(state).view(-1)
        critic_1_loss = 0.5*F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5*F.mse_loss(q2_old_policy, q_hat)

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1_nn.optimizer.step()
        self.critic_2_nn.optimizer.step()
        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = Hyper.tau

        target_value_params = self.target_value_nn.named_parameters()
        value_params = self.value_nn.named_parameters()

        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
            value_state_dict[name] = tau*value_state_dict[name].clone() + (1-tau)*target_value_state_dict[name].clone()

        self.target_value_nn.load_state_dict(value_state_dict)

    def save_models(self):
        print('.... saving models ....')
        self.actor_nn.save_checkpoint()
        self.value_nn.save_checkpoint()
        self.target_value_nn.save_checkpoint()
        self.critic_1_nn.save_checkpoint()
        self.critic_2_nn.save_checkpoint()

    def load_models(self):
        print('.... loading models ....')
        self.actor_nn.load_checkpoint()
        self.value_nn.load_checkpoint()
        self.target_value_nn.load_checkpoint()
        self.critic_1_nn.load_checkpoint()
        self.critic_2_nn.load_checkpoint()

    
