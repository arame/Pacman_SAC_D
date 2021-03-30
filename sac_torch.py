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
        self.gamma = Hyper.beta
        self.tau = Hyper.tau
        self.memory = ReplayBuffer(input_dims)
        self.batch_size = Hyper.batch_size
        self.n_actions = n_actions

        self.actor = ActorNetwork(input_dims, n_actions=n_actions, name=Constants.env_id+'_actor', max_action=env.action_space.n)
        self.critic_1 = CriticNetwork(input_dims, n_actions=n_actions, name=Constants.env_id+'_critic_1')
        self.critic_2 = CriticNetwork(input_dims, n_actions=n_actions, name=Constants.env_id+'_critic_2')
        self.value = ValueNetwork(input_dims, name=Constants.env_id+'_value')
        self.target_value = ValueNetwork(input_dims, name=Constants.env_id+'_target_value')

        self.scale = Hyper.reward_scale
        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        state = T.Tensor([observation]).to(Constants.device)
        _, _, max_probability_action = self.actor.sample_action(state)
        return max_probability_action

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = self.memory.sample_buffer()

        reward = T.tensor(reward, dtype=T.float).to(Constants.device)
        done = T.tensor(done).to(Constants.device)
        state_ = T.tensor(new_state, dtype=T.float).to(Constants.device)
        state = T.tensor(state, dtype=T.float).to(Constants.device)
        action = T.tensor(action, dtype=T.float).to(Constants.device)

        value = self.value(state).view(-1)
        value_ = self.target_value(state_).view(-1)
        value_[done] = 0.0
       
        actions, probabilities, _ = self.actor.sample_action(state)
        log_probs = probabilities[1].view(-1)
        actions = actions.to(Constants.device)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)
        
        self.value.optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * (F.mse_loss(value, value_target))
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        actions, probabilities, maximum_prob_action = self.actor.sample_action(state)
        log_probs = probabilities[1].view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        actor_loss = log_probs - critic_value
        actor_loss = T.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        q_hat = self.scale*reward + self.gamma*value_
        q1_old_policy = self.critic_1.forward(state, action).view(-1)
        q2_old_policy = self.critic_2.forward(state, action).view(-1)
        critic_1_loss = 0.5*F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5*F.mse_loss(q2_old_policy, q_hat)

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()
        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
            value_state_dict[name] = tau*value_state_dict[name].clone() + \
                    (1-tau)*target_value_state_dict[name].clone()

        self.target_value.load_state_dict(value_state_dict)

    def save_models(self):
        print('.... saving models ....')
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()

    def load_models(self):
        print('.... loading models ....')
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()

    
