import os
import torch as T
from torch.distributions import Categorical, normal, MultivariateNormal
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np
from PIL import Image
from config import Hyper, Constants
CUDA_LAUNCH_BLOCKING=1

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, n_actions, name):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = Hyper.layer1_size
        self.fc2_dims = Hyper.layer2_size
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_file = os.path.join(Hyper.chkpt_dir, name+'_sac')
        self.conv1 = nn.Conv2d(input_dims[0], 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        fc_input_dims = self.calculate_conv_output_dims(input_dims)
        self.fc1 = nn.Linear(fc_input_dims, self.fc1_dims) 
        self.fc2 = nn.Linear(self.fc1_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=Hyper.beta)
        self.to(Constants.device)

    def calculate_conv_output_dims(self, input_dims):
        state = T.zeros(1, *input_dims)
        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)
        return int(np.prod(dims.size()))

    def forward(self, state):
        conv1 = F.relu(self.conv1(state))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))   
        conv_state = conv3.view(conv3.size()[0], -1)
        # calculate action value from state
        q1_action_value = F.relu(self.fc1(conv_state))
        action_logits = F.relu(self.fc2(q1_action_value))
        return action_logits
        """ greedy_actions = T.argmax(action_logits, dim=1, keepdim=True)
        return greedy_actions """

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class ActorNetwork(nn.Module):
    def __init__(self, input_dims, n_actions, max_action, name):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = Hyper.layer1_size
        self.fc2_dims = Hyper.layer2_size
        self.n_actions = n_actions
        self.name = name
        self.max_action = max_action
        self.checkpoint_file = os.path.join(Hyper.chkpt_dir, name+'_sac')
        self.reparam_noise = 1e-6
        self.conv1 = nn.Conv2d(input_dims[0], 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        fc_input_dims = self.calculate_conv_output_dims(input_dims)
        self.fc1 = nn.Linear(fc_input_dims, self.fc1_dims) 
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.action_probabilities = nn.Linear(self.fc2_dims, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=Hyper.alpha)
        self.to(Constants.device)

    def calculate_conv_output_dims(self, input_dims):
        state = T.zeros(1, *input_dims)
        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)
        return int(np.prod(dims.size()))

    def forward(self, state):
        conv1 = F.relu(self.conv1(state))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        conv_state = conv3.view(conv3.size()[0], -1)
        prob = F.relu(self.fc1(conv_state))
        prob = F.relu(self.fc2(prob))
        action_probs = F.softmax(self.action_probabilities(prob), dim = 1)
        return action_probs

    def sample_action(self, state):
        # CHANGE0001 
        # It is now more efficient to have the soft Q-function output the Q-value of each possible action rather 
        # than simply the action provided as an input.
        # CHANGE0002
        # There is now no need for our policy to output the mean and covariance of our action distribution, 
        # instead it can directly output our action distribution.
        action_probs = self.forward(state)
        action_probs_1 = F.softmax(action_probs, dim=1)
        action_dist = Categorical(action_probs_1)
        max_probability_action = T.argmax(action_probs, dim=-1)
        max_probability_action = max_probability_action[0].cpu().item()
        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probs == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = T.log(action_probs + z)
        return (action_probs, log_action_probabilities), max_probability_action

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class ValueNetwork(nn.Module):
    def __init__(self, input_dims, name):
        super(ValueNetwork, self).__init__()
        self.state_size = input_dims[0] * input_dims[1] * input_dims[2]
        self.fc1_dims = Hyper.layer1_size
        self.fc2_dims = Hyper.layer2_size
        self.name = name
        self.checkpoint_file = os.path.join(Hyper.chkpt_dir, name+'_sac')
        self.conv1 = nn.Conv2d(input_dims[0], 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        fc_input_dims = self.calculate_conv_output_dims(input_dims)
        self.fc1 = nn.Linear(fc_input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.v = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=Hyper.beta)
        self.to(Constants.device)

    def calculate_conv_output_dims(self, input_dims):
        state = T.zeros(1, *input_dims)
        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)
        return int(np.prod(dims.size()))

    def forward(self, state):
        conv1 = F.relu(self.conv1(state))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        conv_state = conv3.view(conv3.size()[0], -1)
        state_value = F.relu(self.fc1(conv_state))
        state_value = F.relu(self.fc2(state_value))
        v = self.v(state_value)

        return v

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

