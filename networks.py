import os
import torch as T
from torch.distributions import Categorical, normal, MultivariateNormal
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np
from config import Hyper, Constants

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, n_actions, name):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = Hyper.layer1_size
        self.fc2_dims = Hyper.layer2_size
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_file = os.path.join(Constants.chkpt_dir, name+'_sac')
        self.conv1 = nn.Conv2d(input_dims[0], 32, 1)
        self.conv2 = nn.Conv2d(32, 64, 1)
        self.conv3 = nn.Conv2d(64, 64, 1)
        fc_input_dims = self.calculate_conv_output_dims(input_dims)
        self.fc1 = nn.Linear(fc_input_dims, self.fc1_dims) 
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q1 = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=Hyper.beta)
        self.to(Constants.device)

    def calculate_conv_output_dims(self, input_dims):
        state = T.zeros(1, *input_dims)
        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)
        return int(np.prod(dims.size()))

    def forward(self, state, action):
        conv1 = F.relu(self.conv1(state))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        conv_state = conv3.view(conv3.size()[0], -1)
        q1_action_value = T.cat([conv_state, action], dim=1)
        q1_action_value = F.relu(self.fc1(q1_action_value))
        q1_action_value = F.relu(self.fc2(q1_action_value))

        q1 = self.q1(q1_action_value)

        return q1

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
        self.checkpoint_file = os.path.join(Constants.chkpt_dir, name+'_sac')
        self.reparam_noise = 1e-6
        self.conv1 = nn.Conv2d(input_dims[0], 32, 1)
        self.conv2 = nn.Conv2d(32, 64, 1)
        self.conv3 = nn.Conv2d(64, 64, 1)
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
        action_probs = self.action_probabilities(prob)
        return action_probs

    def sample_normal(self, state):
        action_probs = self.forward(state)
        max_probability_action = torch.argmax(action_probabilities, dim=-1)

        action_distribution = Categorical(action_probs)
        action = action_distribution.sample().cpu()
        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probabilities == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probabilities + z)
        return action, (action_probs, log_action_probabilities), max_probability_action

    def sample_mvnormal(self, state, reparameterize=True):
        """
            Doesn't quite seem to work.  The agent never learns.
        """
        mu, sigma = self.forward(state)
        n_batches = sigma.size()[0]

        cov = [sigma[i] * T.eye(self.n_actions).to(self.device) for i in range(n_batches)]
        cov = T.stack(cov)
        probabilities = T.distributions.MultivariateNormal(mu, cov)

        if reparameterize:
            actions = probabilities.rsample() # reparameterizes the policy
        else:
            actions = probabilities.sample()

        action = T.tanh(actions) # enforce the action bound for (-1, 1)
        log_probs = probabilities.log_prob(actions)
        log_probs -= T.sum(T.log(1-action.pow(2) + self.reparam_noise))
        log_probs = log_probs.sum(-1, keepdim=True)

        return action, log_probs

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
        self.checkpoint_file = os.path.join(Constants.chkpt_dir, name+'_sac')
        self.conv1 = nn.Conv2d(input_dims[0], 32, 1)
        self.conv2 = nn.Conv2d(32, 64, 1)
        self.conv3 = nn.Conv2d(64, 64, 1)
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

