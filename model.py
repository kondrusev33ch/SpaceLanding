import torch.cuda
import torch.nn as nn


class FCQ(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_dims=(64,),
                 activation_fc=nn.Tanh()):
        super(FCQ, self).__init__()
        self.activation_fc = activation_fc

        # Create input layer
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])

        # Create hidden layers
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i + 1])
            self.hidden_layers.append(hidden_layer)

        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

        # Set device
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
        self.device = torch.device(device)
        self.to(self.device)

    def _format(self, state):
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)
            x = x.unsqueeze(0)
        return x

    def forward(self, state):
        x = self._format(state)  # convert to tensor

        # Then we pass it through the input layer and then through the activation function
        x = self.activation_fc(self.input_layer(x))

        # Then we do the same for all hidden layers
        for hidden_layer in self.hidden_layers:
            x = self.activation_fc(hidden_layer(x))

        # And finally, for the output layer, without activation function of course
        x = self.output_layer(x)
        return x

    def load(self, experiences):
        # Process data
        states, actions, new_states, rewards, is_terminals = experiences
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device)
        new_states = torch.from_numpy(new_states).float().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        is_terminals = torch.from_numpy(is_terminals).float().to(self.device)
        return states, actions, new_states, rewards, is_terminals

    def act(self, state):
        # Make an action
        state_t = torch.as_tensor(state, device=self.device, dtype=torch.float32)
        q_values = self(state_t.unsqueeze(0))

        max_q_index = torch.argmax(q_values, dim=1)[0]
        action = max_q_index.detach().item()
        return action
