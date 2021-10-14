from environment import CapsuleLander
from replay_buffer import ReplayBuffer
from model import FCQ
from collections import deque
from itertools import count
from IPython import display
import matplotlib.pyplot as plt
import random
import torch.optim as optim
import numpy as np

MIN_REPLAY_SIZE = 1000  # how many transitions we want in replay buffer before we start computing gradients and training
HIDDEN_DIMS = (128,)
LR = 0.0005

EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 10_000  # decay period

GAMMA = 0.99  # discount rate
BATCH_SIZE = 32  # how many transitions we're going to sample from the replay buffer
TARGET_UPDATE_FREQ = 1000  # number of steps where we set the target parameters equal to the online params


# -----------------------------------------------------
class DQN:
    def __init__(self):
        self.replay_buffer = ReplayBuffer(max_size=50_000, batch_size=64)
        self.reward_buffer = deque([0.0], maxlen=100)
        self.env = CapsuleLander()

        self.target_model = None
        self.online_model = None
        self.optimizer = None
        self.steps = deque(maxlen=1000)
        self.mean_rewards = deque(maxlen=1000)

    def train(self):
        # Get number of input and output neurons
        in_layer = self.env.state.size
        out_layer = self.env.action.size

        # Build target and online models
        self.target_model = FCQ(in_layer, out_layer, hidden_dims=HIDDEN_DIMS)
        self.online_model = FCQ(in_layer, out_layer, hidden_dims=HIDDEN_DIMS)
        self.target_model.load_state_dict(self.online_model.state_dict())

        # Create optimizer
        self.optimizer = optim.Adam(self.online_model.parameters(), lr=LR)

        # Fill replay buffer
        observation = self.env.reset()
        for _ in range(MIN_REPLAY_SIZE):
            # Get random action
            action = self.env.get_random_action()

            # Get new_observation, reward, is_terminal, info from environment
            new_observation, reward, is_terminal, _ = self.env.step(action)

            # Save all to buffer
            experience = (observation, action, reward, new_observation, float(is_terminal))
            self.replay_buffer.store(experience)

            # Update observation
            observation = new_observation

            if is_terminal:  # start new game
                observation = self.env.reset()

        # Train
        observation = self.env.reset()
        episode_reward = 0.0
        for step in count():
            # Create exploration exploitation strategy
            epsilon = np.interp(step, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])  # from 1 to 0.02 random actions

            # Implement exploration exploitation strategy
            rnd_sample = random.random()
            if rnd_sample <= epsilon:  # get random action
                action = self.env.get_random_action()
            else:
                action = self.online_model.act(observation)  # get action from model

            # Get data from environment
            new_observation, reward, is_terminal, _ = self.env.step(action)
            experience = (observation, action, reward, new_observation, float(is_terminal))

            # Save data to replay_buffer
            self.replay_buffer.store(experience)
            observation = new_observation

            # Accumulate reward
            episode_reward += reward

            if is_terminal:  # start new game
                observation = self.env.reset()

                # Save episode reward to reward buffer
                self.reward_buffer.append(episode_reward)
                episode_reward = 0.0  # do not forget to zero it down

            # Start gradient step
            experiences = self.replay_buffer.sample()
            experiences = self.online_model.load(experiences)
            self.optimize_model(experiences)

            # Update target network
            if step % TARGET_UPDATE_FREQ == 0:
                self.target_model.load_state_dict(self.online_model.state_dict())

            # Logging
            if step % 1000 == 0:
                print()
                self.steps.append(step/1000)
                print('Step:', step)

                mean_reward = np.mean(self.reward_buffer)
                self.mean_rewards.append(mean_reward)
                print('Average reward:', mean_reward)

                plot(self.mean_rewards)

    def optimize_model(self, experiences):
        observations, actions, rewards, new_observations, is_terminals = experiences

        # Query a target network to get the estimate of the next state
        q_sp = self.target_model(new_observations).detach()

        # We grab the maximum of those values, and make sure to treat terminal states appropriately
        max_a_q_sp = q_sp.max(1)[0].unsqueeze(1)
        max_a_q_sp *= (1 - is_terminals)

        # We create the TD targets
        target_q_sa = rewards + GAMMA * max_a_q_sp

        # Query the current "online" estimate
        q_sa = self.online_model(observations).gather(1, actions)

        # Use those values to create the errors
        td_error = q_sa - target_q_sa
        value_loss = td_error.pow(2).mul(0.5).mean()
        self.optimizer.zero_grad()
        value_loss.backward()
        self.optimizer.step()


# -----------------------------------------------------
def plot(mean_rewards):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Steps in thousands')
    plt.ylabel('Average Reward')
    plt.plot(mean_rewards)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(.1)


# -----------------------------------------------------
if __name__ == '__main__':
    plt.ion()
    plt.style.use('fivethirtyeight')

    agent = DQN()
    agent.train()
