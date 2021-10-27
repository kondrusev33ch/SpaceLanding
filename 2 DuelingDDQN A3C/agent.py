# Resource: https://github.com/mimoralea/gdrl

from environment_mp import CapsuleLanderMP
from replay_buffer import ReplayBuffer
from model import FCDuelingQ, FCDAP, FCV
from optimizers import SharedAdam, SharedRMSprop
from IPython import display
from itertools import count
from collections import deque
from statistics import mean
from time import time
import matplotlib.pyplot as plt
import random
import torch
import torch.optim as optim
import torch.multiprocessing as mp
import numpy as np

HIDDEN_DIMS = (128, 128)
LR = 0.0005

EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 10_000


# -----------------------------------------------------
def plot(mean_rewards):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Episodes')
    plt.ylabel('Episode Reward')
    plt.plot(mean_rewards)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(.1)


# -----------------------------------------------------
class DuelingDDQN:
    def __init__(self):
        plt.ion()  # for real time plotting

        self.env = CapsuleLanderMP()
        self.input_size = self.env.state.size
        self.output_size = self.env.action.size

        self.target_model = FCDuelingQ(self.input_size, self.output_size, HIDDEN_DIMS)
        self.online_model = FCDuelingQ(self.input_size, self.output_size, HIDDEN_DIMS)
        self.update_network(1.0)

        self.optimizer = optim.RMSprop(self.online_model.parameters(), lr=LR)
        self.replay_buffer = ReplayBuffer(max_size=50_000, batch_size=64)

        self.gamma = 0.99
        self.epsilon = 1.0
        self.n_warmup_batches = 5
        self.episode_reward = []
        self.step = 1

    def optimize_model(self, experiences):
        states, actions, rewards, next_states, is_terminals = experiences
        batch_size = len(is_terminals)

        argmax_a_q_sp = self.online_model(next_states).max(1)[1]
        q_sp = self.target_model(next_states).detach()
        max_a_q_sp = q_sp[np.arange(batch_size), argmax_a_q_sp].unsqueeze(1)
        target_q_sa = rewards + (self.gamma * max_a_q_sp * (1 - is_terminals))
        q_sa = self.online_model(states).gather(1, actions)

        td_error = q_sa - target_q_sa
        value_loss = td_error.pow(2).mul(0.5).mean()
        self.optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_model.parameters(), float('inf'))
        self.optimizer.step()

    def update_network(self, tau):
        for target, online in zip(self.target_model.parameters(),
                                  self.online_model.parameters()):
            # tau representing the ratio of the online net that will be
            # mixed into the target network
            target_ratio = (1.0 - tau) * target.data
            online_ratio = tau * online.data

            # Mix the weights and copy into the target network
            mixed_weights = target_ratio + online_ratio
            target.data.copy_(mixed_weights)

    def interaction_step(self, state):
        if random.random() <= self.epsilon:
            action = self.env.get_random_action()
        else:
            action = self.get_net_action(state)

        new_state, reward, is_terminal, info = self.env.step(action)
        experience = (state, action, reward, new_state, float(is_terminal))
        self.replay_buffer.store(experience)
        self.episode_reward[-1] += reward

        return new_state, is_terminal

    def train(self, max_episodes=400):
        start_time = time()
        for episode in range(1, max_episodes + 1):
            state, is_terminal = self.env.reset(), False
            self.episode_reward.append(0.0)

            while not is_terminal:
                self.epsilon = np.interp(self.step, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])
                if self.step % EPSILON_DECAY != 0:
                    self.step += 1

                state, is_terminal = self.interaction_step(state)

                min_samples = self.replay_buffer.batch_size * self.n_warmup_batches
                if self.replay_buffer.size > min_samples:
                    experiences = self.replay_buffer.sample()
                    experiences = self.online_model.load(experiences)
                    self.optimize_model(experiences)

                self.update_network(1.0)

            # print()
            # print('Episode:', episode)
            # print('Reward:', self.episode_reward[-1])
            plot(self.episode_reward)

        return time() - start_time

    def get_net_action(self, state):
        with torch.no_grad():
            q_values = self.online_model(state).cpu().numpy()
        return np.argmax(q_values)


# -----------------------------------------------------
class A3C:
    def __init__(self, n_workers=8):
        self.env = CapsuleLanderMP()
        self.input_size = self.env.state.size
        self.output_size = self.env.action.size

        self.shared_policy_model = FCDAP(self.input_size,
                                         self.output_size,
                                         hidden_dims=(128, 64)).share_memory()
        self.shared_policy_optimizer = SharedAdam(self.shared_policy_model.parameters(), lr=LR)

        self.shared_value_model = FCV(self.input_size, hidden_dims=(256, 128)).share_memory()
        self.shared_value_optimizer = SharedRMSprop(self.shared_value_model.parameters(), lr=LR)

        self.get_out_lock = mp.Lock()
        self.get_out_signal = torch.zeros(1, dtype=torch.int).share_memory_()
        self.reached_max_episodes = torch.zeros(1, dtype=torch.int).share_memory_()

        self.n_workers = n_workers
        self.max_n_steps = 200

        self.gamma = 0.99

        self.rewards = None

    def optimize_model(self, logpas, entropies, rewards, values, policy_model, value_model):
        T = len(rewards)
        discounts = np.logspace(0, T, num=T, base=self.gamma, endpoint=False)
        returns = np.array([np.sum(discounts[:T - t] * rewards[t:]) for t in range(T)])

        discounts = torch.FloatTensor(discounts[:-1]).unsqueeze(1)
        returns = torch.FloatTensor(returns[:-1]).unsqueeze(1)

        logpas = torch.cat(logpas)
        entropies = torch.cat(entropies)
        values = torch.cat(values)

        value_error = returns - values
        policy_loss = -(discounts * value_error.detach() * logpas).mean()
        entropy_loss = -entropies.mean()

        loss = policy_loss + 0.001 * entropy_loss
        self.shared_policy_optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1)
        for param, shared_param in zip(policy_model.parameters(),
                                       self.shared_policy_model.parameters()):
            if shared_param.grad is None:
                shared_param._grad = param.grad

        self.shared_policy_optimizer.step()
        policy_model.load_state_dict(self.shared_policy_model.state_dict())

        value_loss = value_error.pow(2).mul(0.5).mean()
        self.shared_value_optimizer.zero_grad()
        value_loss.backward()

        torch.nn.utils.clip_grad_norm_(value_model.parameters(), float('inf'))
        for param, shared_param in zip(value_model.parameters(),
                                       self.shared_value_model.parameters()):
            if shared_param.grad is None:
                shared_param._grad = param.grad

        self.shared_value_optimizer.step()
        value_model.load_state_dict(self.shared_value_model.state_dict())

    @staticmethod
    def interaction_step(state, env, policy_model, value_model,
                         logpas, entropies, rewards, values):
        action, logpa, entropy = policy_model.full_pass(state)
        new_state, reward, is_terminal, info = env.step(action)

        logpas.append(logpa)
        entropies.append(entropy)
        rewards.append(reward)
        values.append(value_model(state))

        return new_state, reward, is_terminal, True  # True, because our env isn't endless

    def work(self, index, max_episodes):
        env = CapsuleLanderMP()
        input_size = env.state.size
        output_size = env.action.size

        local_policy_model = FCDAP(input_size, output_size, hidden_dims=(128, 64))
        local_policy_model.load_state_dict(self.shared_policy_model.state_dict())

        local_value_model = FCV(input_size, hidden_dims=(256, 128))
        local_value_model.load_state_dict(self.shared_value_model.state_dict())

        episode = 0
        while not self.get_out_signal and episode < max_episodes:
            state, is_terminal = env.reset(), False
            n_steps_start, total_episode_steps, total_episode_rewards = 0, 0, 0
            logpas, entropies, rewards, values = [], [], [], []

            for step in count(start=1):
                state, reward, is_terminal, is_truncated = self.interaction_step(
                    state, env, local_policy_model, local_value_model,
                    logpas, entropies, rewards, values)

                total_episode_steps += 1
                total_episode_rewards += reward

                if is_terminal or step - n_steps_start == self.max_n_steps:
                    is_failure = is_terminal and not is_truncated
                    next_value = 0 if is_failure else local_value_model(state).detach().item()
                    rewards.append(next_value)

                    self.optimize_model(logpas, entropies, rewards, values,
                                        local_policy_model, local_value_model)
                    logpas, entropies, rewards, values = [], [], [], []
                    n_steps_start = step

                if is_terminal:
                    # print(episode, total_episode_rewards)
                    self.rewards[index][episode] = total_episode_rewards
                    episode += 1
                    break

    def train(self, max_episodes=200):
        self.rewards = torch.zeros(self.n_workers, max_episodes).share_memory_()
        start_time = time()

        workers = [mp.Process(target=self.work, args=(index, max_episodes)) for index in range(self.n_workers)]
        [w.start() for w in workers]
        [w.join() for w in workers]

        return time() - start_time

    def plot_data(self):
        """Just for fun, you will not see useful data on curves"""
        for worker in range(self.n_workers):
            plt.plot(self.rewards[worker].numpy())

        # Make sure you didn't call plt.ion() in DuelingDDQN()
        plt.show()


# -----------------------------------------------------
def show_result(env, act):
    rewards = deque(maxlen=20)
    state = env.reset()
    game_reward, games = 0, 0
    while games != 20:
        action = act(state)
        state, reward, done, _ = env.step(action)
        game_reward += reward
        env.render()
        if done:
            rewards.append(game_reward)
            game_reward = 0
            games += 1
            env.reset()

    env.close()
    return rewards


# -----------------------------------------------------
if __name__ == '__main__':
    plt.style.use('fivethirtyeight')

    agent = DuelingDDQN()
    d_time = agent.train(200)
    d_reward = mean(show_result(agent.env, agent.get_net_action))

    agent = A3C()
    a_time = agent.train(200)
    a_reward = mean(show_result(agent.env, agent.shared_policy_model.select_action))
    # Note: Once you can get 300+ average reward, next time -60.
    # So 200 episodes is not enough to open it potential.

    print('DuelingDDQN')
    print(f'    Trained {d_time} seconds')
    print(f'    Trained model mean reward based on 20 games: {d_reward}')
    print('A3C')
    print(f'    Trained {a_time} seconds')
    print(f'    Trained model mean reward based on 20 games: {a_reward}')
