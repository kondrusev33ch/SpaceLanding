"""No changes here"""
import numpy as np


class ReplayBuffer:
    def __init__(self, max_size=10_000, batch_size=64):
        # Initialize five arrays to hold states, actions, reward, next state, and flags
        self.ss_mem = np.empty(shape=max_size, dtype=np.ndarray)
        self.as_mem = np.empty(shape=max_size, dtype=np.ndarray)
        self.rs_mem = np.empty(shape=max_size, dtype=np.ndarray)
        self.ps_mem = np.empty(shape=max_size, dtype=np.ndarray)
        self.fs_mem = np.empty(shape=max_size, dtype=np.ndarray)  # is terminal in our case

        # Initialize variables to do storage and sampling
        self.max_size = max_size
        self.batch_size = batch_size
        self._idx = 0
        self.size = 0

    def store(self, sample):
        # Unwrapping the sample and setting arrays
        s, a, r, p, f = sample
        self.ss_mem[self._idx] = s
        self.as_mem[self._idx] = a
        self.rs_mem[self._idx] = r
        self.ps_mem[self._idx] = p
        self.fs_mem[self._idx] = f

        # Increase index
        self._idx += 1
        # Return to the beginning if we reach end
        self._idx = self._idx % self.max_size

        # Increase size
        self.size += 1
        self.size = min(self.size, self.max_size)

    def sample(self, batch_size=None):
        # Determining the batch size
        if batch_size is None:
            batch_size = self.batch_size

        # Sample batch_size ids from 0 to size
        idxs = np.random.choice(self.size, batch_size, replace=False)

        # Extract the experiences from the buffer using the sampled ids
        experiences = np.vstack(self.ss_mem[idxs]), \
                      np.vstack(self.as_mem[idxs]), \
                      np.vstack(self.rs_mem[idxs]), \
                      np.vstack(self.ps_mem[idxs]), \
                      np.vstack(self.fs_mem[idxs])
        # vstack() function is used to stack arrays in sequence vertically (row wise)
        return experiences
