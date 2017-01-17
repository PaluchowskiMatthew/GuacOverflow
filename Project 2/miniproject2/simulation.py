import sys

import pylab as plb
import numpy as np
import mountaincar

from collections import defaultdict
import random


class Neural_net():
    def __init__(self, x_dim, psi_dim):

        # Input layer
        self.x_dim = x_dim
        self.psi_dim = psi_dim
        self.weights = np.array(random.random()) * range(x_dim * psi_dim)

        x_centers = np.linspace(-150, 30, x_dim)
        psi_centers = np.linspace(-15, 15, psi_dim)

        # probably don't need this
        self.centers = list(zip(x_centers, psi_centers))

        # Parameters for the gaussians
        self.x_var = 9.167
        self.psi_var = 1.677

    def _response(self, j, state):
        x, psi = state
        response = np.exp(-((self.x_centers[j] - x)**2) / self.x_var - (self.psi_centers[j] - psi)**2 / self.psi_var)
        response

    def _responses(self, state):
        return np.array(self._response(j, state) for j in range(self.x_dim * self.psi_dim))

    def output(self, a, state):
        # a is the output neuron 1, 2, or 3
        assert(0 <= a <= 2)
        return sum(np.transpose(self.weights).dot(self._responses(state)))



class Sarsa():
    def __init__(self):
        self.Q = defaultdict(int)
        self.actions = [1, 0, -1]
        self.tau = 0.1 # TODO: What should this be?
        self.alpha = 0.01 # TODO: What should this be?
        self.gamma = 0.95

        self

    def action_value(self, state, action):
        """ The action-value function Q(state, action)
        :param state: a tuple (x, x_d), containing the position and speed in the x-plane
        :param action: 1,0, or -1, corresponding to left, no action, right
        :return: the action-state value
        """
        return self.Q[(state, action)]

    def action_probability(self, state, action):
        return np.exp(self.action_value(state, action) / self.tau) / (sum([np.exp(self.action_value(state, action_) / self.tau) for action_ in self.actions]))


    def update_action_value(self, state, action, reward, next_state, next_action):
        Q = self.Q[(state, action)]
        Q_ = self.Q[(next_state, next_action)]

        self.Q[(state, action)] = Q + self.alpha * (reward + self.gamma * Q_ - Q)

    def next_action(self, state):
        # Select an action based on the action_probability values

        action_probabilities = [self.action_probability(state, action) for action in self.actions]

        x = random.uniform(0, 1)
        cumulative_probability = 0.0

        for action, action_probability in zip(self.actions, action_probabilities):
            cumulative_probability += action_probability
            if x < cumulative_probability:
                break

        return action


class Agent():
    """A for the mountain-car task, learning with the Sarsa(λ) algorithm.
    """

    def __init__(self, mountain_car=None):

        if mountain_car is None:
            self.mountain_car = mountaincar.MountainCar()
        else:
            self.mountain_car = mountain_car

    def run_episode(self, n_steps=200):
        """Do a trial without learning, with display.

        Parameters
        ----------
        n_steps -- number of steps to simulate for
        """

        # Initialize the sarsa algorithm
        sarsa = Sarsa()

        # prepare for the visualization
        plb.ion()
        mv = mountaincar.MountainCarViewer(self.mountain_car)
        mv.create_figure(n_steps, n_steps)
        plb.draw()

        # Initialize
        # make sure the mountain-car is reset
        self.mountain_car.reset()

        # get the initial state
        state = self.mountain_car.state()

        # choose an action from the policy:
        action = sarsa.next_action(state)
        for n in range(n_steps):
            self.mountain_car.apply_force(action)

            # simulate the timestep
            self.mountain_car.simulate_timesteps(100, 0.01)

            next_state = self.mountain_car.state()
            reward = self.mountain_car.R
            next_action = sarsa.next_action(next_state)

            sarsa.update_action_value(state, action, reward, next_state, next_action)

            state, action = next_state, next_action

            # update the visualization
            if n % 10 == 0:
                print('\rt =', self.mountain_car.t)
                sys.stdout.flush()
                mv.update_figure()
                plb.show()
                plb.pause(1e-100)

            # check if the episode is finished
            if reward > 0.0:
                print("\rreward obtained at t = ", self.mountain_car.t)
                break


if __name__ == "__main__":
    nn = Neural_net(20,20)


    #d = Agent()
    #d.run_episode(5000)
    #plb.show()
    #input("Press Enter to continue...")
