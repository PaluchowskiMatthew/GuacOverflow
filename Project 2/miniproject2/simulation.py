import sys

import pylab as plb
import numpy as np
import mountaincar

import pickle

from collections import defaultdict
import random

# Model parameters
actions = [-1, 0, 1]
tau = 0.1 # TODO: change during learning
alpha = 0.01 # TODO: find reasonable value
gamma = 0.95 # Given
lambda_ = 0.5 # 0 < lambda < 1
etha = 0.1 # TODO: Find reasonable value

class InputNeuron():
    def __init__(self, x, dx, var_x, var_dx):
        self.center = (x, dx)
        self.x = x
        self.dx = dx
        self.var_x = var_x
        self.var_dx = var_dx

        # One weight going to each output neuron
        self.weight = { a: random.random() for a in actions }
        self.e_trace = { a: 0 for a in actions }

    def response(self, state):
        x, dx = state
        return np.exp(-((self.x - dx) ** 2) / self.var_x - (self.dx - dx) ** 2 / self.var_dx)

    def update_e_trace(self, state, action):
        for a in actions:
            self.e_trace[a] *= gamma * lambda_

            if a == action:
                self.e_trace[a] += self.response(state)

    def update_weights(self, td_error):
        for a in actions:
            # TODO: should it be plus or minus the d_w?
            d_weight = etha * td_error * self.e_trace[a]
            self.weight[a] += d_weight


class NeuralNet():
    def __init__(self, x_dim, dx_dim):

        x_centers = np.linspace(-150, 30, x_dim)
        dx_centers = np.linspace(-15, 15, dx_dim)

        var_x = 180 / x_dim
        var_dx = 30 / dx_dim

        # Grid of input neurons
        self.input_neurons = [InputNeuron(x, dx, var_x, var_dx) for x in x_centers for dx in dx_centers]

    def output(self, state, action):
        # a is the output neuron 1, 2, or 3
        assert(action in actions)
        return sum(inp.response(state) * inp.weight[action] for inp in self.input_neurons)

    def update_e_traces(self, state, action):
        for input_neuron in self.input_neurons:
            input_neuron.update_e_trace(state, action)

    def update_weights(self, td_error):
        for input_neuron in self.input_neurons:
            input_neuron.update_weights(td_error)

    def _action_probability(self, state, action):
        return np.exp(self.output(state, action) / tau) / (sum([np.exp(self.output(state, a) / tau) for a in actions]))

    def next_action(self, state):
        # Select an action based on the action_probability values

        action_probabilities = [self._action_probability(state, action) for action in actions]

        x = random.random()
        cumulative_probability = 0.0
        for action, action_probability in zip(actions, action_probabilities):
            cumulative_probability += action_probability
            if x < cumulative_probability:
                break
        return action

    def save_to_file(self):
        pickle.dump(self.input_neurons, "input_neurons.pkl")



class Agent():
    """A for the mountain-car task, learning with the Sarsa(λ) algorithm.
    """

    def __init__(self, mountain_car=None):

        if mountain_car is None:
            self.mountain_car = mountaincar.MountainCar()
        else:
            self.mountain_car = mountain_car

        self.nn = NeuralNet(20,20)

    def run_simulation(self, n_episodes=9000):
        for episode in range(n_episodes):
            print('Running ', episode, ': ', end='')
            self.run_episode(n_steps=1000)
            self.nn.save_to_file()

    def run_episode(self, n_steps=200):
        """Do a trial without learning, with display.

        Parameters
        ----------
        n_steps -- number of steps to simulate for
        """

        # Initialize
        # make sure the mountain-car is reset
        self.mountain_car.reset()

        # get the initial state
        state = self.mountain_car.state()

        # choose an action from the policy:
        action = self.nn.next_action(state)
        for n in range(n_steps):
            self.mountain_car.apply_force(action)

            # simulate the timestep
            self.mountain_car.simulate_timesteps(100, 0.01)

            next_state = self.mountain_car.state()
            next_action = self.nn.next_action(next_state)
            reward = self.mountain_car.R

            td_error = reward +  gamma * self.nn.output(next_state, next_action) - self.nn.output(state, action)
            self.nn.update_weights(td_error)
            self.nn.update_e_traces(state, action)

            state, action = next_state, next_action

            # check if the episode is finished
            if reward > 0.0:
                print("\rreward obtained at t = ", self.mountain_car.t)
                break
        if reward == 0.0:
            print("\rreward not obtained")


if __name__ == "__main__":
    d = Agent()
    d.run_simulation()
    plb.show()
    input("Press Enter to continue...")
