import sys

import pylab as plb
import numpy as np
import mountaincar

import pickle
from numpy.random import uniform

# Model parameters
actions = [-1, 0, 1]    # Actions corresponding to "left", "no action", "right"
tau = 100               # "Temperature" of the action selection, controlling the exploration/exploitation ratio
alpha = 0.5             # Learning rate (etha in lecture notes) 0 <= alpha <= 1
gamma = 0.95            # Discount factor: importance of future rewards
lambda_ = 0.5          # 0 < lambda < 1

np.seterr(over='print')


class InputNeuron:
    def __init__(self, x, dx, var_x, var_dx):
        self.center = (x, dx)
        self.x = x
        self.dx = dx
        self.var_x = var_x
        self.var_dx = var_dx

        # One weight going to each output neuron
        self.weight = {a: uniform() for a in actions}
        self.e_trace = {a: 0 for a in actions}

    def response(self, state):
        x, dx = state
        return np.exp(-((self.x - x) ** 2) / self.var_x - ((self.dx - dx) ** 2) / self.var_dx)

    def increase_e_trace(self, state, action):
        for a in actions:
            self.e_trace[a] *= gamma * lambda_

            if a == action:
                self.e_trace[a] += self.response(state)

    def decay_e_traces(self, state, action):
        for a in actions:
            self.e_trace[a] *= gamma * lambda_

            if a == action:
                self.e_trace[a] += self.response(state)

    def update_weights(self, td_error):
        for a in actions:
            d_weight = alpha * td_error * self.e_trace[a]
            self.weight[a] += d_weight


class NeuralNet:
    def __init__(self, x_dim, dx_dim):

        x_centers = np.linspace(-150, 30, x_dim)
        dx_centers = np.linspace(-15, 15, dx_dim)

        var_x = (150 + 30) / x_dim
        var_dx = (15 + 15) / dx_dim

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
        # Numerically stable Softmax
        outputs = [self.output(state, a) for a in actions]

        outputs -= max(outputs)
        output = self.output(state, action) - max(outputs)

        return np.exp(output / tau) / (sum(np.exp(outputs / tau)))

    def next_action(self, state):
        # Select an action based on the action_probability values

        action_probabilities = [self._action_probability(state, action) for action in actions]

        x = uniform()
        cumulative_probability = 0.0
        for action, action_probability in zip(actions, action_probabilities):
            if 1 < action_probability or 0 > action_probability:
                print(state, action, action_probability)

            cumulative_probability += action_probability
            if x < cumulative_probability:
                break
        return action

    def save_to_file(self):
        pickle.dump(self.input_neurons, open("input_neurons.pkl", "wb"))


class Agent():
    """A for the mountain-car task, learning with the Sarsa(λ) algorithm.
    """

    def __init__(self, mountain_car=None):

        if mountain_car is None:
            self.mountain_car = mountaincar.MountainCar()
        else:
            self.mountain_car = mountain_car

        self.nn = NeuralNet(20, 20)


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
            # The actions of mountain_car goes from -1 to 1, while ours goes from 0 to 2
            self.mountain_car.apply_force(action - 1)

            # simulate the timestep
            self.mountain_car.simulate_timesteps(100, 0.01)

            next_state = self.mountain_car.state()
            next_action = self.nn.next_action(next_state)
            reward = self.mountain_car.R

            td_error = reward - self.nn.output(state, action) + gamma * self.nn.output(next_state, next_action)

            self.nn.increase_e_trace(state, action)
            self.nn.update_weights(td_error)

            self.nn.decay_e_traces(state, action)


            state, action, next_state, next_action = next_state, next_action, None, None

            # check if the episode is finished
            if reward > 0.0:
                print("(%d)" % n, end="")
                sys.stdout.flush()
                return self.mountain_car.t

        # Did not reach the reward
        print("(*)", end="")
        return self.mountain_car.t


if __name__ == "__main__":
    n_agents = 10
    n_steps = 10000
    n_episodes = 10000

    results = []
    agents = [Agent() for _ in range(n_agents)]

    print("Starting simulation (%d agents, %d episodes):" % (n_agents, n_episodes))
    for i in range(n_episodes):
        print("Episode %d:" % i, end=' ')
        round = []
        for a in agents:
            res = a.run_episode(n_steps)
            round.append(res)

        results.append(round)
        print('\n\tAverage completion time:', np.mean(round))

    pickle.dump(results, open("results.pkl", "wb"))
    print(results)
    input("Press Enter to continue...")
