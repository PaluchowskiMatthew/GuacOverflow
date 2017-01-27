import sys

import pylab as plb
import numpy as np
import mountaincar
import matplotlib.pyplot as plt

import pickle

class SARSAAgent():
    """A pretty good agent for the mountain-car task.
    """

    def __init__(self, mountain_car=None, size=20, eta=0.05, gamma=0.99, tau=1, eligibity_trace_decay=0.95, tau_decay=True):
        # GridWorld / neural net size
        self.N = size

        self.t = 0

        # reward administered t the target location and when
        # bumping into walls
        self.reward_at_target = 1.

        # learning rate
        self.eta = eta

        # discount factor - quantifies how far into the future
        # a reward is still considered important for the
        # current action
        self.gamma = gamma

        # the decay factor for the eligibility trace the
        # default is 0., which corresponds to no eligibility
        # trace at all.
        self.eligibity_trace_decay = eligibity_trace_decay

        # Exploration parameter
        self.tau = tau
        self.tau_decay = tau_decay

        # Grid centers
        x_centers = np.linspace(-150, 30, self.N)
        dx_centers = np.linspace(-15, 15, self.N)

        # Variance for the input function of the centers
        self.var_x = ((150 + 30) / self.N)**2
        self.var_dx = ((15 + 15) / self.N)**2

        # Create grid given the centers
        self.x_grid, self.dx_grid = np.meshgrid(x_centers, dx_centers)

        if mountain_car is None:
            self.mountain_car = mountaincar.MountainCar()
        else:
            self.mountain_car = mountain_car

        # initialize the weights and eligibility traces
        self.w = np.zeros((3, self.N ** 2))
        self._reset_e_values()

    def _tau(self):
        if self.tau_decay:
            tau = 0.0005 + self.tau * np.exp(- .0001 * self.t)
        else:
            tau = self.tau

        return tau

    def _reset_e_values(self):
        self.traces = np.zeros((3, self.N ** 2))

    def _input_activity(self, state):
        current_x, current_dx = state

        x_dist = np.abs(self.x_grid - current_x).flatten()
        dx_dist = np.abs(self.dx_grid - current_dx).flatten()

        rj = np.exp(-(x_dist ** 2)/self.var_x - (dx_dist ** 2) / self.var_dx)
        return rj # N*N vector

    def _output_activity(self, state):
        rj = self._input_activity(state)

        return np.dot(self.w, rj) # 3 vector

    def _update_traces(self, state, action):
        self.traces[action, :] += self._input_activity(state)
        self.traces[:, :] *= self.gamma * self.eligibity_trace_decay

    def _action_probabilities(self, state):
        # Softmax
        x = self._output_activity(state) / self._tau()
        e_x = np.exp(x - np.max(x))
        probabilities = e_x / e_x.sum()
        return probabilities

    def _next_action(self, state):
        probabilities = self._action_probabilities(state)
        return np.random.choice(3, p=probabilities)



    def visualize_trial(self, agent=None, n_steps = 200):
        """Do a trial without learning, with display.

        Parameters
        ----------
        n_steps -- number of steps to simulate for
        """

        # prepare for the visualization
        plb.ion()
        mv = mountaincar.MountainCarViewer(self.mountain_car)
        mv.create_figure(n_steps, n_steps)
        plb.draw()


        # make sure the mountain-car is reset
        self.mountain_car.reset()

        # get the initial state
        state = self.mountain_car.state()

        # choose an action from the policy:
        action = self._next_action(state)

        for n in range(n_steps):
            # Apply action
            self.mountain_car.apply_force(action - 1)

            # Simulate the time step
            self.mountain_car.simulate_timesteps(100, 0.01)

            # Observe the reward
            reward = self.mountain_car.R

            # check for rewards
            if reward > 0.0:
                print("\rreward obtained at t = ", self.mountain_car.t)
                break

            # Get the next action
            state = self.mountain_car.state()
            action = agent._next_action(state)

        return self.mountain_car.t

    def learn(self, n_episodes, max_steps):

        step_history = []
        for episode in range(n_episodes):
            self.mountain_car.reset()
            self._reset_e_values()

            state = self.mountain_car.state()
            action = self._next_action(state)

            for step in range(max_steps):

                self.mountain_car.apply_force(action - 1)

                # simulate the timestep
                self.mountain_car.simulate_timesteps(100, 0.01)
                self.t += 1  # self.mountain_car.t

                next_state = self.mountain_car.state()
                next_action = self._next_action(next_state)
                reward = self.mountain_car.R

                Q_sa = self._output_activity(state)[action]
                Q_next_sa = self._output_activity(next_state)[next_action]

                self._update_traces(state, action)

                td_error = reward - (Q_sa - self.gamma * Q_next_sa)

                self.w += self.eta * td_error * self.traces

                state, action = next_state, next_action

                # check if the episode is finished
                if reward > 0.0:
                    break

            step_history.append(self.mountain_car.t)
            print("Episode %d: %d" % (episode, self.mountain_car.t))
        return step_history


def explore_tau(n_agents, max_steps, n_episodes):
    agents = {
        ('tau=0, no decay', SARSAAgent(tau=0, tau_decay=False)),
        ('tau=1, no decay', SARSAAgent(tau=1, tau_decay=False)),
        ('tau=inf, no decay', SARSAAgent(tau=np.inf, tau_decay=False)),
        ('tau=1, decay', SARSAAgent(tau=1, tau_decay=True))
    }

    results = {}
    for name, agent in agents:
        result = agent.learn(n_episodes, max_steps)
        results[name] = result
        print(name, results)

    pickle.dump(results, open("tau_variations.pkl", "wb"))


def explore_lambda(n_agents, max_steps, n_episodes):
    agents = {
        ('lambda = 0.95', SARSAAgent(tau=1, tau_decay=True, eligibity_trace_decay=0.95)),
        ('lambda = 0.5', SARSAAgent(tau=1, tau_decay=True, eligibity_trace_decay=0.5)),
        ('lambda = 0.0', SARSAAgent(tau=1, tau_decay=True, eligibity_trace_decay=0.0))
    }

    results = {}
    for name, agent in agents:
        result = agent.learn(n_episodes, max_steps)
        results[name] = result
        print(name, results)

    pickle.dump(results, open("lambda_variations.pkl", "wb"))



if __name__ == "__main__":
    n_agents = 5
    max_steps = 5000
    n_episodes = 100

    #explore_tau(n_agents, max_steps, n_episodes)
    explore_lambda(n_agents, max_steps, n_episodes)




    # results = []
    # agents = [SARSAAgent() for _ in range(n_agents)]
    #
    # print("Starting simulation (%d agents, %d episodes):" % (n_agents, n_episodes))
    # for i in range(n_episodes):
    #     print("Episode %d:" % i, end=' ')
    #     round = []
    #     for a in agents:
    #         res = a.run_episode(max_steps)
    #         round.append(res)
    #
    #     results.append(round)
    #     print('\n\tAverage completion time:', np.mean(round))
    #
    # pickle.dump(results, open("results.pkl", "wb"))
    # print(results)
    # input("Press Enter to continue...")