import sys

import pylab as plb
import numpy as np
import mountaincar
import matplotlib.pyplot as plt

import pickle

class SARSAAgent():
    """A not so good agent for the mountain-car task.
    """

    def __init__(self, mountain_car=None, N=20, eta=0.005, gamma=0.99, tau=0.01, lambda_eligibility=0.95):
        # Gridworld / neural net size
        self.N = N

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
        self.lambda_eligibility = lambda_eligibility

        # Exploration parameter
        self.tau = tau

        # Grid centers
        x_centers = np.linspace(-150, 30, self.N)
        dx_centers = np.linspace(-15, 15, self.N)

        # Gaussian width between the centers
        self.var_x = ((150 + 30) / self.N)**2
        self.var_dx = ((15 + 15) / self.N)**2

        # Create grid given the centers
        self.x_grid, self.dx_grid = np.meshgrid(x_centers, dx_centers)

        if mountain_car is None:
            self.mountain_car = mountaincar.MountainCar()
        else:
            self.mountain_car = mountain_car

        # initialize the Q-values etc.
        self._init_values()

    def _init_values(self):
        """
        Initialize the Q-values, eligibility trace, position etc.
        """
        # initialize the Q-values, weights and the eligibility trace
        self.w = np.zeros((3, self.N ** 2))
        self._reset_e_values()

        # list that contains the times it took the agent to reach the target for all trials
        # serves to track the progress of learning
        self.latency_list = []

    def _reset_e_values(self):
        self.e = np.zeros((3, self.N**2))

    def _rj_activity(self, state):
        current_x, current_dx = state

        x_dist = np.abs(self.x_grid - current_x).flatten()
        dx_dist = np.abs(self.dx_grid - current_dx).flatten()

        rj = np.exp(-(x_dist ** 2)/self.var_x - (dx_dist ** 2) / self.var_dx)
        return rj # N*N vector

    def _Q_activity(self, state):
        rj = self._rj_activity(state)

        return np.dot(self.w, rj) # 3 vector

    def _e_udpdate(self, state, action):
        self.e[action, :] += self._rj_activity(state)
        self.e[:, :] *= self.gamma * self.lambda_eligibility


    def _action_probabilities(self, state):
        # Softmax

        Q_values = self._Q_activity(state)
        return (np.exp(Q_values / self.tau))/(np.sum(np.exp(Q_values / self.tau)))

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

        # get our SARSA agent
        if agent is None:
            self.agent = SARSAAgent()
        else:
            self.agent = agent

        # make sure the mountain-car is reset
        self.mountain_car.reset()

        # get the initial state
        state = self.mountain_car.state()

        # choose an action from the policy:
        action = self._next_action(state)

        for n in range(n_steps):
            #print('\rtime =', self.mountain_car.t)
            #sys.stdout.flush()

            # choose action from the policy
            self.mountain_car.apply_force(action - 1)
            # simulate the timestep
            self.mountain_car.simulate_timesteps(100, 0.01)

            next_state = self.mountain_car.state()
            next_action = agent._next_action(next_state)
            reward = self.mountain_car.R

            # check for rewards
            if self.mountain_car.R > 0.0:
                print("\rreward obtained at t = ", self.mountain_car.t)
                break


        return self.mountain_car.t



    def learn(self):
        return

#
# if __name__ == "__main__":
#     s = SARSAAgent()
#     s.learn()
#     plb.show()


class Simulator():

    def __init__(self, mountain_car=None):

        if mountain_car is None:
            self.mountain_car = mountaincar.MountainCar()
        else:
            self.mountain_car = mountain_car

        self.SARSA = SARSAAgent(mountain_car = mountain_car)


    def run_episode(self, n_steps=200):
        """Do a trial with learning, without display.

        Parameters
        ----------
        n_steps -- number of steps to simulate for
        """

        # Initialize
        # make sure the mountain-car is reset
        self.mountain_car.reset()
        # and so is SARSAAgent eligibility trace
        self.SARSA._reset_e_values()

        # get the initial state
        state = self.mountain_car.state()

        # choose an action from the policy:
        action = self.SARSA._next_action(state)

        for n in range(n_steps):
            self.mountain_car.apply_force(action -1)

            # simulate the timestep
            self.mountain_car.simulate_timesteps(100, 0.01)

            next_state = self.mountain_car.state()
            next_action = self.SARSA._next_action(next_state)
            reward = self.mountain_car.R

            Q_sa = self.SARSA._Q_activity(state)[action]
            Q_next_sa = self.SARSA._Q_activity(next_state)[next_action]

            self.SARSA._e_udpdate(state, action)

            td_error = reward - (Q_sa - self.SARSA.gamma * Q_next_sa)

            self.SARSA.w += self.SARSA.eta * td_error * self.SARSA.e

            state, action = next_state, next_action

            # check if the episode is finished
            if reward > 0.0:
                sys.stdout.flush()
                return self.mountain_car.t


        # Did not reach the reward
        print("(*)", end="")
        return self.mountain_car.t

if __name__ == "__main__":

    agent = SARSAAgent()


    #state = (-60,0)
    #activity = agent._rj_activity(state).reshape((20, 20))
    #plt.matshow(activity)
    #plt.show()
    #input("...")



    n_agents = 5
    n_steps = 5000
    n_episodes = 200

    results = []
    agents = [Simulator() for _ in range(n_agents)]

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