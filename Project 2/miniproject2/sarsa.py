import sys

import pylab as plb
import numpy as np
import mountaincar

import pickle

class SARSAAgent():
    """A not so good agent for the mountain-car task.
    """

    def __init__(self, mountain_car = None, N=20, eta = 0.01, gamma=0.95, tau=0.1, lambda_eligibility=0.7):
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
        self.var_x = (150 + 30) / self.N
        self.var_dx = (15 + 15) / self.N

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
        self.w = 0.01 * np.random.rand(3, self.N**2) + 0.1
        self._reset_values()

        # list that contains the times it took the agent to reach the target for all trials
        # serves to track the progress of learning
        self.latency_list = []

    def _reset_values(self):
        self.Q = 0.01 * np.random.rand(self.N,self.N,3) + 0.1
        self.e = np.zeros((3, self.N**2))


    def _rj_activity(self, state):
        current_x, current_dx = state

        # r_j for all j
        rj = np.exp(-((self.x_grid - current_x) ** 2) / self.var_x - ((self.dx_grid - current_dx) ** 2) / self.var_dx)

        return rj.flatten() # N*N vector

    def _Q_activity(self, state):
        rj = self._rj_activity(state)

        return np.dot(self.w, rj) # 3 vector

    def _w_update(self):
        delta = self._TD_error()
        delta_w = self.eta * np.multiply(delta, self.e)
        self.w += delta_w

    def _e_udpdate(self, state, action):
        self.e *= self.gamma * self.lambda_eligibility

        self.e[action,:] += self._rj_activity(state)

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

            Q_sa = agent._Q_activity(state)[action]
            Q_next_sa = agent._Q_activity(next_state)[next_action]

            td_error = reward - (Q_sa - self.gamma*Q_next_sa)

            self.w += self.eta * td_error * self.e
            self._e_udpdate(state, action)

            state, action = next_state, next_action

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

        self.SARASA = SARSAAgent(mountain_car = mountain_car)


    def run_episode(self, n_steps=200):
        """Do a trial without learning, with display.

        Parameters
        ----------
        n_steps -- number of steps to simulate for
        """

        # Initialize
        # make sure the mountain-car is reset
        self.mountain_car.reset()
        # and so is SARSAAgent Q and e
        self.SARASA._reset_values()

        # get the initial state
        state = self.mountain_car.state()

        # choose an action from the policy:
        action = self.SARASA._next_action(state)
        for n in range(n_steps):
            self.mountain_car.apply_force(action -1)

            # simulate the timestep
            self.mountain_car.simulate_timesteps(100, 0.01)

            next_state = self.mountain_car.state()
            next_action = self.SARASA._next_action(next_state)
            reward = self.mountain_car.R

            Q_sa = self.SARASA._Q_activity(state)[action]
            Q_next_sa = self.SARASA._Q_activity(next_state)[next_action]

            td_error = reward - (Q_sa - self.SARASA.gamma*Q_next_sa)

            self.SARASA.w += self.SARASA.eta * td_error * self.SARASA.e
            self.SARASA._e_udpdate(state, action)

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
    n_agents = 1
    n_steps = 1000
    n_episodes = 500

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
