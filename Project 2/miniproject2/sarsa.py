import sys

import pylab as plb
import numpy as np
import mountaincar

class SARSAAgent():
    """A not so good agent for the mountain-car task.
    """

    def __init__(self, mountain_car = None, N=20, eta = 0.1, gamma=0.99, tau=0.1, lambda_eligibility=0.):
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

        if mountain_car is None:
            self.mountain_car = mountaincar.MountainCar()
        else:
            self.mountain_car = mountain_car

        # initialize the Q-values etc.
        self._init_values(x_centers, dx_centers)

    def _init_values(self, x_centers, dx_centers):
        """
        Initialize the Q-values, eligibility trace, position etc.
        """
        # initialize the Q-values, weights and the eligibility trace
        self.Q = 0.01 * np.random.rand(self.N,self.N,3) + 0.1
        self.w = 0.01 * np.random.rand(3, self.N**2) + 0.1
        self.e = np.zeros((3, self.N**2))

        # list that contains the times it took the agent to reach the target for all trials
        # serves to track the progress of learning
        self.latency_list = []

        # Create grid given the centers
        self.x_grid, self.dx_grid = np.meshgrid(x_centers, dx_centers)

    def _rj_activity(self, state):
        current_x, current_dx = state

        # r_j for all j
        rj = np.exp(-((x_grid - current_x) ** 2) / self.var_x - ((self.dx_grid - current_dx) ** 2) / self.var_dx)

        return rj.flatten() # N*N vector

    def _Q_activity(self, w, state):
        rj = self._rj_activity(state)

        return np.dot(w, rj) # 3 vector

    # def _TD_error(self, current_state, next_state):
    #     R = self.mountain_car.R
    #
    #     delta = R - ( self._Q_activity(self.w, current_state) - self.gamma * self._Q_activity(self.w, next_state) )
    #
    #     return delta_TD

    def _w_update(self):
        delta = self._TD_error()
        delta_w = self.eta * np.multiply(delta, self.e)
        self.w += delta_w

    def _e_udpdate(self, action):
        self.e *= gamma * lambda_eligibility
        self.self.e[:,:,action_idx] += self._rj_activity()


    def _next_action(self, action):




    def visualize_trial(self, n_steps = 200):
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
        next_action = self.next_action(state)

        for n in range(n_steps):
            print '\rtime =', self.mountain_car.t,
            sys.stdout.flush()

            # choose action from the policy
            self.mountain_car.apply_force(next_action - 1)
            # simulate the timestep
            self.mountain_car.simulate_timesteps(100, 0.01)

            # update the visualization
            mv.update_figure()
            plb.draw()

            # check for rewards
            if self.mountain_car.R > 0.0:
                print "\rreward obtained at t = ", self.mountain_car.t
                break

    def learn(self):
        # This is your job!
        pass

if __name__ == "__main__":
    s = SARSAAgent()
    s.visualize_trial()
    plb.show()
