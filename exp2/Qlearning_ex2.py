import numpy as np
import gym

epsilon = 0.1
discountFactor = 1
learningRate = 0.15
tryEpisodes = 1000
maxStep = 1000
times = 10

class Qlearning:
    def __init__(self, sigma):
        self.sigma = sigma
        self.env = gym.make('MountainCar-v0').env
        self.actions = self.env.action_space
        self.num_pos = 20
        self.num_vel = 15
        self.Q = np.random.normal(loc=0, scale=0.01, size=(self.num_pos * self.num_vel, self.actions.n))
        self.pos_bins = self.toBins(-1.2, 0.6, self.num_pos)
        self.vel_bins = self.toBins(-0.07, 0.07, self.num_vel)

    def learn(self):
        countStep = np.zeros(tryEpisodes)
        for i in range(tryEpisodes):
            observation = self.env.reset()
            state = self.digitizeState(observation)
            done=False
            for t in range(maxStep):
                action = self.selectAction(state)
                nextState, reward, done = self.update_state(action)
                tdError = reward + discountFactor * self.getmaxQ(nextState) - self.Q[state, action]
                self.Q[state, action] += learningRate * tdError
                state = nextState
                if done:
                    countStep[i] = t
                    break
            if not done:
                countStep[i] = maxStep
        return countStep

    def selectAction(self, state):
        if np.random.random() < epsilon:
            return self.actions.sample()
        else:
            return np.argmax(self.Q[state])

    def update_state(self, action):
        observation, reward, done, info = self.env.step(action)
        reward = np.random.normal(loc=-1, scale=self.sigma)
        nextState = self.digitizeState(observation)
        return nextState, reward, done

    def getmaxQ(self, nextState):
        return np.max(self.Q[nextState])

    def toBins(self, clip_min, clip_max, num):
        return np.linspace(clip_min, clip_max, num + 1)

    def digit(self, x, bin):
        n = np.digitize(x, bins=bin)
        if x == bin[-1]:
            n = n - 1
        return n

    def digitizeState(self, observation):
        cart_pos, cart_v = observation
        digitized = [self.digit(cart_pos, self.pos_bins),
                     self.digit(cart_v, self.vel_bins), ]
        return (digitized[1] - 1) * self.num_pos + digitized[0] - 1

    def clear(self):
        self.Q = np.random.normal(loc=0, scale=0.01, size=(self.num_pos * self.num_vel, self.actions.n))
